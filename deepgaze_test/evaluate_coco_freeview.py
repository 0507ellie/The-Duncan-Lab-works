"""
Evaluate DeepGaze III on COCO-FreeView (standalone, no pysaliency)

Usage:
    python evaluate_coco_freeview.py --json PATH_TO_JSON --images PATH_TO_COCO_IMAGES
    python evaluate_coco_freeview.py --json PATH --images PATH --num-images 50

The JSON file is COCOFreeView_fixations_trainval.json from:
    https://sites.google.com/view/cocosearch/coco-freeview
The images are standard MS COCO images.

This script evaluates DeepGaze III as a scanpath model with real fixation
history, computing LL, IG, AUC, NSS — all from scratch, no pysaliency.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import torch.nn.functional as F
from tqdm import tqdm

import deepgaze_pytorch
from deepgaze_pytorch.modules import encode_scanpath_features

# ============================================
# Args
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, required=True,
                    help='Path to COCOFreeView_fixations_trainval.json')
parser.add_argument('--images', type=str, required=True,
                    help='Path to COCO images directory')
parser.add_argument('--num-images', type=int, default=None,
                    help='Evaluate on a random subset of N images (default: all)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for image sampling (default: 42)')
parser.add_argument('--split', type=str, default=None, choices=['train', 'val'],
                    help='Only evaluate on this split (default: both)')
args = parser.parse_args()

print("=" * 60)
print("DeepGaze III Evaluation on COCO-FreeView (Standalone)")
print("=" * 60)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ============================================
# 1. Load JSON — parse scanpaths
# ============================================
print(f"\nLoading fixation data from {args.json}...")
with open(args.json, 'r') as f:
    raw_data = json.load(f)

# Filter by split if requested
if args.split:
    raw_data = [entry for entry in raw_data if entry.get('split') == args.split]
    print(f"Filtered to split='{args.split}'")

# Group scanpaths by image name
# Each image has multiple scanpaths (one per subject)
scanpaths_by_image = defaultdict(list)
for entry in raw_data:
    scanpaths_by_image[entry['name']].append({
        'subject': entry['subject'],
        'X': np.array(entry['X'], dtype=np.float64),
        'Y': np.array(entry['Y'], dtype=np.float64),
        'T': np.array(entry['T'], dtype=np.float64),
        'length': entry['length'],
    })

all_image_names = sorted(scanpaths_by_image.keys())
print(f"Total images: {len(all_image_names)}")
print(f"Total scanpaths: {len(raw_data)}")

# Optionally subset
if args.num_images is not None and args.num_images < len(all_image_names):
    rng = np.random.default_rng(args.seed)
    subset_idx = sorted(rng.choice(len(all_image_names), size=args.num_images, replace=False))
    all_image_names = [all_image_names[i] for i in subset_idx]
    print(f"Subsetted to {len(all_image_names)} images (seed={args.seed})")

total_scanpaths = sum(len(scanpaths_by_image[name]) for name in all_image_names)
total_fixations = sum(
    sp['length'] for name in all_image_names for sp in scanpaths_by_image[name]
)
print(f"Scanpaths to evaluate: {total_scanpaths}")
print(f"Total fixations (including initial): {total_fixations}")

# Build image path lookup (handles images in category subfolders)
print("\nBuilding image path index...")
_all_jpgs = glob.glob(os.path.join(args.images, '**', '*.jpg'), recursive=True)
image_path_lookup = {os.path.basename(p): p for p in _all_jpgs}
available = [name for name in all_image_names if name in image_path_lookup]
print(f"Images found on disk: {len(available)} / {len(all_image_names)}")
if len(available) < len(all_image_names):
    print(f"  ({len(all_image_names) - len(available)} images missing, will be skipped)")
all_image_names = available

# ============================================
# 2. Load model + centerbias
# ============================================
print("\nLoading DeepGaze III model...")
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)
model.eval()

centerbias_template = np.load(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'centerbias_mit1003.npy')
)

# ============================================
# 3. Helper: run backbone once per image, cache
# ============================================
def compute_image_features(img_np, centerbias_template, model, device):
    """Run DenseNet backbone + saliency networks once for an image.
    Returns cached tensors for the scanpath-dependent layers."""
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    h, w = img_np.shape[:2]

    # Centerbias
    cb = zoom(
        centerbias_template,
        (h / centerbias_template.shape[0], w / centerbias_template.shape[1]),
        order=0, mode='nearest',
    )
    cb -= logsumexp(cb)
    cb_tensor = torch.tensor([cb], dtype=torch.float32).to(device)

    # Also keep a numpy version for the centerbias baseline
    cb_log_density = cb.copy()

    # Backbone
    img_tensor = torch.tensor(
        [img_np.transpose(2, 0, 1)], dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        orig_shape = img_tensor.shape
        x = F.interpolate(
            img_tensor,
            scale_factor=1 / model.downsample,
            recompute_scale_factor=False,
        )
        x = model.features(x)
        rs = [
            math.ceil(orig_shape[2] / model.downsample / model.readout_factor),
            math.ceil(orig_shape[3] / model.downsample / model.readout_factor),
        ]
        x = [F.interpolate(item, rs) for item in x]
        x = torch.cat(x, dim=1)

        sal_outputs = [sn(x) for sn in model.saliency_networks]

    return {
        'orig_shape': orig_shape,
        'readout_shape': rs,
        'centerbias': cb_tensor,
        'cb_log_density': cb_log_density,
        'saliency_outputs': sal_outputs,
        'height': h,
        'width': w,
    }


# ============================================
# 4. Helper: predict log density for one fixation
# ============================================
def predict_log_density(cache, x_hist_np, y_hist_np, model, device):
    """Run scanpath-dependent layers given cached image features."""
    orig_shape = cache['orig_shape']
    rs = cache['readout_shape']

    # Handle empty history (initial fixation)
    if len(x_hist_np) == 0:
        x_hist_np = np.array([orig_shape[3] / 2.0])
        y_hist_np = np.array([orig_shape[2] / 2.0])

    # Select last 4 fixations, pad with NaN if fewer
    _xh, _yh = [], []
    for idx in model.included_fixations:  # [-1, -2, -3, -4]
        try:
            _xh.append(float(x_hist_np[idx]))
            _yh.append(float(y_hist_np[idx]))
        except IndexError:
            _xh.append(np.nan)
            _yh.append(np.nan)

    xh = torch.tensor([_xh], dtype=torch.float32).to(device)
    yh = torch.tensor([_yh], dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = []
        for i, (scan_net, fsel_net, finalizer) in enumerate(zip(
            model.scanpath_networks,
            model.fixation_selection_networks,
            model.finalizers,
        )):
            sal = cache['saliency_outputs'][i]
            sf = encode_scanpath_features(
                xh, yh,
                size=(orig_shape[2], orig_shape[3]),
                device=sal.device,
            )
            sf = F.interpolate(sf, rs)
            y = scan_net(sf)
            out = fsel_net((sal, y))
            out = finalizer(out, cache['centerbias'])
            predictions.append(out[:, np.newaxis, :, :])

        preds = torch.cat(predictions, dim=1) - np.log(len(model.saliency_networks))
        log_density = preds.logsumexp(dim=1, keepdim=True)

    return log_density.cpu().numpy()[0, 0]


# ============================================
# 5. Helper: compute metrics for one fixation
# ============================================
def compute_fixation_metrics(log_density, cb_log_density, fix_x, fix_y, h, w, rng):
    """Compute LL, IG, AUC, NSS for a single fixation.

    Args:
        log_density: (H, W) predicted log probability map
        cb_log_density: (H, W) centerbias log probability map
        fix_x, fix_y: actual fixation coordinates (float)
        h, w: image height, width
        rng: numpy random generator (for AUC negative sampling)

    Returns:
        dict with ll_nats, ig_nats, auc, nss
    """
    # Clamp to valid pixel range
    fy = int(np.clip(np.round(fix_y), 0, h - 1))
    fx = int(np.clip(np.round(fix_x), 0, w - 1))

    # LL: log probability at fixation location (in nats)
    ll_nats = float(log_density[fy, fx])

    # Centerbias LL at fixation
    cb_ll_nats = float(cb_log_density[fy, fx])

    # IG: model LL - centerbias LL (in nats, convert to bits later)
    ig_nats = ll_nats - cb_ll_nats

    # Uniform LL for this image size
    uniform_ll = -np.log(h * w)

    # AUC: compare saliency at fixation vs 100 random locations
    saliency = np.exp(log_density)
    pos_val = saliency[fy, fx]
    n_neg = 100
    neg_ys = rng.integers(0, h, size=n_neg)
    neg_xs = rng.integers(0, w, size=n_neg)
    neg_vals = saliency[neg_ys, neg_xs]
    auc = float(np.mean(pos_val > neg_vals) + 0.5 * np.mean(pos_val == neg_vals))

    # NSS: normalized saliency at fixation
    saliency_norm = (saliency - saliency.mean()) / (saliency.std() + 1e-12)
    nss = float(saliency_norm[fy, fx])

    return {
        'll_nats': ll_nats,
        'uniform_ll': uniform_ll,
        'ig_nats': ig_nats,
        'auc': auc,
        'nss': nss,
    }


# ============================================
# 6. Main evaluation loop
# ============================================
print("\n" + "=" * 60)
print("Evaluating (per-fixation with real scanpath history)...")
print("=" * 60)

all_metrics = []  # one dict per non-initial fixation
eval_rng = np.random.default_rng(args.seed)
n_initial_skipped = 0

for img_idx, img_name in enumerate(tqdm(all_image_names, desc="Images")):
    # Load image
    img_path = image_path_lookup[img_name]

    img_np = np.array(Image.open(img_path).convert('RGB'))

    # Run backbone once for this image
    cache = compute_image_features(img_np, centerbias_template, model, DEVICE)

    # Process each scanpath on this image
    for sp in scanpaths_by_image[img_name]:
        xs = sp['X']
        ys = sp['Y']

        for fix_idx in range(sp['length']):
            # Skip initial fixation (no history to condition on)
            if fix_idx == 0:
                n_initial_skipped += 1
                continue

            # Build history: all fixations before this one
            x_hist = xs[:fix_idx]
            y_hist = ys[:fix_idx]

            # Predict
            log_density = predict_log_density(cache, x_hist, y_hist, model, DEVICE)

            # Score
            metrics = compute_fixation_metrics(
                log_density, cache['cb_log_density'],
                xs[fix_idx], ys[fix_idx],
                cache['height'], cache['width'],
                eval_rng,
            )
            all_metrics.append(metrics)

print(f"\nInitial fixations skipped: {n_initial_skipped}")
print(f"Fixations evaluated: {len(all_metrics)}")

# ============================================
# 7. Aggregate metrics
# ============================================
ll_nats = np.array([m['ll_nats'] for m in all_metrics])
uniform_lls = np.array([m['uniform_ll'] for m in all_metrics])
ig_nats = np.array([m['ig_nats'] for m in all_metrics])
aucs = np.array([m['auc'] for m in all_metrics])
nsss = np.array([m['nss'] for m in all_metrics])

ll_bits = np.mean((ll_nats - uniform_lls) / np.log(2))
ig_bits = np.mean(ig_nats / np.log(2))
auc = np.mean(aucs)
nss = np.mean(nsss)

# ============================================
# 8. Results
# ============================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\n  LL (bits/fix, over uniform):     {ll_bits:.3f}")
print(f"  IG (bits/fix, over centerbias):   {ig_bits:.3f}")
print(f"  AUC:                              {auc:.3f}")
print(f"  NSS:                              {nss:.3f}")

print("\n" + "=" * 60)
print("Note:")
print("  - DeepGaze III was trained on MIT1003, this is out-of-domain")
print("  - We use MIT1003 centerbias (no COCO-FreeView-specific centerbias)")
print("  - AUC uses 100 random negative samples per fixation")
print("=" * 60)

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_results_coco_freeview.npz')
np.savez(
    out_path,
    ll_nats=ll_nats,
    ll_bits=ll_bits,
    ig_bits=ig_bits,
    auc=auc,
    nss=nss,
    aucs=aucs,
    nsss=nsss,
)
print(f"\nResults saved to {out_path}")
