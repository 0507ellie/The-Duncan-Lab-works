"""
Evaluate DeepGaze III on CAT2000 using pysaliency

Expected results from paper (Table 4):
- LL (log-likelihood): 3.064 bit/fixation
- IG (information gain): 1.625 bit/fixation
- AUC: 0.932
- NSS: 5.106

Note: Paper results are from models trained on CAT2000 with cross-validation.
We evaluate the single pretrained model (trained on MIT1003), so results
will differ. This is an out-of-domain evaluation.

Key: DeepGaze III is a SCANPATH model. It must be evaluated per-fixation
using real scanpath history, not as a static saliency model.
We wrap it as pysaliency.ScanpathModel (not pysaliency.Model).
"""

import argparse
import math
import os
import numpy as np
import pysaliency
from pysaliency.saliency_map_models import ScanpathSaliencyMapModel
from pysaliency.utils import remove_trailing_nans, MatlabOptions

# CAT2000 requires MATLAB (not Octave) because allFixData.mat uses
# containers.Map which only MATLAB can read.
# If MATLAB R2025b doesn't work, you may need to extract manually.

import deepgaze_pytorch
from deepgaze_pytorch.modules import encode_scanpath_features
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy.special import logsumexp

parser = argparse.ArgumentParser()
parser.add_argument('--num-images', type=int, default=None,
                    help='Evaluate on a random subset of N images (default: all 2000)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for image sampling (default: 42)')
args = parser.parse_args()

print("=" * 60)
print("DeepGaze III Evaluation on CAT2000 (Scanpath Model)")
print("=" * 60)

# ============================================
# Setup
# ============================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ============================================
# Load Dataset
# ============================================
DATASET_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
print("\nLoading CAT2000 dataset...")
stimuli, fixations = pysaliency.external_datasets.get_cat2000_train(location=DATASET_LOCATION)
print(f"Loaded {len(stimuli)} images")
print(f"Total fixations: {len(fixations.x)}")
print(f"Unique subjects: {len(np.unique(fixations.subjects))}")

# Optionally subset to N random images
if args.num_images is not None and args.num_images < len(stimuli):
    rng = np.random.default_rng(args.seed)
    subset_ids = sorted(rng.choice(len(stimuli), size=args.num_images, replace=False))
    id_map = {old: new for new, old in enumerate(subset_ids)}
    subset_set = set(subset_ids)
    fix_mask = np.array([n in subset_set for n in fixations.n])
    fixations = fixations[fix_mask]
    fixations.n = np.array([id_map[n] for n in fixations.n])
    stimuli = pysaliency.Stimuli([stimuli.stimuli[i] for i in subset_ids])
    print(f"Subsetted to {len(stimuli)} images (seed={args.seed})")
    print(f"Fixations after subsetting: {len(fixations.x)}")

# Identify non-initial fixations (those with at least one preceding fixation).
non_initial_mask = np.array([
    len(remove_trailing_nans(fixations.x_hist[i])) > 0
    for i in range(len(fixations.x))
])
print(f"Non-initial fixations (evaluation targets): {non_initial_mask.sum()}")

# ============================================
# Load Model
# ============================================
print("\nLoading DeepGaze III model...")
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)
model.eval()

# ============================================
# Load centerbias
# ============================================
centerbias_template = np.load(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'centerbias_mit1003.npy')
)


# ============================================
# Scanpath model wrapper
# ============================================
class DeepGazeIIIScanpathModel(pysaliency.ScanpathModel, ScanpathSaliencyMapModel):
    """Wrap DeepGaze III as a pysaliency ScanpathModel.

    Caches DenseNet features + saliency network outputs per image so that
    only the lightweight scanpath-dependent layers run per fixation.
    """

    def __init__(self, torch_model, centerbias_template, device):
        self.torch_model = torch_model
        self.centerbias_template = centerbias_template
        self.device = device
        self._feature_cache = {}

    # ------------------------------------------------------------------
    # Per-image feature caching (backbone + saliency networks)
    # ------------------------------------------------------------------
    def _get_features(self, stimulus):
        key = id(stimulus)
        if key in self._feature_cache:
            return self._feature_cache[key]

        img = np.array(stimulus.stimulus_data)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w = img.shape[:2]

        # Centerbias
        cb = zoom(
            self.centerbias_template,
            (h / self.centerbias_template.shape[0],
             w / self.centerbias_template.shape[1]),
            order=0, mode='nearest',
        )
        cb -= logsumexp(cb)
        cb_tensor = torch.tensor([cb], dtype=torch.float32).to(self.device)

        # Backbone features (the expensive part)
        img_tensor = torch.tensor(
            [img.transpose(2, 0, 1)], dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            orig_shape = img_tensor.shape
            x = F.interpolate(
                img_tensor,
                scale_factor=1 / self.torch_model.downsample,
                recompute_scale_factor=False,
            )
            x = self.torch_model.features(x)
            rs = [
                math.ceil(orig_shape[2] / self.torch_model.downsample
                          / self.torch_model.readout_factor),
                math.ceil(orig_shape[3] / self.torch_model.downsample
                          / self.torch_model.readout_factor),
            ]
            x = [F.interpolate(item, rs) for item in x]
            x = torch.cat(x, dim=1)

            # Saliency network outputs (also image-only)
            sal_outputs = [sn(x) for sn in self.torch_model.saliency_networks]

        cache = {
            'orig_shape': orig_shape,
            'readout_shape': rs,
            'centerbias': cb_tensor,
            'saliency_outputs': sal_outputs,
        }
        self._feature_cache[key] = cache
        return cache

    # ------------------------------------------------------------------
    # Core prediction (runs only scanpath-dependent layers)
    # ------------------------------------------------------------------
    def _predict(self, stimulus, x_hist_np, y_hist_np):
        cache = self._get_features(stimulus)
        orig_shape = cache['orig_shape']
        rs = cache['readout_shape']

        if len(x_hist_np) == 0:
            x_hist_np = np.array([orig_shape[3] / 2.0])
            y_hist_np = np.array([orig_shape[2] / 2.0])

        _xh, _yh = [], []
        for idx in self.torch_model.included_fixations:
            try:
                _xh.append(float(x_hist_np[idx]))
                _yh.append(float(y_hist_np[idx]))
            except IndexError:
                _xh.append(np.nan)
                _yh.append(np.nan)

        xh = torch.tensor([_xh], dtype=torch.float32).to(self.device)
        yh = torch.tensor([_yh], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = []
            for i, (scan_net, fsel_net, finalizer) in enumerate(zip(
                self.torch_model.scanpath_networks,
                self.torch_model.fixation_selection_networks,
                self.torch_model.finalizers,
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

            preds = torch.cat(predictions, dim=1) - np.log(
                len(self.torch_model.saliency_networks)
            )
            log_density = preds.logsumexp(dim=1, keepdim=True)

        return log_density.cpu().numpy()[0, 0]

    # ------------------------------------------------------------------
    # pysaliency ScanpathModel interface
    # ------------------------------------------------------------------
    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist,
                                attributes=None, out=None):
        return self._predict(stimulus, x_hist, y_hist)

    # ------------------------------------------------------------------
    # pysaliency ScanpathSaliencyMapModel interface (for AUC / NSS)
    # ------------------------------------------------------------------
    def conditional_saliency_map(self, stimulus, x_hist, y_hist, t_hist,
                                 attributes=None, out=None):
        return np.exp(self.conditional_log_density(
            stimulus, x_hist, y_hist, t_hist))


# ============================================
# Center-bias baseline (for Information Gain)
# ============================================
class CenterBiasModel(pysaliency.Model):
    """Static center-bias model used as IG baseline."""

    def __init__(self, centerbias_template):
        super().__init__()
        self.centerbias_template = centerbias_template

    def _log_density(self, stimulus):
        img = stimulus
        h, w = img.shape[:2]
        cb = zoom(
            self.centerbias_template,
            (h / self.centerbias_template.shape[0],
             w / self.centerbias_template.shape[1]),
            order=0, mode='nearest',
        )
        cb -= logsumexp(cb)
        return cb


# ============================================
# Create models
# ============================================
wrapped_model = DeepGazeIIIScanpathModel(model, centerbias_template, DEVICE)
centerbias_model = CenterBiasModel(centerbias_template)

print("\nModel wrapped as pysaliency ScanpathModel")
print("(DenseNet features cached per image; only scanpath layers run per fixation)")

# ============================================
# Compute metrics
# ============================================
print("\n" + "=" * 60)
print("Computing metrics (per-fixation with real scanpath history)...")
print("This will take a while on first run.")
print("=" * 60)

# 1. Log-likelihoods (one value per fixation, in nats)
print("\n[1/4] Computing log-likelihoods for all fixations...")
all_log_lls = wrapped_model.log_likelihoods(stimuli, fixations, verbose=True)

# Filter to non-initial fixations
log_lls = all_log_lls[non_initial_mask]

# 2. LL in bits/fixation (= information gain over uniform)
image_sizes = [stimuli.sizes[n] for n in fixations.n[non_initial_mask]]
uniform_lls = np.array([-np.log(h * w) for h, w in image_sizes])
ll_bits = np.mean((log_lls - uniform_lls) / np.log(2))
print(f"\nLL: {ll_bits:.3f} bit/fixation")

# 3. IG in bits/fixation (= information gain over center bias)
print("\n[2/4] Computing center-bias log-likelihoods...")
all_cb_lls = centerbias_model.log_likelihoods(stimuli, fixations, verbose=True)
cb_lls = all_cb_lls[non_initial_mask]
ig_bits = np.mean((log_lls - cb_lls) / np.log(2))
print(f"\nIG: {ig_bits:.3f} bit/fixation")

# 4. AUC (per-fixation using conditional saliency maps)
print("\n[3/4] Computing AUC...")
all_aucs = wrapped_model.AUCs(stimuli, fixations, verbose=True)
auc = np.mean(np.array(all_aucs)[non_initial_mask])
print(f"\nAUC: {auc:.3f}")

# 5. NSS (per-fixation using conditional saliency maps)
print("\n[4/4] Computing NSS...")
all_nsss = wrapped_model.NSSs(stimuli, fixations, verbose=True)
nss = np.mean(np.array(all_nsss)[non_initial_mask])
print(f"\nNSS: {nss:.3f}")

# ============================================
# Results Summary
# ============================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\n{'Metric':<10} {'Ours':>10} {'Paper':>18}")
print("-" * 40)
print(f"{'LL':<10} {ll_bits:>10.3f} {'3.064':>18}")
print(f"{'IG':<10} {ig_bits:>10.3f} {'1.625':>18}")
print(f"{'AUC':<10} {auc:>10.3f} {'0.932':>18}")
print(f"{'NSS':<10} {nss:>10.3f} {'5.106':>18}")

print("\n" + "=" * 60)
print("Note: Differences from paper are expected because:")
print("  - Paper used models trained on CAT2000 (cross-validation)")
print("  - We evaluate the pretrained model (trained on MIT1003)")
print("  - This is an out-of-domain evaluation")
print("  - We use MIT1003 centerbias (no CAT2000-specific centerbias)")
print("=" * 60)

# Save results
np.savez(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_results_cat2000.npz'),
    log_likelihoods=log_lls,
    ll_bits=ll_bits,
    ig_bits=ig_bits,
    auc=auc,
    nss=nss,
    all_log_lls=all_log_lls,
    non_initial_mask=non_initial_mask,
)
print("\nResults saved to evaluation_results_cat2000.npz")
