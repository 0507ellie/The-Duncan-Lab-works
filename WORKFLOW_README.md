# Scanpath Prediction 

Two neural networks are trained against each other:

| Name | Role | Analogy |
|---|---|---|
| **Generator** | Decides where to look next | The student |
| **Discriminator** | Judges whether a sequence of gaze points looks human | The teacher/examiner |

The Generator practises until the Discriminator cannot tell its gaze sequences
apart from real ones. This is the core training loop.

---

## File Order (Training Pipeline)

```
PHASE 0 — Pre-processing (done once, before training)
  resize.py
  extract_DCBs_demo.py

PHASE 1 — Setup (runs at the start of every training session)
  hparams/coco_search18.json
  irl_dcb/config.py
  dataset.py
  irl_dcb/data.py
  irl_dcb/utils.py
  irl_dcb/models.py
  irl_dcb/environment.py
  irl_dcb/builder.py

PHASE 2 — Training Loop
  train.py               ← entry point
  irl_dcb/trainer.py     ← master loop
  irl_dcb/gail.py        ← trains the Discriminator
  irl_dcb/ppo.py         ← trains the Generator

PHASE 3 — Evaluation & Metrics
  irl_dcb/metrics.py
  irl_dcb/multimatch.py

PHASE 4 — Inference & Visualisation
  test_single_image.py
  plot_scanpath.py
```

---

---

# PHASE 0 — Pre-processing

These two scripts are run **once** before any training begins, to convert raw
images into the compact numerical format the model needs.

---

## `resize.py`

**Purpose:** Shrink each raw image to the fixed size the model expects
(512 pixels wide × 320 pixels tall).

```python
from PIL import Image
```
Import the Python image library.

```python
img = Image.open("test_office.jpg").resize((512, 320))
```
Open the image file from disk. Call `.resize((512, 320))` to scale it down
(or up) so that it is exactly 512 pixels wide and 320 pixels tall. The model
was trained on this size, so every image must match it.

```python
img.save("test_office_resized.jpg")
```
Write the resized image back to disk under a new filename so the original is
not overwritten.

---

## `extract_DCBs_demo.py`

**Purpose:** For every image, pre-compute two "belief maps" — one sharp, one
blurry — that encode where different kinds of objects are located. These maps
are what the model actually reads; it never looks at raw pixel colours.

A **belief map** (DCB = Dynamic Contextual Belief) is a compact grid of
numbers. There are 134 layers per image (80 object types + 54 background
region types from the COCO dataset). Each layer is a 20×32 grid where each
cell says "how strongly is this object type present in this region of the
image?"

```python
import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageFilter
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
```
Import all required libraries. `detectron2` is Facebook's pre-trained object
detection library. `PIL` handles image files.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Automatically pick the GPU if one is available, otherwise use the CPU.
(GPU is much faster for this kind of computation.)

---

### Function `pred2feat(seg, info)`

Converts the raw output of the object detector into the compact 134-layer
grid format.

```python
def pred2feat(seg, info):
    seg = seg.cpu()
```
Move the detector output from GPU memory to normal CPU memory so it can be
processed by numpy.

```python
    feat = torch.zeros([80 + 54, 320, 512])
```
Create an empty grid: 134 layers (channels), 320 rows, 512 columns. All
values start at zero.

```python
    for pred in info:
        mask = (seg == pred['id']).float()
```
For each detected object/region (`pred`), create a mask — a grid of 1s and 0s
where 1 means "this object is here" and 0 means "it is not here".

```python
        if pred['isthing']:
            feat[pred['category_id'], :, :] = mask * pred['score']
```
If this is a "thing" (a countable object like a person or a chair), store the
mask in the corresponding layer, multiplied by the detector's confidence score.
For example, if the detector is 90% sure there is a laptop here, the laptop
layer gets 0.9 in those cells.

```python
        else:
            feat[pred['category_id'] + 80, :, :] = mask
```
If this is "stuff" (uncountable background like sky, floor, grass), store the
mask in layers 80–133. No confidence score is used for stuff — it is either
there or it is not.

```python
    return F.interpolate(feat.unsqueeze(0), size=[20, 32]).squeeze(0)
```
Shrink the 134 × 320 × 512 grid down to 134 × 20 × 32 using bilinear
interpolation (a smooth averaging method). This smaller size is what the model
actually uses — it is fast to process and still captures the spatial layout.

---

### Function `get_DCBs(img_path, predictor, radius=1)`

Produces the sharp (HR) and blurry (LR) belief maps for one image.

```python
def get_DCBs(img_path, predictor, radius=1):
    high = Image.open(img_path).convert('RGB').resize((512, 320))
```
Open the image, convert to RGB (in case it is grayscale or RGBA), and resize
to the standard 512×320.

```python
    low = high.filter(ImageFilter.GaussianBlur(radius=radius))
```
Create a blurred copy of the image using a Gaussian blur. This simulates
peripheral vision — things in your peripheral field look fuzzy and lack fine
detail. The `radius=1` controls how blurry: 1 is a mild blur.

```python
    high_panoptic_seg, high_segments_info = predictor(np.array(high))["panoptic_seg"]
    low_panoptic_seg,  low_segments_info  = predictor(np.array(low)) ["panoptic_seg"]
```
Run the object detector on both the sharp image and the blurry image. The
detector returns two things for each:
- a segmentation map (every pixel labelled with an object ID)
- a list of objects it found with their IDs, categories, and confidence scores

```python
    high_feat = pred2feat(high_panoptic_seg, high_segments_info)
    low_feat  = pred2feat(low_panoptic_seg,  low_segments_info)
    return high_feat, low_feat
```
Convert both detector outputs to the 134×20×32 format and return them.

---

### Main block

```python
if __name__ == '__main__':
    img_path = "/home/ellie/Scanpath_Prediction/test_office_resized.jpg"
    out_dir  = "my_data/DCBs"
```
Set the input image path and where to save the output files. These two lines
are the only ones a user needs to edit.

```python
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
```
Load the pre-trained Panoptic FPN object detector. `get_cfg()` creates a
configuration object. `merge_from_file` loads the model architecture
definition. `MODEL.WEIGHTS` points to the pre-trained weights file (downloaded
automatically ~240 MB on first run). `DefaultPredictor` wraps the whole thing
into a convenient callable object.

```python
    high_feat, low_feat = get_DCBs(img_path, predictor)
    print("HR shape:", high_feat.shape)   # torch.Size([134, 20, 32])
    print("LR shape:", low_feat.shape)    # torch.Size([134, 20, 32])
```
Actually compute the two belief maps and print their shapes to confirm they
are correct (should be 134 × 20 × 32 in both cases).

```python
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(f"{out_dir}/HR", exist_ok=True)
    os.makedirs(f"{out_dir}/LR", exist_ok=True)
    np.save(f"{out_dir}/HR/{img_name}.npy", high_feat.numpy())
    np.save(f"{out_dir}/LR/{img_name}.npy", low_feat.numpy())
```
Extract just the filename (without path or extension), create the output
folders if they do not exist, then save the sharp belief map under `HR/` and
the blurry belief map under `LR/` as `.npy` files (numpy binary format).

---

---

# PHASE 1 — Setup

---

## `hparams/coco_search18.json`

**Purpose:** Stores all the hyper-parameters (settings) for the model and
training run in one place. Changing these numbers changes how the model
behaves without touching any code.

```json
"Data": {
    "im_w": 512,          // image width in pixels
    "im_h": 320,          // image height in pixels
    "patch_num": [32, 20],// grid is 32 columns × 20 rows = 640 cells total
    "patch_size": [16,16],// each grid cell covers 16×16 pixels
    "patch_count": 640,   // total number of grid cells (32 × 20)
    "fovea_radius": 2,    // radius (in grid cells) of the foveal spotlight
    "IOR_size": 1,        // inhibition-of-return radius (cells around a fixation that get blocked)
    "max_traj_length": 6  // maximum number of fixations per scanpath
}
```

```json
"Train": {
    "gamma": 0.9,              // discount factor: future rewards worth 90% of immediate ones
    "adv_est": "GAE",          // advantage estimation method (Generalised Advantage Estimation)
    "exclude_wrong_trials": false, // keep scanpaths even if the person failed to find the target
    "tau": 0.96,               // GAE smoothing parameter
    "batch_size": 128,         // number of images processed at once
    "stop_criteria": "SOT",    // "Stop On Target" — scanpath ends when target is found
    "log_root": "./assets",    // where to save logs and checkpoints
    "num_epoch": 30,           // number of full passes through the training data
    "num_step": 4,             // how many rollouts to collect before each Discriminator update
    "checkpoint_every": 100,   // save model weights every 100 training steps
    "max_checkpoints": 5,      // keep at most 5 saved checkpoints on disk
    "evaluate_every": 20,      // run validation evaluation every 20 steps
    "num_critic": 1,           // update Generator every 1 Discriminator update
    "gail_milestones": [10000],// reduce Discriminator learning rate at step 10000
    "gail_lr": 5e-05,          // Discriminator learning rate (0.00005)
    "adam_betas": [0.9, 0.999] // Adam optimiser momentum parameters
}
```

```json
"PPO": {
    "lr": 1e-05,         // Generator learning rate (0.00001)
    "clip_param": 0.2,   // PPO clipping: policy cannot change by more than 20% in one step
    "num_epoch": 1,      // number of passes over each batch of rollouts
    "batch_size": 64,    // mini-batch size for PPO updates
    "value_coef": 1.0,   // how much weight to give the value (critic) loss
    "entropy_coef": 0.01 // encourage randomness/exploration (prevents early convergence)
}
```

---

## `irl_dcb/config.py`

**Purpose:** Load the JSON settings file and make its contents accessible
throughout the code using dot notation (e.g. `hparams.Data.im_w`).

```python
import os, json, datetime
```
Standard libraries: `os` for file paths, `json` for reading JSON files,
`datetime` for timestamping saved logs.

```python
class JsonConfig(dict):
```
Define a class that behaves like a Python dictionary but also allows dot-access
(so you can write `config.Data.im_w` instead of `config["Data"]["im_w"]`).

```python
    def __init__(self, *argv, **kwargs):
        super().__init__()
        super().__setitem__("__name", "default")
```
Initialise the underlying dictionary and store a default name `"default"`.

```python
        if isinstance(arg, str):
            super().__setitem__("__name",
                                os.path.splitext(os.path.basename(arg))[0])
            with open(arg, "r") as load_f:
                arg = json.load(load_f)
```
If a file path string is passed in, extract the filename (without extension) as
the config name, then open and parse the JSON file into a Python dictionary.

```python
        if isinstance(arg, dict):
            for key in arg:
                value = arg[key]
                if isinstance(value, dict):
                    value = JsonConfig(value)
                super().__setitem__(key, value)
```
Walk through every key-value pair. If a value is itself a dictionary (a
nested section like `"Data": {...}`), recursively wrap it in another
`JsonConfig`. This is what allows writing `config.Data.im_w` — each level is
its own `JsonConfig` object.

```python
    def __setattr__(self, attr, value):
        raise Exception("[JsonConfig]: Can't set constant key {}".format(attr))
    def __setitem__(self, item, value):
        raise Exception("[JsonConfig]: Can't set constant key {}".format(item))
```
Make the config read-only. Attempting to change any value after loading will
raise an error. This prevents accidental modification of settings during
training.

```python
    def __getattr__(self, attr):
        return super().__getitem__(attr)
```
Allow dot-access: `config.Data` looks up the key `"Data"` in the dictionary.

```python
    def dump(self, dir_path, json_name=None):
        ...
        json.dump(self.to_dict(), fout, indent=JsonConfig.Indent)
```
Save a copy of the current settings to the log folder at the start of
training. This ensures there is always a record of exactly which settings
were used for each training run.

---

## `dataset.py`

**Purpose:** Load all raw data (belief maps + human eye-tracking recordings)
and organise it into structured datasets that the training loop can iterate
over.

```python
import numpy as np
from irl_dcb.data import LHF_IRL, LHF_Human_Gaze
from irl_dcb.utils import compute_search_cdf, preprocess_fixations
```
Import the two custom dataset classes and two utility functions defined
elsewhere in the project.

---

### Function `process_data(...)`

```python
def process_data(trajs_train, trajs_valid, DCB_HR_dir, DCB_LR_dir,
                 target_annos, hparams, is_testing=False):
```
The main data-preparation function. It takes:
- `trajs_train` / `trajs_valid`: lists of human scanpath dictionaries (loaded
  from the JSON files in `train.py`)
- `DCB_HR_dir` / `DCB_LR_dir`: folder paths to the pre-computed belief maps
- `target_annos`: bounding boxes for every target object in every image
- `hparams`: the loaded settings object

```python
    target_init_fixs = {}
    for traj in trajs_train + trajs_valid:
        key = traj['task'] + '_' + traj['name']
        target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                 traj['Y'][0] / hparams.Data.im_h)
```
Loop over every scanpath in both splits. Build a dictionary that maps
`"category_imagename"` → the normalised (x, y) coordinates of the very first
fixation in that scanpath. Normalised means divided by image width/height so
the value is between 0 and 1, which is resolution-independent.
This first fixation is used to initialise each scanpath at the same starting
point as the real human.

```python
    cat_names = list(np.unique([x['task'] for x in trajs_train]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))
```
Collect all unique target category names (e.g. `["bottle", "bowl", "car",
...]`). Assign each a number starting from 0. For example:
`{"bottle": 0, "bowl": 1, "car": 2, ...}`. This numeric ID is what the
neural networks actually use — they cannot process text directly.

```python
    train_task_img_pair = np.unique(
        [traj['task'] + '_' + traj['name'] for traj in trajs_train])
```
Build the list of unique `"category_imagename"` pairs in the training set.
Multiple human subjects may have scanned the same image looking for the same
target, so `np.unique` removes duplicates — each image+category pair should
only appear once.

```python
    train_fix_labels = preprocess_fixations(
        trajs_train,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        truncate_num=hparams.Data.max_traj_length)
```
Convert the raw (x, y) pixel coordinates in each human scanpath into
discrete grid-cell indices and build the training labels for the
Discriminator. Each label is a tuple of:
`(image_name, category_name, list_of_previous_fixations, next_grid_cell)`.
The function also enforces inhibition of return (no revisiting cells) and
truncates scanpaths to at most `max_traj_length` = 6 fixations.
(See `utils.py` → `preprocess_fixations` for the step-by-step detail.)

```python
    valid_task_img_pair = np.unique(...)
    human_mean_cdf, _ = compute_search_cdf(trajs_valid, target_annos,
                                           hparams.Data.max_traj_length)
    print('target fixation prob (valid).:', human_mean_cdf)
```
Same process for the validation set. Additionally, compute the **human
search efficiency curve**: for each step 1 through 6, what fraction of
real humans had found the target by that step? This is the gold-standard
benchmark. During training, the model's efficiency will be compared against
this curve.

```python
    train_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                train_task_img_pair, target_annos,
                                hparams.Data, catIds)
    valid_img_dataset = LHF_IRL(...)
```
Create image-dataset objects for training and validation. These are used by
the Generator: given an image+category, they load the belief maps and set
up the starting state.

```python
    train_HG_dataset = LHF_Human_Gaze(DCB_HR_dir, DCB_LR_dir, train_fix_labels,
                                      target_annos, hparams.Data, catIds)
    valid_HG_dataset  = LHF_Human_Gaze(..., blur_action=True)
```
Create human-gaze datasets for training and validation. These are used by the
Discriminator: each example reconstructs the belief-map state at a given step
of a real human scanpath and provides the next gaze location as the ground
truth. The validation version blurs the action label (spreads the probability
over neighbouring cells) to make evaluation less strict.

```python
    return {
        'catIds': catIds,
        'img_train': train_img_dataset,
        'img_valid': valid_img_dataset,
        'gaze_train': train_HG_dataset,
        'gaze_valid': valid_HG_dataset,
        'human_mean_cdf': human_mean_cdf,
        'bbox_annos': target_annos
    }
```
Return everything bundled into a single dictionary so `train.py` can pass it
to the `Trainer`.

---

## `irl_dcb/data.py`

**Purpose:** Defines the four data-container classes used during training.

---

### Class `LHF_IRL` — image data for the Generator

```python
class LHF_IRL(Dataset):
    def __init__(self, DCB_HR_dir, DCB_LR_dir, initial_fix, img_info,
                 annos, pa, catIds):
```
Store all the paths and metadata. `pa` is the `Data` section of the settings.
`catIds` is the name→number mapping.

```python
    def __getitem__(self, idx):
        cat_name, img_name = self.img_info[idx].split('_')
        feat_name = img_name[:-3] + 'pth.tar'
        lr_path = join(self.LR_dir, cat_name.replace(' ', '_'), feat_name)
        hr_path = join(self.HR_dir, cat_name.replace(' ', '_'), feat_name)
        lr = torch.load(lr_path)
        hr = torch.load(hr_path)
```
When the training loop asks for item number `idx`, split the
`"category_imagename"` string into its two parts. Build file paths for the
HR and LR belief map files. Load both from disk as PyTorch tensors (each is
134 × 20 × 32).

```python
        init_fix = self.initial_fix[imgId]
        px, py = init_fix
        px, py = px * lr.size(-1), py * lr.size(-2)
        mask = utils.foveal2mask(px, py, self.pa.fovea_radius, ...)
        mask = torch.from_numpy(mask).unsqueeze(0).repeat(hr.size(0), 1, 1)
        lr = (1 - mask) * lr + mask * hr
```
Look up where the first human fixation was for this image+category. Convert
from normalised coordinates (0–1) to grid coordinates. Create a circular
spotlight mask at that location. Blend: cells inside the spotlight get the
HR (sharp) values; cells outside keep the LR (blurry) values. This is the
starting state — it simulates what the person would have perceived at the very
beginning of their search.

```python
        history_map = torch.zeros((hr.size(-2), hr.size(-1)))
        history_map = (1 - mask[0]) * history_map + mask[0] * 1
```
Create a blank history map (all zeros). Mark the initial fixation region as
visited (set to 1). This map is a record of "where has been looked at so far".

```python
        action_mask = torch.zeros((self.pa.patch_num[1], self.pa.patch_num[0]),
                                  dtype=torch.uint8)
        px, py = int(px * ...), int(py * ...)
        action_mask[py - IOR_size : py + IOR_size + 1,
                    px - IOR_size : px + IOR_size + 1] = 1
```
Create the action mask — a grid of blocked cells. Mark the region around the
initial fixation as 1 = "already visited, cannot choose this cell again".
This is inhibition of return (IOR).

```python
        coding = utils.multi_hot_coding(self.annos[imgId], ...)
        coding = torch.from_numpy(coding).view(1, -1)
```
Create the target-location label: a flat 640-element vector where cells that
overlap with the target object's bounding box are 1, and all other cells are 0.
This tells the environment when the Generator has "found" the target.

```python
        return {'task_id': ..., 'img_name': ..., 'cat_name': ...,
                'lr_feats': lr, 'hr_feats': hr, 'history_map': ...,
                'init_fix': ..., 'label_coding': coding, 'action_mask': ...}
```
Return all of the above as a dictionary. One dictionary per image = one item
in a batch.

---

### Class `LHF_Human_Gaze` — human gaze data for the Discriminator

```python
class LHF_Human_Gaze(Dataset):
    def __getitem__(self, idx):
        img_name, cat_name, fixs, action = self.fix_labels[idx]
```
Each item in this dataset is one step from a real human scanpath. `fixs` is
the list of all previous fixation positions (history); `action` is the
grid-cell index of the next fixation.

```python
        state = torch.load(lr_path)   # start from the blurry LR belief map
        hr    = torch.load(hr_path)
        history_map = torch.zeros(...)
        for i in range(len(fixs)):
            px, py = fixs[i]
            px, py = px / remap_ratio, py / remap_ratio
            mask = utils.foveal2mask(px, py, ...)
            mask = torch.from_numpy(mask).unsqueeze(0).repeat(...)
            state = (1 - mask) * state + mask * hr
            history_map = (1 - mask[0]) * history_map + mask[0]
```
Replay the human's history. Start with the blurry LR state. For each
previous fixation in order, create a spotlight at that location and blend the
HR detail into the state. By the end of this loop, `state` represents exactly
what the image "looked like" to the model at the current step — all previously
fixated regions are in sharp detail; everything else is still blurry.

```python
        ret = {
            "task_id":     self.catIds[cat_name],
            "true_state":  state,          # belief map state at this step
            "true_action": torch.tensor([action], dtype=torch.long),
            'label_coding': coding,
            'history_map': history_map,
            ...
        }
        if self.blur_action:
            action_map = np.zeros(self.pa.patch_count, dtype=np.float32)
            action_map[action] = 1
            action_map = action_map.reshape(...)
            action_map = filters.gaussian_filter(action_map, sigma=1)
            ret['action_map'] = action_map
```
Return the reconstructed state and the ground-truth next action. If
`blur_action=True` (used for validation), convert the single action index
into a soft 2D probability map by placing a 1 at the correct cell and then
smoothing it with a Gaussian filter (spreads the probability slightly over
neighbouring cells).

---

### Class `RolloutStorage` — packages Generator trajectories for PPO

```python
class RolloutStorage(object):
    def __init__(self, trajs_all, shuffle=True, norm_adv=False):
        self.obs_fovs = torch.cat([traj["curr_states"] for traj in trajs_all])
        self.actions  = torch.cat([traj["actions"]     for traj in trajs_all])
        self.lprobs   = torch.cat([traj['log_probs']   for traj in trajs_all])
        self.tids     = torch.cat([traj['task_id']     for traj in trajs_all])
        self.returns  = torch.cat([traj['acc_rewards'] for traj in trajs_all]).view(-1)
        self.advs     = torch.cat([traj['advantages']  for traj in trajs_all]).view(-1)
```
Concatenate the states, actions, log-probabilities, task IDs, discounted
returns, and advantages from all collected trajectories into single large
tensors. This pools all steps from all scanpaths together.

```python
        if norm_adv:
            self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)
```
Optionally normalise the advantages (subtract mean, divide by standard
deviation). This keeps the numbers in a stable range and makes training
less sensitive to the scale of rewards.

```python
    def get_generator(self, minibatch_size):
        sampler = BatchSampler(SubsetRandomSampler(range(self.sample_num)),
                               minibatch_size, drop_last=True)
        for ind in sampler:
            ...
            yield (obs_fov_batch, tids_batch), actions_batch, return_batch, \
                  log_probs_batch, advantage_batch
```
Each time the PPO loop asks for a batch, randomly sample `minibatch_size`
indices and yield the corresponding slices of data. `drop_last=True` discards
the final batch if it is smaller than `minibatch_size`.

---

### Class `FakeDataRollout` — packages fake trajectories for GAIL

```python
class FakeDataRollout(object):
    def __init__(self, trajs_all, minibatch_size, shuffle=True):
        self.GS   = torch.cat([traj['curr_states'] for traj in trajs_all])
        self.GA   = torch.cat([traj['actions']     for traj in trajs_all]).unsqueeze(1)
        self.tids = torch.cat([traj['task_id']     for traj in trajs_all])
        self.GP   = torch.exp(
                        torch.cat([traj["log_probs"] for traj in trajs_all])).unsqueeze(1)
```
Collect all fake states (GS), fake actions (GA), task IDs, and action
probabilities (GP = exp of log-prob, to get back to the probability itself).
Add an extra dimension to GA and GP with `.unsqueeze(1)`.

```python
    def get_generator(self):
        sampler = BatchSampler(SubsetRandomSampler(range(self.sample_num)),
                               self.batch_size, drop_last=True)
        for ind in sampler:
            yield GS_batch, GA_batch, GP_batch, tid_batch
```
Same random mini-batching as `RolloutStorage`, but yields the four fake-data
quantities needed by the Discriminator training in GAIL.

---

## `irl_dcb/utils.py`

**Purpose:** A collection of helper functions used everywhere in the project.

---

### `foveal2mask(x, y, r, h, w)`

```python
def foveal2mask(x, y, r, h, w):
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    mask = dist <= r
    return mask.astype(np.float32)
```
Create a circular "spotlight" mask of size h × w. `np.ogrid` creates two
arrays — one counting rows 0 to h, one counting columns 0 to w. For every
cell, compute the Euclidean distance from the centre point (x, y). Set cells
within radius `r` to 1 (inside the spotlight), all others to 0. Return as
float values.

---

### `pos_to_action(center_x, center_y, patch_size, patch_num)`

```python
def pos_to_action(center_x, center_y, patch_size, patch_num):
    x = center_x // patch_size[0]
    y = center_y // patch_size[1]
    return int(patch_num[0] * y + x)
```
Convert a pixel coordinate to a flat grid-cell index.
- Integer-divide x by patch width to get column index.
- Integer-divide y by patch height to get row index.
- Combine: `row × num_columns + column`. For a 32×20 grid, cell (col=5,
  row=3) → index 3×32 + 5 = 101.

---

### `action_to_pos(acts, patch_size, patch_num)`

```python
def action_to_pos(acts, patch_size, patch_num):
    patch_y = acts // patch_num[0]
    patch_x = acts % patch_num[0]
    pixel_x = patch_x * patch_size[0] + patch_size[0] / 2
    pixel_y = patch_y * patch_size[1] + patch_size[1] / 2
    return pixel_x, pixel_y
```
Reverse of the above. Given a flat index, recover row and column, then
multiply back by patch size and add half a patch to get the centre pixel of
that cell.

---

### `preprocess_fixations(trajs, patch_size, patch_num, im_h, im_w, truncate_num)`

```python
def preprocess_fixations(...):
    fix_labels = []
    for traj in trajs:
        traj['X'][0], traj['Y'][0] = im_w / 2, im_h / 2
```
For every scanpath, force the first fixation to the exact image centre
(256, 160). In the real data, people's first fixations may vary slightly, but
the model always starts at the centre.

```python
        label = pos_to_action(traj['X'][0], traj['Y'][0], patch_size, patch_num)
        tar_x, tar_y = action_to_pos(label, patch_size, patch_num)
        fixs = [(tar_x, tar_y)]
        label_his = [label]
```
Convert the first fixation to a grid-cell index, then snap it back to the
exact centre of that cell (so everything is on the grid). Start the history
lists with this first fixation.

```python
        for i in range(1, traj_len):
            label = pos_to_action(traj['X'][i], traj['Y'][i], ...)
            if label in label_his:
                continue
```
For each subsequent fixation, convert to a grid cell. If this cell has already
been visited, skip it (inhibition of return — humans rarely go back to the
same location).

```python
            label_his.append(label)
            fix_label = (traj['name'], traj['task'], copy(fixs), label)
            tar_x, tar_y = action_to_pos(label, patch_size, patch_num)
            fixs.append((tar_x, tar_y))
            fix_labels.append(fix_label)
```
Record `(image_name, category, history_of_fixations_so_far, next_cell)` as
one training example. Then append the new cell to the history. Note
`copy(fixs)` — a snapshot of the current history is stored, not a reference,
so it does not change as more fixations are added.

```python
    return fix_labels
```
Return the full list of training examples. Each entry is one step of one
human's scanpath, reformatted for the Discriminator.

---

### `select_action(obs, policy, sample_action, action_mask)`

```python
def select_action(obs, policy, sample_action, action_mask=None, ...):
    probs, values = policy(*obs)
```
Feed the current image state and task ID through the Generator. Get back:
- `probs`: a 640-element probability distribution over all grid cells
- `values`: the Generator's estimate of how good this state is (used by PPO)

```python
    if sample_action:
        probs_new = probs.clone().detach()
        probs_new[action_mask.bool()] = eps     # zero out visited cells
        probs_new /= probs_new.sum(dim=1).view(...)  # re-normalise
        m_new = Categorical(probs_new)
        actions = m_new.sample()
        log_probs = m.log_prob(actions)
```
If sampling (stochastic mode — used during training and inference):
- Copy the probabilities and block out already-visited cells.
- Re-normalise so the remaining probabilities sum to 1.
- Sample one cell randomly, weighted by its probability.
- Compute the log-probability of the chosen action (used by PPO later).

```python
    else:
        probs_new[action_mask.view(...).bool()] = 0
        actions = torch.argmax(probs_new, dim=1)
```
If not sampling (greedy mode — used for deterministic evaluation): block
visited cells and pick the single cell with the highest probability.

```python
    return actions.view(-1), log_probs, values.view(-1), probs
```
Return the chosen actions, their log-probabilities, state values, and the
original (unmasked) probability distribution.

---

### `collect_trajs(env, policy, patch_num, max_traj_length, is_eval, sample_action)`

The function that actually runs the Generator in the environment and collects
one complete scanpath (or a batch of them).

```python
    obs_fov = env.observe()
    act, log_prob, value, prob = select_action((obs_fov, env.task_ids),
                                               policy, sample_action,
                                               action_mask=env.action_mask)
```
Take the first observation (initial state after center fixation). Select the
first action.

**Evaluation mode (`is_eval=True`):**

```python
    actions = []
    while i < max_traj_length:
        new_obs_fov, curr_status = env.step(act)
        status.append(curr_status)
        actions.append(act)
        obs_fov = new_obs_fov
        act, ... = select_action(...)
        i += 1
    trajs = {'status': torch.stack(status), 'actions': torch.stack(actions)}
```
Keep taking actions until the maximum number of fixations is reached. Record
every action. Do NOT stop early even if the target is found (collect the full
sequence for analysis).

**Training mode (`is_eval=False`):**

```python
    while i < max_traj_length and env.status.min() < 1:
        new_obs_fov, curr_status = env.step(act)
        SASPs.append((obs_fov, act, new_obs_fov))
        ...
        i += 1
```
Stop as soon as the target is found (`env.status.min() >= 1`) OR the maximum
step count is reached. Record `(state, action, next_state)` tuples at each
step.

```python
    for i in range(bs):
        ind = (status[:, i] == 1).argmax().item() + 1
        if status[:, i].sum() == 0:
            ind = status.size(0)
        trajs.append({'curr_states': S[:ind, i], 'actions': A[:ind, i],
                      'values': V[:ind+1, i], 'log_probs': LogP[:ind, i],
                      'rewards': R[:ind, i], 'task_id': ..., 'length': ind})
```
For each image in the batch, find the step at which the target was found
(`status == 1`). Slice the trajectory up to that point. Package everything
(states, actions, values, log-probs, rewards, task ID) into a dictionary.
`length` records how many steps were taken.

---

### `process_trajs(trajs, gamma, mtd, tau)`

```python
    for traj in trajs:
        acc_reward[-1] = traj['rewards'][-1]
        for i in reversed(range(acc_reward.size(0) - 1)):
            acc_reward[i] = traj['rewards'][i] + gamma * acc_reward[i + 1]
        traj['acc_rewards'] = acc_reward
```
Compute the **discounted cumulative reward** for each step, working backwards.
The reward at step t = immediate reward + 0.9 × reward at step t+1 + ...
This means rewards in the near future count more than rewards far in the
future. `gamma=0.9` is the discount factor from the settings.

```python
        # GAE (Generalised Advantage Estimation)
        delta = traj['rewards'] + gamma * values[1:] - values[:-1]
        adv[-1] = delta[-1]
        for i in reversed(range(delta.size(0) - 1)):
            adv[i] = delta[i] + gamma * tau * adv[i + 1]
        traj['advantages'] = adv
```
Compute the **advantage** at each step — roughly, "was this action better or
worse than expected?" The GAE method smoothly blends short-term and long-term
advantage estimates using `tau=0.96`. This is more stable than using raw
rewards directly.

---

### `compute_search_cdf(scanpaths, annos, max_step)`

```python
    task_names = np.unique([traj['task'] for traj in scanpaths])
    num_steps = get_num_steps(scanpaths, annos, task_names)
    cdf_tasks = get_mean_cdf(num_steps, task_names, max_step + 1)
    mean_cdf = np.mean(cdf_tasks, axis=0)
    return mean_cdf, std_cdf
```
For each scanpath, count how many steps it took to land on the target (using
`get_num_step2target`). Group by target category. For each category, compute
what fraction of scanpaths found the target by step 1, by step 2, ..., by step
6. Average across all categories. Return the mean curve and its standard
deviation.

---

### `cutFixOnTarget(trajs, target_annos)`

```python
    for i, traj in enumerate(task_trajs):
        key = traj['task'] + '_' + traj['name']
        bbox = target_annos[key]
        traj_len = get_num_step2target(traj['X'], traj['Y'], bbox)
        traj['X'] = traj['X'][:traj_len]
        traj['Y'] = traj['Y'][:traj_len]
```
Find the first fixation that falls inside the target's bounding box and trim
the scanpath at that point. This is used before computing evaluation metrics
so that only the search portion of the scanpath is evaluated (not any
fixations after the target was already found).

---

### `actions2scanpaths(actions, patch_num, im_w, im_h)`

```python
    for traj in actions:
        task_name, img_name, condition, actions = traj
        py = (actions // patch_num[0]) / float(patch_num[1])
        px = (actions % patch_num[0]) / float(patch_num[0])
        fixs = torch.stack([px, py])
        fixs = np.concatenate([np.array([[0.5], [0.5]]), fixs.cpu().numpy()], axis=1)
        scanpaths.append({'X': fixs[0] * im_w, 'Y': fixs[1] * im_h, ...})
```
Convert a sequence of grid-cell indices back into pixel coordinates.
- `actions // patch_num[0]` → row index
- `actions % patch_num[0]`  → column index
- Divide by grid dimensions to get normalised 0–1 values.
- Prepend `[0.5, 0.5]` (the forced centre start) at the beginning.
- Multiply by image width/height to get pixel coordinates.
Return as a list of scanpath dictionaries with `X` and `Y` arrays.

---

### `save(...)` and `load(...)`

```python
def save(global_step, model, optim, name, pkg_dir, is_best, max_checkpoints):
    state = {"global_step": global_step,
             "model": model.state_dict(),
             "optim": optim.state_dict()}
    torch.save(state, save_path)
    if is_best:
        copyfile(save_path, best_path)
    # delete oldest checkpoint if over the limit
    while len(history) > max_checkpoints:
        os.remove(oldest_checkpoint)
```
Package the model weights, optimizer state, and current training step into a
dictionary and write it to disk. If `is_best=True`, also copy it to
`trained_generator.pkg` / `trained_discriminator.pkg`. Then check if there are
more than `max_checkpoints=5` checkpoint files and delete the oldest ones.

```python
def load(step_or_path, model, name, optim, pkg_dir, device):
    state = torch.load(save_path, map_location=device)
    global_step = state["global_step"]
    model.load_state_dict(state["model"])
    if optim is not None:
        optim.load_state_dict(state["optim"])
    return global_step
```
Read a checkpoint file. Restore the model weights and (optionally) the
optimizer state. Return the step number so training can resume from where it
left off.

---

## `irl_dcb/models.py`

**Purpose:** Defines the two neural networks — the Generator and the
Discriminator.

---

### Class `LHF_Policy_Cond_Small` — the Generator

The Generator takes in "what the image currently looks like" (the blended
belief-map state) plus "what object am I searching for" and outputs a
probability for every grid cell: "how likely should the next fixation be here?"

```python
class LHF_Policy_Cond_Small(nn.Module):
    def __init__(self, action_num, target_size, task_eye, ch):
        ...
        self.feat_enc = nn.Conv2d(ch + target_size, 128, 5, padding=2)
        self.actor1   = nn.Conv2d(128 + target_size, 64, 3, padding=1)
        self.actor2   = nn.Conv2d(64  + target_size, 32, 3, padding=1)
        self.actor3   = nn.Conv2d(32  + target_size, 1,  1)
        self.critic0  = nn.Conv2d(128 + target_size, 128, 3)
        self.critic1  = nn.Conv2d(128 + target_size, 256, 3)
        self.critic2  = nn.Linear(256 + target_size, 64)
        self.critic3  = nn.Linear(64, 1)
        self.task_eye = task_eye
```
Define all the layers. `nn.Conv2d` is a 2D convolution layer — a sliding
window that detects spatial patterns. `nn.Linear` is a fully-connected layer.
`task_eye` is an 18×18 identity matrix — used to create one-hot category
vectors.

```python
    def get_one_hot(self, tid):
        return self.task_eye[tid]
```
Look up the one-hot vector for category `tid`. For example, if `tid=9`
(laptop), return a vector of 18 zeros with a 1 in position 9.

```python
    def modulate_features(self, feat_maps, tid_onehot):
        bs, _, h, w = feat_maps.size()
        task_maps = tid_onehot.expand(bs, tid_onehot.size(1), h, w)
        return torch.cat([feat_maps, task_maps], dim=1)
```
"Inject" the target information into the feature maps. Broadcast the 18-element
task vector to the same spatial size (h × w) and concatenate it as extra
channels. Now every spatial location knows what the search target is.

```python
    def forward(self, x, tid, act_only=False):
        tid_onehot = self.get_one_hot(tid).view(bs, tid_onehot.size(1), 1, 1)
        x = self.modulate_features(x, tid_onehot)   # (134+18) channels
        x = torch.relu(self.feat_enc(x))             # → 128 channels
```
First forward pass: concatenate the task vector, then apply the feature
encoder convolution and ReLU activation. ReLU sets all negative values to
zero, introducing non-linearity.

```python
        # Actor (policy) branch
        act_logits = self.modulate_features(x, tid_onehot)
        act_logits = torch.relu(self.actor1(act_logits))  # → 64 ch
        act_logits = self.modulate_features(act_logits, tid_onehot)
        act_logits = torch.relu(self.actor2(act_logits))  # → 32 ch
        act_logits = self.modulate_features(act_logits, tid_onehot)
        act_logits = self.actor3(act_logits).view(bs, -1) # → 640 values
        act_probs  = F.softmax(act_logits, dim=-1)
```
The actor branch (decides where to look). Four successive convolutions with
task injection at each step, reducing from 128 to 64 to 32 to 1 channel.
Flatten the final 1×20×32 output to a 640-element vector. Apply softmax to
convert raw scores into probabilities that all sum to 1.

```python
        if act_only:
            return act_probs, None

        # Critic (value) branch
        x = self.max_pool(torch.relu(self.critic0(x)))   # downsample
        x = self.modulate_features(x, tid_onehot)
        x = self.max_pool(torch.relu(self.critic1(x)))   # downsample again
        x = x.view(bs, x.size(1), -1).mean(dim=-1)       # global average pool
        x = torch.cat([x, tid_onehot.view(bs, -1)], dim=1)
        x = torch.relu(self.critic2(x))
        state_values = self.critic3(x)
        return act_probs, state_values
```
The critic branch (estimates expected future reward). Apply two rounds of
convolution + max-pooling (each halves the spatial size). Flatten and average
over all spatial positions. Concatenate the task vector once more. Pass
through two fully-connected layers to get a single scalar value.

---

### Class `LHF_Discriminator_Cond` — the Discriminator

```python
class LHF_Discriminator_Cond(nn.Module):
    def __init__(self, action_num, target_size, task_eye, ch):
        self.max_pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(ch + target_size, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128 + target_size, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64  + target_size, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32  + target_size, 1,  1)
```
Similar structure to the Generator's actor branch but dedicated entirely to
scoring how human-like a (state, action) pair is.

```python
    def forward(self, x, action, tid):
        x = self.modulate_features(x, tid_onehot)
        x = torch.relu(self.conv1(x))
        if h == 80:
            x = self.max_pool(x)
        x = self.modulate_features(x, tid_onehot)
        x = torch.relu(self.conv2(x))
        if h == 80:
            x = self.max_pool(x)
        x = self.modulate_features(x, tid_onehot)
        x = torch.relu(self.conv3(x))
        x = self.modulate_features(x, tid_onehot)
        x = self.conv4(x).view(bs, -1)   # → 640 raw scores
```
Process the state through 4 convolution layers with task injection at each
step. If the input is higher-resolution (h=80), apply max-pooling to shrink
it. The output is 640 raw scores — one per grid cell.

```python
        if action is None:
            return x                              # return the full reward map
        else:
            return x[torch.arange(bs), action.squeeze()]
```
If an action is specified, pick only the score at the chosen grid cell and
return it as a single number. If no action is given (full reward map mode),
return all 640 scores — this is used to visualise which regions the
Discriminator thinks are most human-like.

---

## `irl_dcb/environment.py`

**Purpose:** Simulates the "world" the Generator acts in. Manages the image
state, fixation history, and whether the target has been found.

```python
class IRL_Env4LHF:
    def __init__(self, pa, max_step, mask_size, status_update_mtd,
                 device, inhibit_return=False, init_mtd='center'):
        self.pa = pa
        self.max_step = max_step + 1   # +1 to include the initial centre fixation
        self.inhibit_return = inhibit_return
        self.mask_size = mask_size
        self.status_update_mtd = status_update_mtd
```
Store all configuration. `mask_size` is the IOR radius. `status_update_mtd`
is `"SOT"` (stop on target). `inhibit_return=True` means already-visited cells
are blocked.

---

### `set_data(data)`

```python
    def set_data(self, data):
        self.label_coding    = data['label_coding'].to(self.device)
        self.img_names       = data['img_name']
        self.cat_names       = data['cat_name']
        self.init_fix        = data['init_fix'].to(self.device)
        self.init_action_mask= data['action_mask'].to(self.device)
        self.init_history_map= data['history_map'].to(self.device)
        self.task_ids        = data['task_id'].to(self.device)
        self.lr_feats        = data['lr_feats'].to(self.device)
        self.hr_feats        = data['hr_feats'].to(self.device)
        self.batch_size      = self.hr_feats.size(0)
        ...
        self.reset()
```
Load a batch of image data into the environment. Move all tensors to the GPU.
Immediately call `reset()` to initialise the environment state.

---

### `reset()`

```python
    def reset(self):
        self.step_id   = 0
        self.fixations = torch.zeros((self.batch_size, self.max_step, 2), ...)
        self.status    = torch.zeros(self.batch_size, dtype=torch.uint8, ...)
        self.is_active = torch.ones (self.batch_size, dtype=torch.uint8, ...)
        self.states    = self.lr_feats.clone()
        self.action_mask   = self.init_action_mask.clone()
        self.history_map   = self.init_history_map.clone()
```
Reset all counters and maps to their starting values. Copy the initial LR
belief map as the starting state. Copy the initial action mask (centre region
blocked). All scanpaths start as "active" (not done).

```python
        # centre initialisation
        elif self.init == 'center':
            self.fixations[:, 0] = torch.tensor(
                [[self.pa.patch_num[0] / 2, self.pa.patch_num[1] / 2]], ...)
```
Record the first fixation as the grid centre (column 16, row 10 for a 32×20
grid). This is step 0.

---

### `observe(accumulate=True)`

```python
    def observe(self, accumulate=True):
        if self.step_id > 0:
            remap_ratio = self.pa.patch_num[0] / float(self.states.size(-1))
            lastest_fixation_on_feats = self.fixations[:, self.step_id].to(
                dtype=torch.float32) / remap_ratio
            px = lastest_fixation_on_feats[:, 0]
            py = lastest_fixation_on_feats[:, 1]
```
At step 0, skip updating (state was already set up by `set_data`).
For step > 0, get the coordinates of the most recent fixation and convert from
grid coordinates to the feature-map coordinate system.

```python
            for i in range(self.batch_size):
                mask = foveal2mask(px[i], py[i], self.pa.fovea_radius, ...)
                mask = torch.from_numpy(mask).to(self.device)
                mask = mask.unsqueeze(0).repeat(self.states.size(1), 1, 1)
                masks.append(mask)
            masks = torch.stack(masks)
```
For each image in the batch, create a circular spotlight mask at the current
fixation location.

```python
            if accumulate:
                self.states = (1 - masks) * self.states + masks * self.hr_feats
            else:
                self.states = (1 - masks) * self.lr_feats  + masks * self.hr_feats
            self.history_map = (1 - masks[:,0]) * self.history_map + masks[:,0]
```
Update the image state. If `accumulate=True` (the normal case), blend the
sharp HR detail into the previously-accumulated state — previously revealed
regions stay sharp. If `accumulate=False`, re-blend from scratch each time.
Update the history map to mark this region as visited.

```python
        return self.states.clone()
```
Return a copy of the current state (copy prevents the caller from accidentally
modifying the environment's internal state).

---

### `step(act_batch)`

```python
    def step(self, act_batch):
        self.step_id += 1
        assert self.step_id < self.max_step

        py, px = act_batch // self.pa.patch_num[0], act_batch % self.pa.patch_num[0]
        self.fixations[:, self.step_id, 1] = py
        self.fixations[:, self.step_id, 0] = px
```
Increment the step counter (and check we have not exceeded the maximum). Convert
the flat grid-cell index to (row, column) coordinates. Record the new fixation.

```python
        if self.inhibit_return:
            self.action_mask[...,
                             py-self.mask_size : py+self.mask_size+1,
                             px-self.mask_size : px+self.mask_size+1] = 1
```
If inhibition of return is on, mark the region around the new fixation as
blocked in the action mask.

```python
        obs = self.observe()
        self.status_update(act_batch)
        return obs, self.status
```
Update the image state by calling `observe()`. Check if the target was found.
Return the new state and the status vector.

---

### `status_update(act_batch)`

```python
    def status_update(self, act_batch):
        if self.status_update_mtd == 'SOT':
            done = self.label_coding[torch.arange(self.batch_size), 0, act_batch]
        done[self.status > 0] = 2
        self.status = done.to(torch.uint8)
```
For each image, look up whether the chosen grid cell overlaps with the target
object (`label_coding` contains 1s where the target is, 0s elsewhere).
If yes, `done=1` (found target). If the scanpath was already finished
(`status > 0`), set `done=2` (already done). Update the status tensor.

---

## `irl_dcb/builder.py`

**Purpose:** Assembles the Generator, Discriminator, and two environments into
a single ready-to-train package.

```python
def build(hparams, is_training, device, catIds, load_path=None):
    input_size = 134
    task_eye = torch.eye(len(catIds)).to(device)
```
The number of belief-map channels is 134. Create the 18×18 identity matrix
on the GPU — this is shared between both models.

```python
    discriminator = LHF_Discriminator_Cond(
        hparams.Data.patch_count, len(catIds), task_eye, input_size).to(device)
    generator = LHF_Policy_Cond_Small(
        hparams.Data.patch_count, len(catIds), task_eye, input_size).to(device)
```
Instantiate both neural networks and move them to the GPU. Both networks
receive: total number of grid cells (640), number of categories (18), the
shared identity matrix, and the number of input channels (134).

```python
    if load_path:
        load('best', generator,     'generator',     pkg_dir=load_path)
        global_step = load('best', discriminator, 'discriminator', pkg_dir=load_path)
    else:
        global_step = 0
```
If a checkpoint path is given, load saved weights for both models and recover
the training step counter. Otherwise start fresh from step 0.

```python
    env_train = IRL_Env4LHF(hparams.Data, max_step=hparams.Data.max_traj_length,
                             mask_size=hparams.Data.IOR_size,
                             status_update_mtd=hparams.Train.stop_criteria,
                             device=device, inhibit_return=True)
    env_valid = IRL_Env4LHF(...)
```
Create separate environment instances for training and validation. Having two
separate instances means they can hold different data simultaneously without
interfering with each other.

```python
    return {'env': {'train': env_train, 'valid': env_valid},
            'model': {'gen': generator, 'disc': discriminator},
            'loaded_step': global_step}
```
Return all components in a structured dictionary.

---

---

# PHASE 2 — Training Loop

---

## `train.py` — Entry Point

```python
from docopt import docopt
args = docopt(__doc__)
device = torch.device('cuda:{}'.format(args['--cuda']))
hparams = JsonConfig(hparams)
```
Parse command-line arguments. Select the GPU. Load the settings file.

```python
DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
DCB_dir_LR = join(dataset_root, 'DCBs/LR/')
bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'), allow_pickle=True).item()
```
Build paths to the pre-computed belief maps. Load the bounding box
annotations dictionary (maps `"category_imagename"` → `[x, y, width, height]`).

```python
with open(join(dataset_root, 'coco_search18_fixations_TP_train.json')) as json_file:
    human_scanpaths_train = json.load(json_file)
with open(join(dataset_root, 'coco_search18_fixations_TP_validation.json')) as json_file:
    human_scanpaths_valid = json.load(json_file)
```
Load the ground-truth human scanpaths. Each file is a JSON array of scanpath
dictionaries (see README for the format). `TP` = target-present trials only.

```python
if hparams.Train.exclude_wrong_trials:
    human_scanpaths_train = list(filter(lambda x: x['correct'] == 1, human_scanpaths_train))
    human_scanpaths_valid = list(filter(lambda x: x['correct'] == 1, human_scanpaths_valid))
```
If the setting `exclude_wrong_trials` is true (it is `false` by default),
remove all scanpaths where `correct=0` (the person failed to find the target).
`filter` + `lambda` = keep only items that satisfy the condition.

```python
dataset = process_data(human_scanpaths_train, human_scanpaths_valid,
                       DCB_dir_HR, DCB_dir_LR, bbox_annos, hparams)
built   = build(hparams, True, device, dataset['catIds'])
trainer = Trainer(**built, dataset=dataset, device=device, hparams=hparams)
trainer.train()
```
Prepare all data, build all components, create the Trainer, and start
training. `**built` unpacks the dictionary as keyword arguments.

---

## `irl_dcb/trainer.py` — Master Training Loop

### `__init__`

```python
self.log_dir = os.path.join(hparams.Train.log_root, "log_" + date)
self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
os.makedirs(self.log_dir)
hparams.dump(self.log_dir)
os.makedirs(self.checkpoints_dir)
```
Create a timestamped folder for this training run (e.g.
`./assets/log_20260312_143022/`). Create a `checkpoints/` subdirectory inside
it. Save a copy of the current settings to the log folder.

```python
self.train_img_loader = DataLoader(dataset['img_train'],
                                   batch_size=self.batch_size,
                                   shuffle=True, num_workers=16)
self.valid_img_loader = DataLoader(dataset['img_valid'],
                                   batch_size=self.batch_size,
                                   shuffle=False, num_workers=16)
self.train_HG_loader  = DataLoader(dataset['gaze_train'],
                                   batch_size=self.batch_size,
                                   shuffle=True, num_workers=16)
```
Wrap the three datasets in `DataLoader` objects. A `DataLoader` handles
batching, shuffling, and parallel data loading automatically. `num_workers=16`
means 16 CPU threads are used to load data in the background while the GPU
is training.

```python
self.ppo  = PPO(self.generator, hparams.PPO.lr, ...)
self.gail = GAIL(self.discriminator, hparams.Train.gail_milestones, ...)
self.writer = SummaryWriter(self.log_dir)
```
Create the PPO trainer (for the Generator), the GAIL trainer (for the
Discriminator), and a TensorBoard writer (for visualising training curves).

---

### `train()`

```python
    def train(self):
        self.generator.train()
        self.discriminator.train()
        self.global_step = self.loaded_step
```
Set both networks to "training mode" (enables dropout and batch normalisation
training behaviour). Set the step counter to either 0 (fresh start) or the
loaded checkpoint step (resumed training).

```python
        for i_epoch in range(self.n_epoches):
            for i_batch, batch in enumerate(self.train_img_loader):
```
Outer loop: repeat for 30 epochs. Inner loop: iterate through all batches of
128 training images.

---

**Step 1 — Collect fake scanpaths:**

```python
                trajs_all = []
                self.env.set_data(batch)
                for i_step in range(self.n_steps):       # n_steps = 4
                    with torch.no_grad():
                        self.env.reset()
                        trajs = utils.collect_trajs(self.env, self.generator,
                                                    self.patch_num, self.max_traj_len)
                        trajs_all.extend(trajs)
```
Load the current batch of images into the training environment. Run the
Generator 4 times (`n_steps=4`), resetting the environment before each run.
This collects 4 independent simulated scanpaths per image, giving more
training signal. `torch.no_grad()` disables gradient computation during
collection (saves memory — gradients are only needed during the weight
update steps).

---

**Step 2 — Train Discriminator (GAIL):**

```python
                fake_data = FakeDataRollout(trajs_all, self.batch_size)
                D_loss, D_real, D_fake = self.gail.update(
                    self.train_HG_loader, fake_data)
                self.writer.add_scalar("discriminator/fake_loss", D_fake, ...)
                self.writer.add_scalar("discriminator/real_loss", D_real, ...)
```
Package the fake scanpaths into a `FakeDataRollout`. Call `gail.update()` to
train the Discriminator on real vs. fake pairs. Log the real and fake loss
values to TensorBoard.

---

**Step 3 — Periodic evaluation:**

```python
                if self.global_step > 0 and self.global_step % self.eval_every == 0:
                    all_actions = []
                    for i_sample in range(10):
                        for batch in self.valid_img_loader:
                            self.env_valid.set_data(batch)
                            with torch.no_grad():
                                self.env_valid.reset()
                                trajs = utils.collect_trajs(self.env_valid,
                                    self.generator, ..., is_eval=True, sample_action=True)
                                all_actions.extend([...])
```
Every 20 steps, switch to evaluation mode. Generate 10 independent scanpath
samples for every validation image (to account for the model's randomness —
like a person who might scan slightly differently each time). Collect all
generated actions.

```python
                    scanpaths = utils.actions2scanpaths(all_actions, ...)
                    utils.cutFixOnTarget(scanpaths, self.bbox_annos)
                    mean_cdf, _ = utils.compute_search_cdf(scanpaths, ...)
                    self.writer.add_scalar('evaluation/TFP_step1', mean_cdf[1], ...)
                    self.writer.add_scalar('evaluation/TFP_step3', mean_cdf[3], ...)
                    self.writer.add_scalar('evaluation/TFP_step6', mean_cdf[6], ...)
                    sad = np.sum(np.abs(self.human_mean_cdf - mean_cdf))
                    self.writer.add_scalar('evaluation/prob_mismatch', sad, ...)
```
Convert actions to pixel coordinates. Trim each scanpath at the first
target fixation. Compute the search efficiency curve. Log the target-found
probability at steps 1, 3, and 6. Compute the total mismatch between the
model's efficiency curve and the human reference curve (sum of absolute
differences, or SAD). A lower SAD means the model is more human-like.

---

**Step 4 — Train Generator (PPO):**

```python
                if i_batch % self.n_critic == 0:         # every batch (n_critic=1)
                    with torch.no_grad():
                        for i in range(len(trajs_all)):
                            states  = trajs_all[i]["curr_states"]
                            actions = trajs_all[i]["actions"].unsqueeze(1)
                            tids    = trajs_all[i]['task_id']
                            rewards = F.logsigmoid(
                                self.discriminator(states, actions, tids))
                            trajs_all[i]["rewards"] = rewards
```
For each fake trajectory, pass the recorded (state, action) pairs through the
Discriminator. The reward is `log(sigmoid(discriminator_score))`. When the
Discriminator gives a high score (thinks the gaze looks human), this reward
is close to 0. When the Discriminator gives a low score (thinks it looks fake),
this reward is a large negative number. So the Generator is rewarded for
fooling the Discriminator.

```python
                    return_train = utils.process_trajs(trajs_all, self.gamma, ...)
                    rollouts = RolloutStorage(trajs_all, shuffle=True, norm_adv=True)
                    loss = self.ppo.update(rollouts)
                    self.writer.add_scalar("generator/ppo_loss", loss, ...)
```
Compute discounted returns and advantages for all trajectories. Package into
`RolloutStorage`. Call PPO to update the Generator weights. Log the loss.

---

**Step 5 — Checkpointing:**

```python
                if self.global_step % self.checkpoint_every == 0 and self.global_step > 0:
                    utils.save(global_step=self.global_step, model=self.generator,
                               optim=self.ppo.optimizer, name='generator', ...)
                    utils.save(global_step=self.global_step, model=self.discriminator,
                               optim=self.gail.optimizer, name='discriminator', ...)
                self.global_step += 1
```
Every 100 steps, save both models to disk. Increment the global step counter.

---

## `irl_dcb/gail.py` — Discriminator Training

```python
class GAIL():
    def __init__(self, discriminator, milestones, state_enc, device, lr, betas):
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.1)
```
Create an Adam optimiser for the Discriminator. `Adam` is an adaptive gradient
method — it adjusts the learning rate individually for each parameter.
`MultiStepLR` will multiply the learning rate by 0.1 at step 10,000 (a common
trick to stabilise late-stage training).

---

### `compute_grad_pen(...)` — Gradient Penalty

```python
    def compute_grad_pen(self, expert_states, ..., type='real', lambda_=5):
        mixup_states = expert_states.detach()
        mixup_states.requires_grad = True
        disc = torch.sigmoid(self.discriminator(*mixup_data))
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(outputs=disc, inputs=mixup_states,
                             grad_outputs=ones, create_graph=True, ...)[0]
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
```
Compute the gradient penalty, a regularisation technique from the WGAN-GP
paper. The idea: compute how quickly the Discriminator's output changes as
the input state changes (the gradient). Penalise it if this gradient's
magnitude is far from 1. This keeps the Discriminator "Lipschitz-smooth" and
prevents training instability. `lambda_=5` controls the strength of this
penalty.

---

### `update(true_data_loader, fake_data, ...)` — Training the Discriminator

```python
    def update(self, true_data_loader, fake_data, ...):
        for i_batch, true_batch in enumerate(true_data_loader):
            if i_batch == len(fake_data): break
            fake_batch = next(fake_data_generator)
```
Pair each real-human-gaze mini-batch with a fake-Generator mini-batch.

```python
            real_S = true_batch['true_state'].to(self.device)
            real_A = true_batch['true_action']
            real_tids = true_batch['task_id']
            fake_S, fake_A, fake_P, fake_tids = fake_batch
```
Unpack the real and fake data: states (S), actions (A), task IDs.

```python
            real_outputs = self.discriminator(real_S, real_A, real_tids)
            fake_outputs = self.discriminator(fake_S, fake_A, fake_tids)
            real_labels = torch.ones (real_outputs.size()).to(self.device)
            fake_labels = torch.zeros(fake_outputs.size()).to(self.device)
```
Pass both real and fake examples through the Discriminator to get scores.
Create target labels: real data should score 1, fake data should score 0.

```python
            expert_loss = F.binary_cross_entropy_with_logits(real_outputs, real_labels)
            policy_loss = F.binary_cross_entropy_with_logits(fake_outputs, fake_labels)
            gail_loss   = expert_loss + policy_loss
```
Compute binary cross-entropy loss: penalise the Discriminator when it scores
real data low or fake data high. Sum both losses.

```python
            grad_pen = self.compute_grad_pen(*x_real, *x_fake)
            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
            self.lr_scheduler.step()
```
Add the gradient penalty. Zero out old gradients. Backpropagate the total
loss to compute how each weight should change. Apply the weight update. Step
the learning rate schedule.

---

## `irl_dcb/ppo.py` — Generator Training

```python
class PPO():
    def __init__(self, policy, lr, betas, clip_param, num_epoch,
                 batch_size, value_coef=1., entropy_coef=0.1):
        self.clip_param = clip_param        # 0.2 — maximum policy change per step
        self.value_coef = value_coef        # 1.0 — weight for critic loss
        self.entropy_coef = entropy_coef    # 0.01 — weight for exploration bonus
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.value_loss_fun = torch.nn.SmoothL1Loss()
```
Create the PPO optimiser. `SmoothL1Loss` is a robust version of mean squared
error — it is less sensitive to outliers.

---

### `evaluate_actions(obs_batch, actions_batch)`

```python
    def evaluate_actions(self, obs_batch, actions_batch):
        probs, values = self.policy(*obs_batch)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions_batch)
        return values, log_probs, dist.entropy().mean()
```
Re-run the Generator on a stored batch of (state, task_id) pairs to get the
current probability distribution. Compute the log-probability of the actions
that were actually taken. Compute the entropy of the distribution (high
entropy = the Generator is uncertain = exploring; low entropy = it has a
strong preference = exploiting).

---

### `update(rollouts)`

```python
    def update(self, rollouts):
        for e in range(self.num_epoch):
            for i, sample in enumerate(rollouts.get_generator(self.minibatch_size)):
                obs_batch, actions_batch, return_batch, \
                    old_action_log_probs_batch, adv_targ = sample
```
Loop through the training data in mini-batches. Each sample has: current
observations, the actions taken, the discounted returns, the log-probabilities
at the time the actions were taken (old policy), and the advantages.

```python
                values, action_log_probs, dist_entropy = \
                    self.evaluate_actions(obs_batch, actions_batch)
```
Run the current (updated) Generator on these states to get new probabilities
and values.

```python
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
```
Compute the probability ratio between the new and old policies for each
action. `exp(log_new - log_old) = new_prob / old_prob`. A ratio of 1 means
the policy has not changed. Greater than 1 means the action is now more likely.

```python
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
```
The PPO "clipped surrogate" objective. `surr1` is the normal policy gradient.
`surr2` clips the ratio to the range [0.8, 1.2] (since `clip_param=0.2`).
Taking the minimum of the two ensures the policy does not change too drastically
in a single update step. The negative sign makes it a minimisation problem
(we minimise the loss, which maximises the return).

```python
                value_loss  = self.value_loss_fun(return_batch,
                                                  values.squeeze()) * self.value_coef
                entropy_loss = -dist_entropy * self.entropy_coef
                loss = value_loss + action_loss + entropy_loss
```
Critic loss: penalise the Generator's value estimate for being inaccurate.
Entropy loss: reward diversity in the action distribution (prevents the
Generator from collapsing to always picking the same cell).

```python
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```
Standard training step: zero gradients, backpropagate, update weights.

---

---

# PHASE 3 — Evaluation & Metrics

---

## `irl_dcb/metrics.py`

**Purpose:** Contains all the functions for quantitatively measuring how
similar the model's predicted scanpaths are to real human scanpaths.

---

### `multimatch(s1, s2, im_size)`

```python
def multimatch(s1, s2, im_size):
    scanpath1 = np.ones((l1, 3), dtype=np.float32)
    scanpath1[:, 0] = s1['X']
    scanpath1[:, 1] = s1['Y']
    mm = docomparison(scanpath1, scanpath2, sz=im_size)
    return mm[0]
```
Format two scanpaths (lists of x, y coordinates) into the format expected by
the MultiMatch algorithm (an N×3 array: x, y, duration). Call `docomparison`
(from `multimatch.py`) and return the 4-element similarity score.

---

### `compute_mm(human_trajs, model_trajs, im_w, im_h)`

```python
    for traj in model_trajs:
        gt_trajs = list(filter(lambda x: x['name'] == img_name and x['task'] == task,
                               human_trajs))
        all_mm_scores.append((task,
            np.mean([multimatch(traj, gt_traj, (im_w, im_h))[:4]
                     for gt_traj in gt_trajs], axis=0)))
    return np.mean([x[1] for x in all_mm_scores], axis=0)
```
For each predicted scanpath, find all real human scanpaths for the same image
and target. Compute MultiMatch similarity against each human, then average.
Average again across all predicted scanpaths to get one 4-element score
`[vector, length, position, duration]`.

---

### `nw_matching(pred_string, gt_string, gap=0.0)`

```python
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            match  = F[i-1, j-1] + zero_one_similarity(pred_string[i-1], gt_string[j-1])
            delete = F[i-1, j] + gap
            insert = F[i, j-1] + gap
            F[i, j] = np.max([match, delete, insert])
    return F[len(pred_string), len(gt_string)] / max(len(pred_string), len(gt_string))
```
Needleman-Wunsch dynamic programming algorithm. Builds a score table where
`F[i][j]` = the best alignment score for the first i elements of the
prediction and first j elements of the ground truth. Match = +1 if the region
labels are the same; gap = 0 (no penalty for insertion/deletion). The final
score is normalised by the length of the longer sequence.

---

### `scanpath_ratio(traj, bbox)`

```python
    traj_dist   = np.sum(np.sqrt((X1-X2)**2 + (Y1-Y2)**2))
    target_dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)
    return min(target_dist / traj_dist, 1.0)
```
Compute the ratio of the straight-line distance from start to target divided
by the actual total path length. A ratio of 1.0 means the path was perfectly
direct. Lower values mean the path was more winding. Capped at 1.0.

---

### `compute_cdf_auc(cdf)` and `compute_prob_mismatch(cdf, human_mean_cdf)`

```python
def compute_cdf_auc(cdf):
    return cdf[0] + cdf[-1] + np.sum(cdf[1:-1])  # trapezoidal sum = area under curve
```
Compute the area under the search efficiency curve. A higher area means the
model found the target faster on average.

```python
def compute_prob_mismatch(cdf, human_mean_cdf):
    return np.sum(np.abs(cdf - human_mean_cdf))
```
Sum of absolute differences between the model's efficiency curve and the
human reference curve. A lower value means the model behaves more like a human.

---

## `irl_dcb/multimatch.py`

**Purpose:** Implements the full MultiMatch algorithm for comparing two
scanpaths geometrically. This is a standard algorithm in eye-tracking research,
adapted from the `multimatch` Python package.

---

### `cart2pol(x, y)`

```python
def cart2pol(x, y):
    rho   = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta
```
Convert a 2D vector from Cartesian coordinates (horizontal, vertical) to
polar coordinates (length, angle). Used to analyse the direction and
magnitude of eye movements (saccades).

---

### `gen_scanpath_structure(data)`

Converts a list of fixation points into a vector-based representation where
each element stores both the fixation position and the saccade vector going to
the next fixation:
- `fixation_x, fixation_y` — where the eye stopped
- `saccade_x, saccade_y` — the direction of the movement to the next fixation
- `saccade_rho` — how far the movement was
- `saccade_theta` — the angle of the movement

---

### `simplify_scanpath(eyedata, ...)` and `_simplify(...)`

These functions simplify scanpaths by removing fixations that are very short
in duration or very small in amplitude (below given thresholds). This removes
noise before comparison.

---

### `cal_vectorsim`, `cal_lengthsim`, `cal_positionsim`, `cal_durationsim`

Four functions that each compute one dimension of similarity between two
aligned scanpaths:
- **Vector similarity:** are the saccade directions the same?
- **Length similarity:** are the saccade distances similar?
- **Position similarity:** are the fixation locations similar?
- **Duration similarity:** are the fixation durations similar?

Each returns a value between 0 and 1.

---

### `docomparison(scanpath1, scanpath2, sz, ...)`

The main function. Takes two scanpaths, simplifies them, aligns them using
Dynamic Time Warping (finds the best match between their elements even if they
have different lengths), then computes all four similarity scores.

---

---

# PHASE 4 — Inference & Visualisation

---

## `test_single_image.py` — Predicting Scanpaths on a New Image

```python
IMG_PATH   = "test_office_resized.jpg"
HR_NPY     = "my_data/DCBs/HR/test_office_resized.npy"
LR_NPY     = "my_data/DCBs/LR/test_office_resized.npy"
MODEL_PATH = "trained_models/trained_generator.pkg"
HPARAMS    = "hparams/coco_search18.json"
TASK_ID    = 9
TASK_NAME  = "laptop"
NUM_RUNS   = 5
NUM_FIXATIONS = 6
```
User-editable configuration block at the top of the file. To predict on a
different image or search target, only these lines need to change.

```python
hr = torch.from_numpy(np.load(HR_NPY))   # shape: [134, 20, 32]
lr = torch.from_numpy(np.load(LR_NPY))   # shape: [134, 20, 32]
```
Load the pre-computed belief maps from the `.npy` files created by
`extract_DCBs_demo.py`. Convert from numpy arrays to PyTorch tensors.

```python
init_px = pa.patch_num[0] / 2.0     # = 16.0 (centre column)
init_py = pa.patch_num[1] / 2.0     # = 10.0 (centre row)
mask_np = foveal2mask(init_px, init_py, pa.fovea_radius, hr.size(-2), hr.size(-1))
mask_t  = torch.from_numpy(mask_np).unsqueeze(0).repeat(hr.size(0), 1, 1)
lr_init = (1 - mask_t) * lr + mask_t * hr
```
Apply the initial centre fixation to the LR belief map: within the centre
spotlight, replace blurry LR values with sharp HR values. This produces the
starting state that would have been seen after the forced first fixation.

```python
task_eye  = torch.eye(NUM_CATEGORIES).to(device)
generator = LHF_Policy_Cond_Small(pa.patch_count, NUM_CATEGORIES, task_eye, 134).to(device)
state = torch.load(MODEL_PATH, map_location=device)
generator.load_state_dict(state["model"])
generator.eval()
print(f"Loaded generator from step {state['global_step']}")
```
Build the Generator with the same architecture used during training. Load the
saved weights from the checkpoint file. `state["model"]` contains the weight
dictionary. `.eval()` switches off training-specific behaviour.

```python
env = IRL_Env4LHF(pa, max_step=NUM_FIXATIONS, mask_size=pa.IOR_size,
                  status_update_mtd="SOT", device=device, inhibit_return=True)
```
Set up the environment with the same settings as training.

```python
action_mask = torch.zeros((1, pa.patch_num[1], pa.patch_num[0]), dtype=torch.uint8)
cx, cy = int(pa.patch_num[0]/2), int(pa.patch_num[1]/2)
action_mask[0, cy-pa.IOR_size:cy+pa.IOR_size+1,
               cx-pa.IOR_size:cx+pa.IOR_size+1] = 1
```
Create the initial action mask (batch size = 1). Block the centre region as
already visited.

```python
data = {
    "lr_feats":     lr_init.unsqueeze(0),   # add batch dimension: [1, 134, 20, 32]
    "hr_feats":     hr.unsqueeze(0),
    "history_map":  torch.from_numpy(mask_np).unsqueeze(0),
    "action_mask":  action_mask,
    "init_fix":     torch.FloatTensor([[init_px/pa.patch_num[0],
                                        init_py/pa.patch_num[1]]]),
    "label_coding": torch.zeros((1, 1, pa.patch_count)),   # no target label needed for inference
    "task_id":      torch.tensor([TASK_ID]),
    "img_name":     [os.path.basename(IMG_PATH)],
    "cat_name":     [TASK_NAME],
}
env.set_data(data)
```
Bundle all the required data into the dictionary format the environment expects.
`unsqueeze(0)` adds a batch dimension (the model processes batches of images;
here the batch size is 1).

```python
for run in range(NUM_RUNS):
    env.reset()
    with torch.no_grad():
        trajs = collect_trajs(env, generator, pa.patch_num, NUM_FIXATIONS,
                              is_eval=True, sample_action=True)
    all_actions = [(TASK_NAME, os.path.basename(IMG_PATH),
                    "present", trajs["actions"][:, 0])]
    scanpaths = actions2scanpaths(all_actions, pa.patch_num, pa.im_w, pa.im_h)
```
Run inference 5 times. Each run is independent — because `sample_action=True`,
the Generator samples probabilistically and will produce a slightly different
path each time. Convert the grid-cell action sequence to pixel coordinates.

```python
    X, Y = sp["X"], sp["Y"]
    T = [1] * len(X)
    plot_scanpath(img, X, Y, T, title=title, save_path=out_path)
```
Extract the X and Y pixel coordinates. Set uniform "durations" (T) of 1 for
all fixations (just for visualisation purposes). Call `plot_scanpath` to draw
and save the result.

---

## `plot_scanpath.py` — Visualisation

```python
def convert_coordinate(X, Y, im_w, im_h):
    display_w, display_h = 1680, 1050
    ...
    X = (X - dif_ux) * scale
    Y = (Y - dif_uy) * scale
    return X, Y
```
The raw COCO-Search18 data was collected on a 1680×1050 display. Images were
shown centred on screen with letterboxing. This function converts from display
coordinates (measured on the full screen) to image pixel coordinates
(measured on just the image). This is only needed when plotting the original
raw data — the rescaled dataset used for training has already been converted.

---

### `plot_scanpath(img, xs, ys, ts, bbox, title, save_path)`

```python
    fig, ax = plt.subplots()
    ax.imshow(img)
```
Create a matplotlib figure and display the image as the background layer.

```python
    cir_rad_min, cir_rad_max = 30, 60
    min_T, max_T = np.min(ts), np.max(ts)
    rad_per_T = (cir_rad_max - cir_rad_min) / float(max_T - min_T)
```
Calculate how much the fixation circle radius scales with duration. Short
fixations get smaller circles (radius 30); long fixations get larger ones
(radius 60).

```python
    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i-1], ys[i-1], xs[i]-xs[i-1], ys[i]-ys[i-1],
                      width=3, color='yellow', alpha=0.5)
```
Draw a yellow arrow from each fixation to the next, showing the direction of
the eye movement (saccade). The arrow goes from the previous fixation position
to the current one.

```python
    for i in range(len(xs)):
        cir_rad = int(25 + rad_per_T * (ts[i] - min_T))
        circle = plt.Circle((xs[i], ys[i]), radius=cir_rad,
                             edgecolor='red', facecolor='yellow', alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(i+1), xy=(xs[i], ys[i]+3), ...)
```
Draw a yellow/red circle at each fixation location. The size of the circle
reflects how long the person looked there (larger = longer). Label each
circle with its order number (1, 2, 3...).

```python
    if bbox is not None:
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                          alpha=0.5, edgecolor='yellow', facecolor='none', linewidth=2)
        ax.add_patch(rect)
```
If a bounding box is provided, draw a yellow rectangle around the target
object.

```python
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
```
Save to disk at 150 DPI if a path was given. Always display the plot on screen.

---

### Main block (command-line usage)

```python
    scanpaths = list(filter(lambda x: x['task'] == args.task, scanpaths))
    if args.subj_id > 0:
        scanpaths = list(filter(lambda x: x['subject'] == args.subj_id, scanpaths))
    if args.random_trial == 1:
        id = np.random.randint(len(scanpaths))
    else:
        id = args.trial_id
```
Filter the loaded scanpaths to the chosen target category (and optionally to a
specific subject). Then pick either a random scanpath or a specific one by ID.

```python
    img_path = './{}/{}/{}'.format(args.image_dir, cat_name, img_name)
    img = mpimg.imread(img_path)
    X, Y, T = scanpath['X'], scanpath['Y'], scanpath['T']
    title = "target={}, correct={}".format(cat_name, scanpath['correct'])
    plot_scanpath(img, X, Y, T, bbox, title)
```
Build the image path from the directory structure. Load the image. Extract the
fixation coordinates and durations. Plot with a title showing the target name
and whether the person found it.

---

## Summary Table

| File | Phase | Role |
|---|---|---|
| `resize.py` | Pre-processing | Scale images to 512×320 |
| `extract_DCBs_demo.py` | Pre-processing | Generate HR/LR belief maps |
| `hparams/coco_search18.json` | Setup | All training settings |
| `irl_dcb/config.py` | Setup | Load and expose settings via dot-access |
| `dataset.py` | Setup | Organise all data into training/validation datasets |
| `irl_dcb/data.py` | Setup | Dataset classes for Generator and Discriminator |
| `irl_dcb/utils.py` | Setup + Training | Helper functions used throughout |
| `irl_dcb/models.py` | Setup | Generator and Discriminator neural networks |
| `irl_dcb/environment.py` | Setup + Training | Simulates the visual search world |
| `irl_dcb/builder.py` | Setup | Assembles all components |
| `train.py` | Training | Entry point — starts the training run |
| `irl_dcb/trainer.py` | Training | Master training loop |
| `irl_dcb/gail.py` | Training | Trains the Discriminator |
| `irl_dcb/ppo.py` | Training | Trains the Generator |
| `irl_dcb/metrics.py` | Evaluation | Quantitative scanpath comparison |
| `irl_dcb/multimatch.py` | Evaluation | MultiMatch geometry algorithm |
| `test_single_image.py` | Inference | Run trained model on a new image |
| `plot_scanpath.py` | Visualisation | Draw scanpath on an image |
