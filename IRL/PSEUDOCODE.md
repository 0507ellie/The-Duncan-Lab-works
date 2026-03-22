# Scanpath Prediction — Full Pseudocode

Two networks are trained together:
- **Generator** — decides where to look next
- **Discriminator** — judges whether a sequence of gaze points looks human or fake

---

---

# PHASE 0 — Pre-processing
> Run once before any training. Converts raw images into the format the model needs.

---

## `resize.py`

```
OPEN the original image file
RESIZE it to 512 pixels wide and 320 pixels tall
SAVE the resized image as a new file
```

---

## `extract_DCBs_demo.py`

```
LOAD a pre-trained object detection model (detectron2 -> object segmentation model)

DEFINE a function to convert detector output into a belief map:
    CREATE an empty grid with 134 layers (80 things + 54 stuff), 320 rows, 512 columns
    FOR each object/region the detector found:
        IF it is a countable object (person, chair, laptop...):
            FILL that object's layer with the detector's confidence score
            at the locations where the object was found
        IF it is background material (floor, sky, grass...):
            FILL that material's layer with 1s where it was found
    SHRINK the grid from 320x512 down to 20x32
        (keeps the spatial layout but makes it faster to process)
    RETURN the compact grid

DEFINE a function to produce two belief maps for one image:
    MAKE a sharp copy of the image (high-resolution)
    MAKE a blurry copy of the same image (low-resolution, which is using a guassian blur)
        (blurry = simulates peripheral vision)
    RUN the detector on the sharp copy  → get a belief map
    RUN the detector on the blurry copy → get a belief map
    RETURN both belief maps

FOR each image in the dataset:
    CALL the above function to get the sharp (HR) and blurry (LR) belief maps
    SAVE the HR belief map to the HR folder
    SAVE the LR belief map to the LR folder
```

---

---

# PHASE 1 — Setup
> Runs at the start of every training session.

---

## `hparams/coco_search18.json`

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

```
DEFINE a settings container:

    LOAD the settings file from disk
    FOR each setting in the file:
        IF the setting contains sub-settings (e.g. "Data", "Train"):
            WRAP those sub-settings in their own container 
        ELSE:
            STORE the setting as-is
    LOCK all settings so nothing can be changed accidentally during training

    PROVIDE a way to save a copy of the settings to the log folder
```

---

## `dataset.py`

```
FUNCTION prepare_all_data(train_scanpaths, valid_scanpaths, HR_folder, LR_folder, bounding_boxes, settings):

    --- Record where each person first looked ---
    FOR every scanpath in train + validation:
        KEY = category name + image name
        STORE the first gaze position (normalised 0-1) under that key

    --- Assign a number to each target category ---
    COLLECT all unique category names (bottle, laptop, microwave, ...)
    ASSIGN each a number: bottle=0, bowl=1, car=2, ...

    --- Build training examples for the Discriminator ---
    FOR each training scanpath:
        SET the first fixation to the image centre
        FOR each subsequent fixation (the function is defined in utils.py):
            SNAP the (x,y) pixel position to the nearest grid cell
            IF this cell was already visited → SKIP (no going back)
            RECORD (image name, category, all previous fixations, next grid cell)
        STOP after 6 fixations maximum

    --- Same for validation ---
    ALSO compute the human search efficiency benchmark:
        FOR step 1, 2, 3, 4, 5, 6:
            CALCULATE what fraction of real humans found the target by this step
        STORE this as the reference curve

    --- Build image datasets ---
    CREATE training image dataset   (each item = one image's belief maps + metadata)
    CREATE validation image dataset (same structure)

    --- Build human gaze datasets ---
    CREATE training gaze dataset    (each item = one step of one human scanpath)
    CREATE validation gaze dataset  (same, but action labels are slightly blurred)

    RETURN all datasets bundled together
```

---

## `irl_dcb/data.py`

```
--- DATASET: Image data (used by the Generator) ---

WHEN the training loop requests image number N:
    LOAD the HR (sharp) belief map from disk
    LOAD the LR (blurry) belief map from disk

    LOOK UP where the first fixation was for this image+category
    CREATE a circular spotlight at that location
    BLEND the image state:
        inside the spotlight  → use the HR (sharp) values
        outside the spotlight → keep the LR (blurry) values
    MARK the spotlight region as "already visited" in the history map
    MARK the spotlight region as "blocked" in the action mask (cannot revisit)

    MARK which grid cells overlap with the target object's bounding box
        (this tells the environment when the target has been "found")

    RETURN all of the above as a package


--- DATASET: Human gaze data (used by the Discriminator) ---

WHEN the training loop requests gaze example number N:
    LOAD the HR and LR belief maps
    START with the blurry LR state
    FOR each fixation the human made before this step:
        CREATE a spotlight at that fixation location
        BLEND: inside spotlight → replace with HR detail; outside → keep as-is
        MARK that region as visited in the history map
    (After this loop, the state reflects exactly what had been "seen" so far)

    STORE the reconstructed state as "true_state"
    STORE the next gaze location as "true_action"
    RETURN the package


--- STORAGE: Fake trajectory storage for Generator update (RolloutStorage) ---

AT CREATION:
    COLLECT all states, actions, rewards, log-probabilities, advantages
    from all fake trajectories into single large lists
    NORMALISE the advantages (subtract mean, divide by standard deviation)

WHEN asked for a mini-batch:
    RANDOMLY pick a subset of examples
    RETURN (states, actions, expected_returns, old_log_probs, advantages)


--- STORAGE: Fake trajectory storage for Discriminator update (FakeDataRollout) ---

AT CREATION:
    COLLECT all fake states, actions, task IDs, action probabilities

WHEN asked for a mini-batch:
    RANDOMLY pick a subset
    RETURN (fake_states, fake_actions, fake_probs, task_ids)
```

---

## `irl_dcb/utils.py`

```
FUNCTION make_spotlight_mask(centre_x, centre_y, radius, grid_height, grid_width):
    FOR each cell in the grid:
        IF distance from cell to centre <= radius:
            SET cell = 1  (inside spotlight)
        ELSE:
            SET cell = 0  (outside spotlight)
    RETURN the mask


FUNCTION pixel_to_grid_cell(x, y, cell_size, grid_columns):
    column = x divided by cell width  (integer)
    row    = y divided by cell height (integer)
    index  = row × number_of_columns + column
    RETURN index


FUNCTION grid_cell_to_pixel(index, cell_size, grid_columns):
    column = index MOD grid_columns
    row    = index DIV grid_columns
    pixel_x = column × cell_width  + half_cell_width
    pixel_y = row    × cell_height + half_cell_height
    RETURN pixel_x, pixel_y


FUNCTION snap_all_fixations(scanpaths, cell_size, grid_size, image_size, max_length):
    all_examples = empty list
    FOR each scanpath:
        SET first fixation to image centre
        CONVERT to grid cell, snap back to cell centre
        history = [first cell]

        FOR each remaining fixation (up to max_length):
            CONVERT to grid cell
            IF already in history → SKIP
            ADD to history
            RECORD (image, category, copy of history so far, this cell)

    RETURN all recorded examples


FUNCTION choose_next_fixation(current_state, task_id, generator, action_mask, sample):
    FEED (current_state, task_id) into the Generator
    GET probability score for each of the 640 grid cells
    GET estimated value of current state (for PPO later)

    ZERO OUT probabilities for already-visited cells
    RE-NORMALISE so remaining probabilities sum to 1

    IF sample is True:
        RANDOMLY pick a cell weighted by its probability
    ELSE:
        PICK the cell with the highest probability

    RETURN chosen cell, its log-probability, state value


FUNCTION run_one_scanpath(environment, generator, max_steps, is_evaluation):
    GET current image state from environment
    CHOOSE first fixation

    IF evaluation mode:
        REPEAT max_steps times:
            TAKE chosen fixation in environment
            RECORD this fixation
            CHOOSE next fixation
        RETURN all recorded fixations

    IF training mode:
        WHILE step count < max_steps AND target not yet found:
            TAKE chosen fixation in environment
            RECORD (previous state, chosen cell, new state)
            CHOOSE next fixation
        PACKAGE recorded steps into a trajectory
        RETURN trajectory


FUNCTION compute_discounted_rewards(trajectories, discount_factor):
    FOR each trajectory:
        WORKING BACKWARDS from the last step to the first:
            reward_at_step = immediate_reward + discount_factor × reward_at_next_step
        COMPUTE advantage = actual_discounted_reward − what_we_predicted
    RETURN average total reward across all trajectories


FUNCTION trim_scanpath_at_target(scanpaths, bounding_boxes):
    FOR each scanpath:
        FIND the first fixation that lands inside the target's bounding box
        REMOVE all fixations after that point


FUNCTION convert_grid_actions_to_pixels(action_sequences, grid_size, image_size):
    FOR each sequence of grid cell indices:
        CONVERT each index to a pixel (x, y) coordinate
        PREPEND the forced centre starting fixation
    RETURN list of (X array, Y array) scanpaths


FUNCTION compute_search_efficiency(scanpaths, bounding_boxes, max_steps):
    FOR each scanpath:
        COUNT how many steps until the gaze landed on the target
    FOR step = 1 to max_steps:
        CALCULATE what fraction of scanpaths found the target by this step
    AVERAGE across all target categories
    RETURN the efficiency curve


FUNCTION save_checkpoint(model, optimiser, step, folder):
    BUNDLE model weights + optimiser state + step number
    WRITE to disk as a file
    ALSO copy to "best" file
    IF more than 5 checkpoints exist → DELETE the oldest one


FUNCTION load_checkpoint(model, optimiser, folder):
    READ checkpoint file from disk
    RESTORE model weights
    RESTORE optimiser state
    RETURN the saved step number
```

---

## `irl_dcb/models.py`

```
--- NEURAL NETWORK: Generator (decides where to look) ---

SETUP:
    CREATE feature encoder layer       (134+18 channels → 128 channels)
    CREATE actor layer 1               (128+18 → 64)
    CREATE actor layer 2               (64+18  → 32)
    CREATE actor layer 3               (32+18  → 1)
    CREATE critic layer 1              (128+18 → 128, then halve spatial size)
    CREATE critic layer 2              (128+18 → 256, then halve spatial size)
    CREATE critic fully-connected 1    (256+18 → 64)
    CREATE critic fully-connected 2    (64 → 1)
    STORE the 18×18 identity matrix for one-hot encoding

WHEN given (image_state, target_category_id):

    STEP 1 — Encode the target:
        CONVERT category ID to an 18-element one-hot vector
            (e.g. laptop=9 → [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
        BROADCAST this vector across the whole 20×32 grid
        ATTACH it to the image state as extra channels

    STEP 2 — Extract features:
        APPLY feature encoder → compress to 128 channels
        APPLY activation function (set all negatives to zero)

    STEP 3 — Actor branch (WHERE to look):
        ATTACH target vector again
        APPLY layer 1 → 64 channels, activation
        ATTACH target vector again
        APPLY layer 2 → 32 channels, activation
        ATTACH target vector again
        APPLY layer 3 → 1 channel
        FLATTEN to 640 values (one per grid cell)
        CONVERT to probabilities (all 640 values sum to 1)
        OUTPUT: probability of looking at each grid cell

    STEP 4 — Critic branch (HOW GOOD is this state):
        APPLY convolution + shrink spatial size
        ATTACH target vector
        APPLY convolution + shrink spatial size again
        FLATTEN + average across all spatial positions
        ATTACH target vector
        APPLY fully-connected layer → 64 values
        APPLY fully-connected layer → 1 value
        OUTPUT: estimated expected future reward from this state


--- NEURAL NETWORK: Discriminator (judges how human-like a gaze step is) ---

SETUP:
    CREATE convolution layer 1   (134+18 → 128)
    CREATE convolution layer 2   (128+18 → 64)
    CREATE convolution layer 3   (64+18  → 32)
    CREATE convolution layer 4   (32+18  → 1)
    STORE the 18×18 identity matrix

WHEN given (image_state, chosen_grid_cell, target_category_id):

    CONVERT category ID to one-hot vector
    BROADCAST across the spatial grid and ATTACH to image state

    APPLY layer 1 → 128 channels, activation
    ATTACH target vector
    APPLY layer 2 → 64 channels, activation
    ATTACH target vector
    APPLY layer 3 → 32 channels, activation
    ATTACH target vector
    APPLY layer 4 → 1 channel
    FLATTEN to 640 scores (one per grid cell)

    PICK the score at the chosen grid cell
    OUTPUT: a single number  (high = looks human, low = looks fake)
```

---

## `irl_dcb/environment.py`

```
--- ENVIRONMENT: Simulates the visual search world ---

SETUP:
    STORE settings (grid size, max steps, IOR radius, stop rule)

WHEN given a batch of images (set_data):
    LOAD HR and LR belief maps, task IDs, initial fixations, action masks
    CALL reset()

RESET:
    SET step counter = 0
    CLEAR all recorded fixations
    MARK all scanpaths as "active" (not done yet)
    SET image state = the blurry LR belief map (starting state)
    RESTORE the initial action mask (centre region blocked)
    RESTORE the blank history map (centre region marked as visited)
    RECORD the forced first fixation at the image centre

OBSERVE (update image state after a fixation):
    IF this is not the first step:
        GET the most recent fixation location
        CREATE a circular spotlight at that location
        BLEND the image state:
            inside the spotlight  → replace with the HR (sharp) values
            outside the spotlight → keep the existing accumulated state
        MARK the spotlight region as visited in the history map
    RETURN the current image state

STEP (take one action):
    INCREMENT step counter
    CHECK we have not exceeded the maximum steps
    RECORD the chosen grid cell as the new fixation
    BLOCK the region around the new fixation in the action mask (IOR)
    CALL observe() to update the image state
    CHECK if the target was found (status_update)
    RETURN new image state + status (found / not found)

STATUS UPDATE:
    FOR each image in the batch:
        IF the chosen grid cell overlaps the target object's location:
            MARK as "done" (target found)
        IF already done:
            KEEP marked as done
```

---

## `irl_dcb/builder.py`

```
FUNCTION build_all_components(settings, device, category_list, checkpoint_path):

    CREATE a shared 18×18 identity matrix on the GPU

    CREATE the Discriminator network
        (takes: 640 grid cells, 18 categories, 134 input channels)
    CREATE the Generator network
        (takes: 640 grid cells, 18 categories, 134 input channels)

    IF a checkpoint path is provided:
        LOAD saved Generator weights from file
        LOAD saved Discriminator weights from file
        RECORD the training step number to resume from
    ELSE:
        START from step 0

    CREATE a training environment
        (inhibition of return ON, stop when target found)
    CREATE a validation environment
        (same settings, separate instance)

    RETURN Generator, Discriminator, train environment, valid environment, step number
```

---

---

# PHASE 2 — Training Loop

---

## `train.py`

```
READ command-line arguments:
    path to settings file
    path to dataset folder
    which GPU to use

LOAD settings from settings file
SET paths to the HR and LR belief map folders
LOAD bounding box annotations
LOAD training human scanpaths from JSON file
LOAD validation human scanpaths from JSON file

IF "exclude wrong trials" is ON:
    REMOVE all scanpaths where the person failed to find the target

CALL prepare_all_data()     → get all datasets ready
CALL build_all_components() → create Generator, Discriminator, environments
CREATE a Trainer with all of the above
CALL trainer.train()        → start the training loop
```

---

## `irl_dcb/trainer.py`

```
SETUP:
    CREATE a timestamped log folder
    SAVE a copy of settings into the log folder
    CREATE a checkpoints subfolder

    SET UP three data loaders:
        training images     (shuffled, 128 at a time)
        validation images   (ordered, 128 at a time)
        training human gaze (shuffled, 128 at a time)

    CREATE PPO  trainer  (will update the Generator)
    CREATE GAIL trainer  (will update the Discriminator)
    CREATE TensorBoard logger (tracks metrics for graphs)

---

TRAINING LOOP:

    SET Generator to training mode
    SET Discriminator to training mode

    REPEAT for 30 epochs:
        FOR each batch of 128 training images:

            ══════════════════════════════════
            STEP 1 — Collect fake scanpaths
            ══════════════════════════════════
            LOAD this image batch into the training environment
            REPEAT 4 times:
                RESET environment to the initial state
                RUN the Generator step-by-step:
                    AT each step:
                        GET current image state
                        GENERATOR picks the next grid cell to look at
                        ENVIRONMENT updates the image state (spotlight blending)
                        RECORD (state, chosen cell) pair
                        IF target found → STOP early
                COLLECT all recorded steps as one trajectory per image
            NOW we have 4 × 128 = 512 fake scanpaths worth of data

            ══════════════════════════════════
            STEP 2 — Train the Discriminator
            ══════════════════════════════════
            PACKAGE the fake scanpaths into batches
            CALL gail.update():
                SHOW Discriminator pairs of (real human gaze, fake generator gaze)
                Discriminator learns:
                    real human gaze  → score HIGH
                    fake gaze        → score LOW
            LOG Discriminator losses to TensorBoard

            ══════════════════════════════════
            STEP 3 — Evaluate (every 20 steps)
            ══════════════════════════════════
            IF global_step is a multiple of 20:
                REPEAT 10 times:
                    FOR each validation image:
                        RUN Generator to produce a full scanpath
                        RECORD the sequence of fixations
                CONVERT all fixation sequences to pixel coordinates
                TRIM each scanpath when it first hits the target
                COMPUTE search efficiency:
                    FOR each step 1 to 6:
                        WHAT fraction of scanpaths found the target by this step?
                COMPARE to the human reference efficiency curve
                    → compute total mismatch (sum of absolute differences -> in terms of "efficiency")
                LOG TFP@step1, TFP@step3, TFP@step6, mismatch to TensorBoard

            ══════════════════════════════════
            STEP 4 — Train the Generator
            ══════════════════════════════════
            FOR each recorded fake (state, action) pair:
                ASK the Discriminator: "how human-like was this?"
                STORE the score as the reward for that step
                    (high score → reward close to 0)
                    (low score  → large negative reward)

            COMPUTE discounted cumulative rewards for each trajectory
            COMPUTE advantages (was each action better or worse than expected?)
            CALL ppo.update():
                Generator learns to take actions that get higher Discriminator scores
            LOG Generator loss to TensorBoard

            ══════════════════════════════════
            STEP 5 — Save checkpoint
            ══════════════════════════════════
            IF global_step is a multiple of 100:
                SAVE Generator weights to disk
                SAVE Discriminator weights to disk
                DELETE oldest checkpoint if more than 5 exist

            INCREMENT global step counter
```

---

## `irl_dcb/gail.py`

```
SETUP:
    CREATE an optimiser for the Discriminator
    SCHEDULE: reduce learning rate by 10× at step 10000

---

FUNCTION compute_gradient_penalty(real_data):
    COMPUTE how quickly the Discriminator's score changes as input changes
    IF the rate of change is far from 1:
        RETURN a penalty proportional to how far it is
    (This keeps the Discriminator stable and prevents extreme outputs)

---

FUNCTION update_discriminator(real_human_data, fake_generator_data):

    FOR each paired mini-batch of (real, fake) data:

        FEED real human gaze through Discriminator → get "realness" scores
        FEED fake generator gaze through Discriminator → get "realness" scores

        SET target for real data = 1  (should score as human)
        SET target for fake data = 0  (should score as machine)

        COMPUTE real loss   = how wrong was Discriminator on real data?
        COMPUTE fake loss   = how wrong was Discriminator on fake data?
        COMPUTE total loss  = real loss + fake loss

        COMPUTE gradient penalty on real data

        UPDATE Discriminator weights to reduce (total loss + gradient penalty)
        STEP the learning rate scheduler

    RETURN average loss, average real score, average fake score
```

---

## `irl_dcb/ppo.py`

```
SETUP:
    CREATE an optimiser for the Generator
    SET clipping threshold = 0.2  (policy cannot change by more than 20% per step)
    SET value loss weight  = 1.0
    SET entropy bonus weight = 0.01

---

FUNCTION re_evaluate_actions(states, actions_taken):
    FEED states through the current Generator
    GET the current probability distribution over all grid cells
    GET the current estimated state values
    COMPUTE the log-probability of the specific actions that were taken 
    COMPUTE entropy of the distribution (how uncertain/exploratory is the policy?)
    RETURN values, log_probabilities, entropy

(measures the change to decide how much to update

  Just measuring the change alone is not useful. The point is:

  IF the policy changed too much  →  clamp it (PPO clipping)
  IF the policy changed too little →  allow a bigger update)
)

---

FUNCTION update_generator(stored_trajectories):

    FOR 1 epoch:
        FOR each mini-batch from the stored trajectories:

            GET: states, actions taken, expected returns,
                 old log-probabilities, advantages

            CALL re_evaluate_actions → new log-probabilities, new values, entropy

            COMPUTE probability ratio:
                ratio = new_probability / old_probability
                (ratio > 1 means this action is now more likely than before)

            COMPUTE clipped policy loss (PPO trick):
                option A = ratio × advantage
                option B = CLAMP(ratio between 0.8 and 1.2) × advantage
                policy loss = −AVERAGE(MINIMUM of option A and option B)
                (the clamp prevents the policy from changing too drastically)

            COMPUTE value loss:
                how wrong was the Generator's prediction of future reward?

            COMPUTE entropy loss:
                −entropy × 0.01
                (reward the Generator for being slightly random = exploration)

            TOTAL loss = policy loss + value loss + entropy loss

            UPDATE Generator weights to reduce total loss

    RETURN average loss
```

---

---

# PHASE 3 — Evaluation & Metrics

---

## `irl_dcb/metrics.py`

```
FUNCTION compare_two_scanpaths(scanpath_1, scanpath_2, image_size):
    FORMAT both scanpaths as coordinate arrays
    CALL multimatch algorithm
    RETURN 4 similarity scores: [shape, direction, length, position]
        each score is between 0 (completely different) and 1 (identical)


FUNCTION compare_model_to_humans(human_scanpaths, predicted_scanpaths):
    FOR each predicted scanpath:
        FIND all real human scanpaths for the same image + target category
        COMPUTE similarity score against each human scanpath
        AVERAGE the scores
    AVERAGE across all predicted scanpaths
    RETURN mean [shape, direction, length, position] scores


FUNCTION align_two_sequences(predicted_labels, ground_truth_labels):
    BUILD a score table using dynamic programming:
        FOR each position in predicted vs each position in ground truth:
            BEST score = MAXIMUM of:
                match this position (score +1 if labels are the same, else +0)
                skip in predicted   (score +0)
                skip in ground truth(score +0)
    NORMALISE by the length of the longer sequence
    RETURN alignment score between 0 and 1


FUNCTION compute_sequence_similarity(predicted_scanpaths, cluster_labels):
    FOR each predicted scanpath:
        CONVERT each (x,y) fixation to the nearest image region label
        COMPARE this sequence to all ground-truth human sequences
            using the alignment function above
        AVERAGE the scores across all humans
    RETURN scores


FUNCTION compute_path_efficiency(scanpath, target_bounding_box):
    COMPUTE total path length (sum of distances between consecutive fixations)
    COMPUTE straight-line distance from start to target centre
    RETURN ratio = straight_line / total_path
        (1.0 = perfectly direct; lower = more winding)


FUNCTION compute_area_under_efficiency_curve(cdf):
    SUM all values in the curve
    RETURN the total  (higher = found target faster on average)


FUNCTION compute_mismatch_vs_humans(model_cdf, human_cdf):
    FOR each step:
        COMPUTE absolute difference between model and human probability
    SUM all differences
    RETURN total mismatch  (lower = more human-like)
```

---

## `irl_dcb/multimatch.py`

```
FUNCTION convert_to_polar(x, y):
    COMPUTE distance from origin = sqrt(x² + y²)
    COMPUTE angle from origin    = arctan(y / x)
    RETURN (distance, angle)


FUNCTION compute_angle_between_vectors(vector_1, vector_2):
    COMPUTE dot product of the two vectors
    DIVIDE by product of their lengths
    RETURN angle in degrees


FUNCTION build_scanpath_structure(fixation_list):
    FOR each consecutive pair of fixations:
        COMPUTE the eye movement (saccade) between them:
            direction (x and y components)
            length (how far the eye moved)
            angle (direction in polar coordinates)
    RETURN fixation positions + saccade vectors


FUNCTION simplify_scanpath(scanpath, min_duration, min_amplitude):
    REMOVE fixations that are too short in duration
    REMOVE eye movements that are too small in amplitude
    RE-COMPUTE the saccade vectors after removal
    RETURN the simplified scanpath


FUNCTION align_scanpaths(scanpath_1, scanpath_2):
    USE dynamic time warping to find the best matching
    between fixations of scanpath 1 and fixations of scanpath 2
    (even if they have different numbers of fixations)
    RETURN the matched pairs


FUNCTION score_vector_similarity(aligned_pairs):
    FOR each matched pair of eye movements:
        COMPUTE angle between the two saccade directions
    CONVERT angle to a 0-1 similarity score
    RETURN average score


FUNCTION score_length_similarity(aligned_pairs):
    FOR each matched pair of eye movements:
        COMPUTE difference in saccade length
    CONVERT to a 0-1 score
    RETURN average score


FUNCTION score_position_similarity(aligned_pairs):
    FOR each matched pair of fixations:
        COMPUTE distance between fixation locations
    CONVERT to a 0-1 score (normalised by image diagonal)
    RETURN average score


FUNCTION score_duration_similarity(aligned_pairs):
    FOR each matched pair of fixations:
        COMPUTE difference in fixation duration
    CONVERT to a 0-1 score
    RETURN average score


FUNCTION compare_two_scanpaths(scanpath_1, scanpath_2, image_size):
    SIMPLIFY both scanpaths
    ALIGN them using dynamic time warping
    COMPUTE all four similarity scores
    RETURN [vector_score, length_score, position_score, duration_score]
```

---

---

# PHASE 4 — Inference & Visualisation

## `plot_scanpath.py` 

```
FUNCTION draw_scanpath(image, x_coords, y_coords, durations, bounding_box, title, save_path):

    DISPLAY the image as background

    CALCULATE circle size range based on min and max duration
        (short fixation = small circle, long fixation = large circle)

    FOR each pair of consecutive fixations:
        DRAW a yellow arrow from fixation i to fixation i+1
        (this shows the direction of eye movement)

    FOR each fixation:
        CALCULATE circle size proportional to how long the eye stayed there
        DRAW a yellow circle at the fixation location
        LABEL the circle with its order number (1, 2, 3, ...)

    IF a bounding box is provided:
        DRAW a yellow rectangle around the target object

    IF a save path is provided:
        SAVE the figure as a PNG file
    DISPLAY the figure on screen


--- Main (when run from command line) ---

READ arguments:
    path to scanpath data file
    path to image folder
    which target category to show
    which subject to show (optional)
    random trial or specific trial ID

LOAD all scanpaths from the data file
FILTER to the chosen target category
FILTER to the chosen subject (if specified)
PICK a random scanpath (or the specified one by ID)

GET the image name, category, and bounding box from the scanpath
BUILD the path to the image file

LOAD the image
GET the fixation x-coordinates, y-coordinates, and durations

CALL draw_scanpath()
```

---

---

# Full Pipeline Summary

```
1.  resize.py               → Resize all images to 512×320
2.  extract_DCBs_demo.py    → Compute HR and LR belief maps for each image
3.  hparams/...json         → Define all training settings
4.  config.py               → Load settings, make them accessible
5.  dataset.py              → Organise raw data into usable datasets
6.  data.py                 → Define how individual examples are loaded
7.  utils.py                → Helper tools used everywhere
8.  models.py               → Define the Generator and Discriminator networks
9.  environment.py          → Define the visual search simulation
10. builder.py              → Assemble all components
11. train.py                → Entry point — start training
12. trainer.py              → Master training loop (orchestrates everything)
13. gail.py                 → Train Discriminator: real vs fake gaze
14. ppo.py                  → Train Generator: produce more human-like gaze
15. metrics.py              → Measure how human-like the predictions are
16. multimatch.py           → Geometric scanpath comparison algorithm
17. plot_scanpath.py        → Draw a scanpath on top of an image
```
