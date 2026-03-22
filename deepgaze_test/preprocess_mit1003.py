"""
Preprocess MIT1003 dataset from local data for pysaliency.

Uses local MIT1003 data at C:\\Users\\chang\\Downloads\\MIT1003 to run
Octave fixation extraction and save as HDF5 files that pysaliency
can load directly (bypassing re-download and re-extraction).

Usage:
    python preprocess_mit1003.py

This only needs to be run ONCE. After that, evaluate_mit1003.py will load
from the cached HDF5 files instantly.
"""

import os
import sys
import glob
import shutil
import zipfile
import tempfile
import subprocess

import numpy as np
from scipy.io import loadmat
from natsort import natsorted
from pkg_resources import resource_string

import pysaliency
from pysaliency.datasets import FixationTrains, FileStimuli
from pysaliency.utils import MatlabOptions, filter_files, run_matlab_cmd

# Force Octave (MATLAB R2025b launcher exits immediately on Windows)
MatlabOptions.matlab_names = []

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MIT1003 = r'C:\Users\chang\Downloads\MIT1003'
DATASET_LOCATION = os.path.join(SCRIPT_DIR, 'datasets')
CACHE_DIR = os.path.join(DATASET_LOCATION, 'MIT1003_initial_fix')

# Check if already cached
if os.path.exists(os.path.join(CACHE_DIR, 'stimuli.hdf5')) and \
   os.path.exists(os.path.join(CACHE_DIR, 'fixations.hdf5')):
    print("Dataset already cached! Loading to verify...")
    stimuli, fixations = pysaliency.external_datasets.get_mit1003_with_initial_fixation(
        location=DATASET_LOCATION
    )
    print(f"  {len(stimuli)} images, {len(fixations.x)} fixations")
    print("Done. No preprocessing needed.")
    sys.exit(0)

print("=" * 60)
print("Preprocessing MIT1003 dataset from local data")
print("This runs Octave extraction (~15000 image-subject pairs).")
print("It will take ~30-60 minutes but only needs to run ONCE.")
print("=" * 60)

# Verify local data exists
for subdir in ('ALLSTIMULI', 'DATA'):
    path = os.path.join(LOCAL_MIT1003, subdir)
    if not os.path.isdir(path):
        print(f"ERROR: {path} not found!")
        sys.exit(1)

os.makedirs(CACHE_DIR, exist_ok=True)

# --- Set up working directory ---
temp_dir = tempfile.mkdtemp(prefix='mit1003_preprocess_')
print(f"\nWorking directory: {temp_dir}")

junctions = []
try:
    # Create directory junctions to local data (fast, no data copied)
    for dirname in ('ALLSTIMULI', 'DATA'):
        link = os.path.join(temp_dir, dirname)
        target = os.path.join(LOCAL_MIT1003, dirname)
        subprocess.run(
            ['cmd', '/c', 'mklink', '/J', link, target],
            check=True, capture_output=True
        )
        junctions.append(link)

    # Extract or copy DatabaseCode (needed by Octave's checkFixations)
    db_zip = os.path.join(LOCAL_MIT1003, 'DatabaseCode.zip')
    db_dir = os.path.join(LOCAL_MIT1003, 'DatabaseCode')
    if os.path.exists(db_zip):
        with zipfile.ZipFile(db_zip) as zf:
            namelist = filter_files(zf.namelist(), ['.svn', '__MACOSX', '.DS_Store'])
            zf.extractall(temp_dir, namelist)
    elif os.path.isdir(db_dir):
        shutil.copytree(db_dir, os.path.join(temp_dir, 'DatabaseCode'),
                        ignore=shutil.ignore_patterns('__MACOSX', '.DS_Store'))
    else:
        print("ERROR: Neither DatabaseCode.zip nor DatabaseCode/ found!")
        sys.exit(1)

    # --- Create stimuli ---
    print('\nCreating stimuli...')
    stimuli_src = os.path.join(temp_dir, 'ALLSTIMULI')

    # Copy stimulus images to cache directory (only .jpeg files, no __MACOSX)
    stimuli_cache = os.path.join(CACHE_DIR, 'stimuli')
    os.makedirs(stimuli_cache, exist_ok=True)

    images = glob.glob(os.path.join(stimuli_src, '*.jpeg'))
    image_names = natsorted([os.path.basename(img) for img in images])
    print(f"  Found {len(image_names)} images")

    for name in image_names:
        src = os.path.join(stimuli_src, name)
        dst = os.path.join(stimuli_cache, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    stimuli_filenames = [os.path.join(stimuli_cache, f) for f in image_names]
    stimuli = FileStimuli(stimuli_filenames)

    # --- Get subjects (filter __MACOSX, .DS_Store, and non-directory entries) ---
    data_dir = os.path.join(temp_dir, 'DATA')
    subjects = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and d not in ('__MACOSX', '.DS_Store')
    ])
    print(f"  Found {len(subjects)} subjects")

    # --- Write extract_fixations.m (pysaliency's Octave script) ---
    with open(os.path.join(temp_dir, 'extract_fixations.m'), 'wb') as f:
        f.write(resource_string('pysaliency.external_datasets',
                                'scripts/extract_fixations.m'))

    # --- Build Octave extraction commands ---
    out_path = 'extracted'
    os.makedirs(os.path.join(temp_dir, out_path))

    cmds = []
    total = len(image_names) * len(subjects)
    for n, stimulus in enumerate(image_names):
        for subject_id, subject in enumerate(subjects):
            # Use forward slashes for Octave compatibility on Windows
            subject_path = 'DATA/' + subject
            outfile = out_path + '/' + stimulus + '_' + subject + '.mat'
            idx = n * len(subjects) + subject_id
            cmds.append("fprintf('%d/%d\\r', {}, {});".format(idx, total))
            cmds.append("extract_fixations('{}', '{}', '{}');".format(
                stimulus, subject_path, outfile))

    with open(os.path.join(temp_dir, 'extract_all_fixations.m'), 'w') as f:
        for cmd in cmds:
            f.write(cmd + '\n')

    print(f'\nExtracting fixations with Octave ({total} pairs)...')
    print('This takes ~30-60 minutes. Progress shown below:')
    run_matlab_cmd('extract_all_fixations;', cwd=temp_dir)
    print('\nOctave extraction complete!')

    # --- Build FixationTrains (replicating pysaliency's logic exactly) ---
    print('Building fixation data...')

    xs, ys, ts, ns = [], [], [], []
    train_subjects = []
    train_durations = []

    for n, stimulus in enumerate(image_names):
        height, width = stimuli.sizes[n]

        for subject_id, subject in enumerate(subjects):
            outfile = stimulus + '_' + subject + '.mat'
            mat_path = os.path.join(temp_dir, out_path, outfile)
            mat_data = loadmat(mat_path)

            _xs = mat_data['fixations'][:, 0]
            _ys = mat_data['fixations'][:, 1]
            _ts = mat_data['starts'].flatten()
            _durations = mat_data['durations'].flatten()
            full_durations = _durations.copy()

            valid_indices = (
                (_xs > 0) & (_xs < width)
                & (_ys > 0) & (_ys < height))

            _xs = _xs[valid_indices]
            _ys = _ys[valid_indices]
            _ts = _ts[valid_indices]
            _durations = _durations[valid_indices]

            _ts = _ts / 240.0          # Eye Tracker rate = 240Hz
            _durations = _durations / 1000  # ms -> seconds

            # If first fixation was invalid, add central fixation
            # (matches pysaliency's include_initial_fixation=True behavior)
            if not valid_indices[0]:
                _xs = np.hstack(([0.5 * width], _xs))
                _ys = np.hstack(([0.5 * height], _ys))
                _ts = np.hstack(([0], _ts))
                _durations = np.hstack(([full_durations[0] / 1000], _durations))

            # first_fixation = 0: include initial fixation (for scanpath models)
            xs.append(_xs)
            ys.append(_ys)
            ts.append(_ts)
            ns.append(n)
            train_subjects.append(subject_id)
            train_durations.append(_durations)

    fixations = FixationTrains.from_fixation_trains(
        xs, ys, ts, ns, train_subjects,
        scanpath_fixation_attributes={'durations': train_durations},
        scanpath_attribute_mapping={'durations': 'duration'}
    )

    # --- Save HDF5 ---
    print('Saving to HDF5...')
    stimuli.to_hdf5(os.path.join(CACHE_DIR, 'stimuli.hdf5'))
    fixations.to_hdf5(os.path.join(CACHE_DIR, 'fixations.hdf5'))

    print(f"\nDone! Cached to {CACHE_DIR}")
    print(f"  {len(stimuli)} images")
    print(f"  {len(fixations.x)} fixations")
    print("\nYou can now run evaluate_mit1003.py (loading will be instant).")

finally:
    # Remove junctions first (does NOT delete target data)
    for link in junctions:
        if os.path.exists(link):
            subprocess.run(['cmd', '/c', 'rmdir', link], capture_output=True)
    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
