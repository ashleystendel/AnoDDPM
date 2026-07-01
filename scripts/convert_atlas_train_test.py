"""
Converts ATLAS 2.0 T1w scans into the folder/.npy format expected by
Output layout (per subject):
  DATASETS/Train/sub-<id>/sub-<id>.npy
  DATASETS/Test/sub-<id>/sub-<id>.npy

Each output .npy is a 3D (256, 233, 192) float32 volume
"""
import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ATLAS_TRAIN_DIR = ROOT_DIR / "DATASETS" / "ATLAS_2" / "data" / "train" / "derivatives" / "ATLAS"
OUT_TRAIN_DIR = ROOT_DIR / "DATASETS" / "Train"
OUT_TEST_DIR = ROOT_DIR / "DATASETS" / "Test"

TARGET_AXIS0 = 256
TARGET_AXIS2 = 192


def find_healthy_subjects(atlas_dir, lesion_voxel_threshold):
    """Returns list of (subject_id, t1w_path) for subjects whose lesion mask
    has fewer than lesion_voxel_threshold non-zero voxels."""
    healthy = []
    for subject_dir in sorted(atlas_dir.glob("sub-*")):
        anat_dir = subject_dir / "ses-1" / "anat"
        t1w_matches = list(anat_dir.glob("*_T1w.nii.gz"))
        mask_matches = list(anat_dir.glob("*mask.nii.gz"))
        if not t1w_matches or not mask_matches:
            continue

        mask = np.asarray(nib.load(mask_matches[0]).dataobj)
        if int(np.count_nonzero(mask)) < lesion_voxel_threshold:
            healthy.append((subject_dir.name, t1w_matches[0]))
    return healthy


def center_pad_or_crop(arr, axis, target):
    size = arr.shape[axis]
    if size == target:
        return arr
    if size < target:
        pad_before = (target - size) // 2
        pad_after = target - size - pad_before
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (pad_before, pad_after)
        return np.pad(arr, pad_width, mode="constant", constant_values=arr.min())

    crop_before = (size - target) // 2
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(crop_before, crop_before + target)
    return arr[tuple(slicer)]


def convert_subject(t1w_path, out_subject_dir):
    image = nib.load(t1w_path).get_fdata().astype(np.float32)

    image_mean = np.mean(image)
    image_std = np.std(image)
    img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
    image = np.clip(image, img_range[0], img_range[1])
    image = image / (img_range[1] - img_range[0])

    image = center_pad_or_crop(image, axis=0, target=TARGET_AXIS0)
    image = center_pad_or_crop(image, axis=2, target=TARGET_AXIS2)

    out_subject_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_subject_dir / f"{out_subject_dir.name}.npy", image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesion-voxel-threshold", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    healthy = find_healthy_subjects(ATLAS_TRAIN_DIR, args.lesion_voxel_threshold)
    print(f"Found {len(healthy)} healthy subjects (lesion voxels < {args.lesion_voxel_threshold})")

    random.Random(args.seed).shuffle(healthy)
    split_idx = round(len(healthy) * args.train_ratio)
    train_subjects, test_subjects = healthy[:split_idx], healthy[split_idx:]
    print(f"Train: {len(train_subjects)} subjects, Test: {len(test_subjects)} subjects")

    for subject_id, t1w_path in train_subjects:
        convert_subject(t1w_path, OUT_TRAIN_DIR / subject_id)
    for subject_id, t1w_path in test_subjects:
        convert_subject(t1w_path, OUT_TEST_DIR / subject_id)

    print("Done")


if __name__ == "__main__":
    main()