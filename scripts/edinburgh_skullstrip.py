"""
Skull-strip the 22 CCBS Edinburgh scans (for the ARGS=30 / NFBS+IXI model, which
was trained on skull-stripped data and can't be fairly evaluated on the original
skull-included Edinburgh set).

Skull mask = sum of SPM tissue-probability maps (c1=GM, c2=WM, c3=CSF) > 0.5,
already present per-patient in tissue_classes/. Applied to the raw T1
(10_COR_.../anon_*.nii), then transposed to match the EXACT orientation already
used by the existing (skull-included) raw_cleaned/mask npys -- verified by
overlaying identical slices -- so the pre-existing tumor masks can be reused
unchanged (just center-cropped to the same width).

Writes ./DATASETS/Edinburgh_noskull/{raw_cleaned,mask}/{id}.npy + slices.json,
in the same format AnomalousMRIDataset expects for a custom (non-Edinburgh-
hardcoded) dataset.
"""
import glob
import json
import os

import nibabel as nib
import numpy as np

SRC = "./DATASETS/CancerousDataset/EdinburghDataset"
DST = "./DATASETS/Edinburgh_noskull"
TARGET_LR = 192

# same tumour ranges already hardcoded in dataset.AnomalousMRIDataset
SLICES = {
    "17904": (165, 205), "18428": (177, 213), "18582": (160, 190), "18638": (160, 212),
    "18675": (140, 200), "18716": (135, 190), "18756": (150, 205), "18863": (130, 190),
    "18886": (120, 180), "18975": (170, 194), "19015": (158, 195), "19085": (155, 195),
    "19275": (184, 213), "19277": (158, 209), "19357": (158, 210), "19398": (164, 200),
    "19423": (142, 200), "19567": (160, 200), "19628": (147, 210), "19691": (155, 200),
    "19723": (140, 170), "19849": (150, 180),
}


def anoddpm_normalize(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    fg = vol[mask]
    lo, hi = np.percentile(fg, 1), np.percentile(fg, 99)
    image = np.clip(vol, lo, hi)
    image = (image - lo) / (hi - lo)
    image[~mask] = 0
    return image.astype(np.float32)


def center_crop_width(vol: np.ndarray, target: int) -> np.ndarray:
    w = vol.shape[2]
    if w <= target:
        return vol
    start = (w - target) // 2
    return vol[:, :, start:start + target]


def main():
    os.makedirs(f"{DST}/raw_cleaned", exist_ok=True)
    os.makedirs(f"{DST}/mask", exist_ok=True)

    slices_out = {}
    for pid in sorted(SLICES):
        t1_glob = glob.glob(f"{SRC}/{pid}/*/anon_*.nii")
        tissue_dir = f"{SRC}/{pid}/tissue_classes"
        t1_path = t1_glob[0]
        suffix = os.path.basename(t1_path)[len("anon_"):]

        raw = nib.load(t1_path).get_fdata().astype(np.float32)
        c1 = nib.load(f"{tissue_dir}/c1anon_{suffix}").get_fdata()
        c2 = nib.load(f"{tissue_dir}/c2anon_{suffix}").get_fdata()
        c3 = nib.load(f"{tissue_dir}/c3anon_{suffix}").get_fdata()
        brain_mask = (c1 + c2 + c3) > 0.5

        stripped = raw * brain_mask
        stripped = stripped.transpose(1, 2, 0)  # matches existing raw_cleaned/mask orientation

        stripped = anoddpm_normalize(stripped)
        stripped = center_crop_width(stripped, TARGET_LR)

        existing_mask = np.load(f"{SRC}/Anomalous-T1/mask/{pid}.npy")
        existing_mask = center_crop_width(existing_mask, TARGET_LR)

        np.save(f"{DST}/raw_cleaned/{pid}.npy", stripped)
        np.save(f"{DST}/mask/{pid}.npy", existing_mask)
        start, stop = SLICES[pid]
        slices_out[pid] = [start, stop]
        print(f"{pid}: stripped shape={stripped.shape}, mask shape={existing_mask.shape}")

    with open(f"{DST}/slices.json", "w") as f:
        json.dump(slices_out, f)
    print(f"\nDone: {len(slices_out)} patients written to {DST}/")


if __name__ == "__main__":
    main()
