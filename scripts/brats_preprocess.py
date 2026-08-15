"""
Preprocess the ASNR-MICCAI BraTS 2023 GLI dataset into the same format/canvas as the
NFBS+IXI training data (skull-stripped, rot90-aligned, percentile [1,99] normalized),
so it can be used as a labeled anomalous test set for detection.py.

Raw layout (from the Kaggle mirror, after --unzip):
    DATASETS/BraTS_raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-XXXXX-YYY/
        BraTS-GLI-XXXXX-YYY-seg.nii
        BraTS-GLI-XXXXX-YYY-t1n.nii/<subject>_brain_t1.nii   (nested one level deeper)

BraTS volumes are already registered to the SRI24 atlas: 240x240x155, 1mm isotropic,
axcodes ('L','P','S') -> axis0=LR, axis1=AP, axis2=SI (axial).

Output, matching AnomalousMRIDataset(cleaned=True):
    DATASETS/BraTS/raw_cleaned/{5-digit id}.npy   -- (SI, 256, 192) float32 image, [0,1]
    DATASETS/BraTS/mask/{5-digit id}.npy          -- (SI, 256, 192) float32 binary mask
    DATASETS/BraTS/slices.json                    -- {id: [tumor_start, tumor_end]}
    DATASETS/BraTS/manifest.json                  -- {id: original BraTS folder name}
"""
import glob
import json
from pathlib import Path

import nibabel as nib
import numpy as np

SRC = Path("./DATASETS/BraTS_raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
DST = Path("./DATASETS/BraTS")
TARGET_AP = 256
TARGET_LR = 192
MARGIN = 6  # slack so linspace(start+5, stop-5, 4) has room


def load_nii(path):
    """`path` may be a real .nii file, or (in this mirror's inconsistent layout) a directory
    named *.nii that contains exactly one differently-named .nii file inside it."""
    if Path(path).is_file():
        return nib.load(path).get_fdata().astype(np.float32)
    matches = glob.glob(str(Path(path) / "*.nii"))
    if not matches:
        raise FileNotFoundError(path)
    return nib.load(matches[0]).get_fdata().astype(np.float32)


def anoddpm_normalize(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    fg = vol[mask]
    lo, hi = np.percentile(fg, 1), np.percentile(fg, 99)
    image = np.clip(vol, lo, hi)
    image = (image - lo) / (hi - lo)
    image[~mask] = 0
    return image.astype(np.float32)


def pad_crop_ap_lr(vol: np.ndarray) -> np.ndarray:
    """vol: (AP, SI, LR) -> pad AP to TARGET_AP, center-crop LR to TARGET_LR."""
    ap, si, lr = vol.shape
    pad_ap = max(0, TARGET_AP - ap)
    top, bot = pad_ap // 2, pad_ap - pad_ap // 2
    vol = np.pad(vol, ((top, bot), (0, 0), (0, 0)), mode="constant")
    ap = vol.shape[0]
    if ap > TARGET_AP:
        start = (ap - TARGET_AP) // 2
        vol = vol[start:start + TARGET_AP]

    lr = vol.shape[2]
    if lr > TARGET_LR:
        start = (lr - TARGET_LR) // 2
        vol = vol[:, :, start:start + TARGET_LR]
    elif lr < TARGET_LR:
        pad_lr = TARGET_LR - lr
        left, right = pad_lr // 2, pad_lr - pad_lr // 2
        vol = np.pad(vol, ((0, 0), (0, 0), (left, right)), mode="constant")
    return vol


def main():
    (DST / "raw_cleaned").mkdir(parents=True, exist_ok=True)
    (DST / "mask").mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(SRC.glob("BraTS-GLI-*"))
    print(f"found {len(patient_dirs)} BraTS patients")

    slices_manifest = {}
    id_manifest = {}
    skipped = 0

    for i, pdir in enumerate(patient_dirs):
        pid_raw = pdir.name  # e.g. BraTS-GLI-00000-000
        t1_path = pdir / f"{pid_raw}-t1n.nii"
        seg_path = pdir / f"{pid_raw}-seg.nii"
        if not seg_path.exists() or not t1_path.exists():
            skipped += 1
            continue

        t1 = load_nii(t1_path)
        seg = nib.load(str(seg_path)).get_fdata().astype(np.float32)

        # raw axes (L, P, S) -> (AP, SI, LR): axis1(P)->0, axis2(S)->1, axis0(L)->2
        t1 = t1.transpose(1, 2, 0)
        seg = seg.transpose(1, 2, 0)

        tumor_si = np.where(seg.sum(axis=(0, 2)) > 0)[0]
        if len(tumor_si) < 2 * MARGIN + 4:
            skipped += 1
            continue
        start, stop = int(tumor_si.min()), int(tumor_si.max()) + 1

        t1 = anoddpm_normalize(t1)
        t1 = pad_crop_ap_lr(t1)
        seg = pad_crop_ap_lr(seg)

        # (AP, SI, LR) -> (SI, AP, LR) to match AnomalousMRIDataset's image[slice_idx, ...] convention
        t1 = t1.transpose(1, 0, 2)
        seg = (seg.transpose(1, 0, 2) > 0).astype(np.float32)

        sid = f"{i:05d}"
        np.save(DST / "raw_cleaned" / f"{sid}.npy", t1)
        np.save(DST / "mask" / f"{sid}.npy", seg)
        slices_manifest[sid] = [start, stop]
        id_manifest[sid] = pid_raw

        if i % 100 == 0:
            print(f"{i}/{len(patient_dirs)} done ({sid} <- {pid_raw}, tumor slices {start}-{stop})")

    with open(DST / "slices.json", "w") as f:
        json.dump(slices_manifest, f)
    with open(DST / "manifest.json", "w") as f:
        json.dump(id_manifest, f)

    print(f"\nDone: {len(slices_manifest)} patients written, {skipped} skipped (missing seg / tiny tumor)")


if __name__ == "__main__":
    main()
