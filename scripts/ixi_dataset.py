"""
Convert IXI PNG slices into 3D .npy volumes that AnoDDPM's
MRIDataset can load.

Normalization:
    lo, hi = mean - 1*std, mean + 2*std
    vol = clip(vol, lo, hi) / (hi - lo)
Producing values roughly in [0, 1] (loader shifts to [-1, 1] later).
-----------------------------------------------------------------
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def natural_key(filename: str):
    """Sort slice_2.png before slice_10.png (not the other way around)."""
    parts = re.split(r'(\d+)', filename)
    return [int(p) if p.isdigit() else p for p in parts]


def load_subject_volume(subj_dir: Path) -> np.ndarray | None:
    """Load all PNGs in a subject folder and stack into (N, H, W)."""
    png_paths = sorted(subj_dir.glob('*.png'), key=lambda p: natural_key(p.name))
    if not png_paths:
        return None

    slices = []
    ref_shape = None
    for p in png_paths:
        arr = np.array(Image.open(p).convert('L'), dtype=np.float32)  # grayscale
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            print(f"  WARN: {p.name} shape {arr.shape} != {ref_shape}, skipping",
                  file=sys.stderr)
            continue
        slices.append(arr)

    if not slices:
        return None

    return np.stack(slices, axis=0)  # (N_slices, H, W)


def anoddpm_normalize(vol: np.ndarray) -> np.ndarray:
    """AnoDDPM's exact normalization: asymmetric mean/std window."""
    vol = vol.astype(np.float32)
    image_mean, image_std = vol.mean(), vol.std()
    img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
    image = np.clip(vol, img_range[0], img_range[1])
    image = image / (img_range[1] - img_range[0])
    return image.astype(np.float32)


def convert_subject(subj_dir: Path, out_root: Path) -> bool:
    subj_id = subj_dir.name

    volume = load_subject_volume(subj_dir)
    if volume is None:
        print(f"  SKIP {subj_id}: no valid PNGs")
        return False

    volume = anoddpm_normalize(volume)

    subj_out_dir = out_root / subj_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = subj_out_dir / f'{subj_id}.npy'
    np.save(out_path, volume)

    print(f"  OK   {subj_id}: shape={volume.shape} dtype={volume.dtype} "
          f"range=[{volume.min():.3f}, {volume.max():.3f}]")
    return True


def main():

    curr_folder = Path(__file__).parent
    root = curr_folder.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input_dir', type=Path, nargs='?', default=Path(root / 'DATASETS/IXI'),
                    help='Root dir containing per-subject PNG folders '
                         '(e.g. image-slice-t1/) (default: %(default)s)')
    ap.add_argument('output_dir', type=Path, nargs='?', default=Path(root / 'DATASETS/Train'),
                    help='Destination Train/ folder for .npy volumes (default: %(default)s)')


    args = ap.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"error: input_dir does not exist: {args.input_dir}")

    subject_dirs = [d for d in args.input_dir.iterdir() if d.is_dir()]

    print(f"Found {len(subject_dirs)} subject folders in {args.input_dir}")
    print(f"Writing to {args.output_dir}")

    ok = 0
    for i, subj_dir in enumerate(subject_dirs, 1):
        print(f"[{i}/{len(subject_dirs)}] {subj_dir.name}")
        if convert_subject(subj_dir, args.output_dir):
            ok += 1

    print(f"\nDone. Converted {ok}/{len(subject_dirs)} subjects.")


if __name__ == '__main__':
    main()