import pickle
import numpy as np
from pathlib import Path
from skimage.transform import resize


curr_folder = Path(__file__).parent
root = curr_folder.parent
src = root / "DATASETS" / "IXI"
dst = root / "DATASETS" / "Train"


def anoddpm_normalize(vol: np.ndarray) -> np.ndarray:
    """Percentile min-max within the brain (foreground): clip brain intensities to
    [1st, 99th] percentile and scale to [0, 1]; background stays 0. Standard for
    skull-stripped brain MRI -- robust to outliers, keeps a natural T1 look
    (white matter bright, gray matter mid, CSF dark)."""
    vol = vol.astype(np.float32)
    mask = vol > 0
    fg = vol[mask]
    lo, hi = np.percentile(fg, 1), np.percentile(fg, 99)
    image = np.clip(vol, lo, hi)
    image = (image - lo) / (hi - lo)
    image[~mask] = 0
    return image.astype(np.float32)


for pkl_path in src.glob('*.pkl'):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # The pkl contains (image, label_map). We want just the image.
    if isinstance(data, tuple) and len(data) == 2:
        img, _label = data
    else:
        img = data  # if it's just an array

    img = np.asarray(img, dtype=np.float32)

    # Raw IXI axes: 0 = left-right, 1 = axial (superior-inferior), 2 = anterior-posterior.
    # NFBS is served by MRIDataset as [:, slice, :] = (AP vertical = 256, LR horizontal = 192),
    # anterior up. Rotate the (LR, AP) plane by +90 deg so AP becomes vertical and anterior
    # ends up at the top. Verified with the cerebellum landmark (this puts the cerebellum at
    # the bottom, matching NFBS). The original 180-deg rot90 left AP horizontal -> 90 deg off.
    img = np.rot90(img, 1, axes=(0, 2))   # (AP=224, SI=192, LR=160)

    # IXI is already 1mm isotropic, same as NFBS, and the brains are the same physical
    # size (~170-180 vox). So DO NOT resize/scale -- just black-pad the axial plane
    # (axis 0 = AP, axis 2 = LR) to 256 x 192, keeping the brain at NFBS scale.
    h, n_slices, w = img.shape            # (224, 192, 160)
    pad_h = 256 - h
    pad_top, pad_bot = pad_h // 2, pad_h - pad_h // 2
    pad_w = 192 - w
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    img = np.pad(img, ((pad_top, pad_bot), (0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)

    img = anoddpm_normalize(img)

    subj_id = pkl_path.stem   # e.g. 'subject_0'
    subj_dir = dst / subj_id
    subj_dir.mkdir(exist_ok=True)
    np.save(subj_dir / f'{subj_id}.npy', img)
    print(f'{subj_id}: shape={img.shape}')
