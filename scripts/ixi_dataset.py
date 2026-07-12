import pickle
import numpy as np
from pathlib import Path
from skimage.transform import resize


curr_folder = Path(__file__).parent
root = curr_folder.parent
src = root / "DATASETS" / "IXI"
dst = root / "DATASETS" / "Train"


def anoddpm_normalize(vol: np.ndarray) -> np.ndarray:
    """Asymmetric mean/std window, computed over foreground (nonzero) voxels only
    so background doesn't skew the contrast window; background is forced back
    to exactly 0 afterward, since clipping against foreground-derived bounds
    would otherwise lift it above black."""
    vol = vol.astype(np.float32)
    mask = vol > 0
    foreground = vol[mask]
    image_mean, image_std = foreground.mean(), foreground.std()
    img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
    image = np.clip(vol, img_range[0], img_range[1])
    image = image / (img_range[1] - img_range[0])
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

    # Raw axes are (sagittal=160, axial=192, coronal=224) — axis 1 is already
    # axial (confirmed by visual inspection), so no transpose is needed.
    # AnoDDPM's MRIDataset expects each axial slice to be 256x192. Scale the
    # per-slice (axis 0, axis 2) plane by a single uniform factor (so we don't
    # stretch/distort the brain's proportions) and black-pad the remainder to
    # reach exactly 256x192, keeping the axial slice count (axis 1) unchanged.
    h, n_slices, w = img.shape
    scale = min(256 / h, 192 / w)
    new_h, new_w = round(h * scale), round(w * scale)
    img = resize(img, (new_h, n_slices, new_w), order=1, preserve_range=True, anti_aliasing=True)

    pad_h = 256 - new_h
    pad_top, pad_bot = pad_h // 2, pad_h - pad_h // 2
    pad_w = 192 - new_w
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    img = np.pad(img, ((pad_top, pad_bot), (0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)

    # rotate each axial slice (axes 0, 2) 180 degrees to match sub-r001s023's orientation
    img = np.rot90(img, 2, axes=(0, 2))

    img = anoddpm_normalize(img)

    subj_id = pkl_path.stem   # e.g. 'subject_0'
    subj_dir = dst / subj_id
    subj_dir.mkdir(exist_ok=True)
    np.save(subj_dir / f'{subj_id}.npy', img)
    print(f'{subj_id}: shape={img.shape}')