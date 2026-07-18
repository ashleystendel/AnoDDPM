"""
Supervised segmentation U-Net (stage 2 of the pipeline).

Takes a 3-channel input built from the AnoDDPM detection outputs:
    channel 0: raw MRI slice (image)
    channel 1: anomaly map
    channel 2: MC uncertainty map
and predicts a per-pixel tumour mask, trained against the ground-truth masks of
the CCBS Edinburgh patients.

Input channels are produced by detection.py, e.g.:
    python detection.py 28 --uncertainty --save-output --sample-distance 250 --n-samples-unc 6
which writes, per slice:
    results/<patient>/<slice>/<patient>-slice=<n>_{image,anomaly,unc,mask}.npy
each of shape (1, 1, 256, 256).

The split is patient-level (no patient appears in more than one of train/val/test).
Per-channel normalisation stats are computed on the TRAIN split only and stored
inside the checkpoint so optuna/inference reproduce them exactly.

No data augmentation (more real data to be added later).

Usage:
    python3 segmentation_unet.py --epochs 200
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "./results"
CHECKPOINT_DIR = "./model/segmentation_unet"
CHANNELS = ("image", "anomaly", "unc")
IMG_SIZE = 256


# ------------------------------------------------------------------------- data
def list_samples(data_dir):
    """Return [(patient_id, prefix), ...] for every slice under data_dir."""
    samples = []
    for f in sorted(glob.glob(f"{data_dir}/*/*/*_image.npy")):
        prefix = f[: -len("_image.npy")]
        patient_id = os.path.relpath(f, data_dir).split(os.sep)[0]
        samples.append((patient_id, prefix))
    return samples


def split_patients(data_dir, n_val, n_test, seed=0):
    """Patient-level train/val/test split (deterministic given seed)."""
    patients = sorted({pid for pid, _ in list_samples(data_dir)})
    if len(patients) < n_val + n_test + 1:
        raise RuntimeError(f"Only {len(patients)} patients found in {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(patients)
    test = sorted(patients[:n_test])
    val = sorted(patients[n_test:n_test + n_val])
    train = sorted(patients[n_test + n_val:])
    return train, val, test


def compute_norm_stats(data_dir, patient_ids):
    """Per-channel mean/std over the given (training) patients."""
    patient_ids = set(patient_ids)
    s = np.zeros(3, dtype=np.float64)
    sq = np.zeros(3, dtype=np.float64)
    n = 0
    for pid, prefix in list_samples(data_dir):
        if pid not in patient_ids:
            continue
        for c, name in enumerate(CHANNELS):
            a = np.load(f"{prefix}_{name}.npy").astype(np.float64).ravel()
            s[c] += a.sum()
            sq[c] += (a * a).sum()
        n += a.size
    if n == 0:
        raise RuntimeError("No training samples found for normalisation stats")
    mean = s / n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class SegmentationDataset(Dataset):
    """3-channel (image, anomaly, unc) input + binary tumour mask target."""

    def __init__(self, data_dir, patient_ids, norm_mean, norm_std):
        patient_ids = set(patient_ids)
        self.samples = [p for p in list_samples(data_dir) if p[0] in patient_ids]
        self.mean = np.asarray(norm_mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(norm_std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def _load(self, prefix, name):
        return np.load(f"{prefix}_{name}.npy").astype(np.float32).reshape(IMG_SIZE, IMG_SIZE)

    def __getitem__(self, idx):
        _, prefix = self.samples[idx]
        x = np.stack([self._load(prefix, name) for name in CHANNELS], axis=0)  # (3,H,W)
        y = (self._load(prefix, "mask") > 0).astype(np.float32)[None]          # (1,H,W)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(y)


# ------------------------------------------------------------------------ model
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SegmentationUNet(nn.Module):
    """Standard 2D U-Net: 3-channel input -> 1-channel mask logits."""

    def __init__(self, in_channels=3, base_channels=32, dropout=0.0):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c, dropout)
        self.enc2 = ConvBlock(c, c * 2, dropout)
        self.enc3 = ConvBlock(c * 2, c * 4, dropout)
        self.enc4 = ConvBlock(c * 4, c * 8, dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 16, dropout)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8, dropout)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4, dropout)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2, dropout)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c, dropout)
        self.out_conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ------------------------------------------------------------------ loss/metrics
def dice_loss(logits, target, smooth=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=[1, 2, 3])
    union = probs.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    dices, ious = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = (torch.sigmoid(model(x)) > threshold).float()
        inter = (preds * y).sum(dim=[1, 2, 3])
        union = preds.sum(dim=[1, 2, 3]) + y.sum(dim=[1, 2, 3])
        u = union - inter
        dices.append((2 * inter + 1e-6) / (union + 1e-6))
        ious.append((inter + 1e-6) / (u + 1e-6))
    if not dices:
        return {"dice": 0.0, "iou": 0.0}
    return {"dice": torch.cat(dices).mean().item(), "iou": torch.cat(ious).mean().item()}


# --------------------------------------------------------------------- training
def train_model(hp, train_ids, val_ids, data_dir=DATA_DIR, epochs=200, device=None,
                checkpoint_path=None, extra=None, seed=0, verbose=True):
    """Train the U-Net for `epochs`, tracking the best val Dice.

    hp keys: lr, base_channels, batch_size, dropout, weight_decay, pos_weight.
    Returns the best validation Dice. If checkpoint_path is given, the best model
    (plus norm stats, hparams and `extra`) is saved there.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mean, std = compute_norm_stats(data_dir, train_ids)
    train_set = SegmentationDataset(data_dir, train_ids, mean, std)
    val_set = SegmentationDataset(data_dir, val_ids, mean, std)
    bs = int(hp["batch_size"])
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=len(train_set) > bs)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)

    model = SegmentationUNet(3, int(hp["base_channels"]), float(hp.get("dropout", 0.0))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp.get("weight_decay", 0.0)))
    pos_weight = torch.tensor([float(hp.get("pos_weight", 1.0))], device=device)

    best = -1.0
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        metrics = evaluate(model, val_loader, device)
        if metrics["dice"] > best:
            best = metrics["dice"]
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                ckpt = {
                        "model_state_dict": model.state_dict(), "hparams": dict(hp),
                        "norm_mean": mean, "norm_std": std, "epoch": epoch, "val_dice": best,
                        }
                if extra:
                    ckpt.update(extra)
                torch.save(ckpt, checkpoint_path)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}: train_loss={np.mean(losses):.4f} "
                  f"val_dice={metrics['dice']:.4f} val_iou={metrics['iou']:.4f} (best {best:.4f})")
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--n_val", type=int, default=4)
    p.add_argument("--n_test", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train_ids, val_ids, test_ids = split_patients(args.data_dir, args.n_val, args.n_test, args.seed)
    print(f"patients -> train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")
    print(f"  val:  {val_ids}")
    print(f"  test: {test_ids}")

    hp = dict(lr=args.lr, base_channels=args.base_channels, batch_size=args.batch_size,
              dropout=args.dropout, weight_decay=args.weight_decay, pos_weight=args.pos_weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = f"{CHECKPOINT_DIR}/best.pt"
    extra = dict(n_val=args.n_val, n_test=args.n_test, split_seed=args.seed, data_dir=args.data_dir)

    best_val = train_model(hp, train_ids, val_ids, args.data_dir, args.epochs, device,
                           ckpt_path, extra=extra, seed=args.seed)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SegmentationUNet(3, hp["base_channels"], hp["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_set = SegmentationDataset(args.data_dir, test_ids, ckpt["norm_mean"], ckpt["norm_std"])
    test_metrics = evaluate(model, DataLoader(test_set, batch_size=args.batch_size), device)
    print(f"\nBest val dice: {best_val:.4f}")
    print(f"TEST: dice={test_metrics['dice']:.4f} iou={test_metrics['iou']:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
