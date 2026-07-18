"""
Inference for the segmentation U-Net.

Loads a trained checkpoint (with its normalisation stats), predicts tumour masks
for a set of slices, saves predicted masks (.npy) and overlay PNGs, and — when
ground-truth masks are present — reports Dice/IoU.

Usage:
    # evaluate on the held-out test split (reproduced from the checkpoint)
    python3 segmentation_inference.py --checkpoint model/segmentation_unet/best.pt --split test

    # or run on specific patients
    python3 segmentation_inference.py --checkpoint model/segmentation_unet/best.pt --patients 17904 19085
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segmentation_unet import CHANNELS, IMG_SIZE, SegmentationUNet, list_samples, split_patients

OUT_DIR = "./segmentation_predictions"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="./model/segmentation_unet/best.pt")
    p.add_argument("--data_dir", default=None, help="defaults to the data_dir stored in the checkpoint")
    p.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    p.add_argument("--patients", nargs="*", default=None, help="explicit patient ids (overrides --split)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = ckpt["hparams"]
    mean = np.asarray(ckpt["norm_mean"], np.float32).reshape(3, 1, 1)
    std = np.asarray(ckpt["norm_std"], np.float32).reshape(3, 1, 1)
    data_dir = args.data_dir or ckpt.get("data_dir", "./results")

    model = SegmentationUNet(3, int(hp["base_channels"]), float(hp.get("dropout", 0.0))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.patients:
        wanted = set(args.patients)
    elif args.split == "all":
        wanted = None
    else:
        tr, va, te = split_patients(data_dir, ckpt.get("n_val", 4), ckpt.get("n_test", 4),
                                    ckpt.get("split_seed", 0))
        wanted = set({"train": tr, "val": va, "test": te}[args.split])

    os.makedirs(args.out_dir, exist_ok=True)
    dices, ious = [], []
    n = 0
    for pid, prefix in list_samples(data_dir):
        if wanted is not None and pid not in wanted:
            continue
        x = np.stack([np.load(f"{prefix}_{c}.npy").astype(np.float32).reshape(IMG_SIZE, IMG_SIZE)
                      for c in CHANNELS], axis=0)
        xin = torch.from_numpy((x - mean) / std)[None].to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(xin))[0, 0].cpu().numpy()
        pred = (prob > args.threshold).astype(np.float32)

        name = os.path.basename(prefix)
        np.save(f"{args.out_dir}/{name}_pred.npy", pred)
        n += 1

        gt = None
        mask_path = f"{prefix}_mask.npy"
        if os.path.exists(mask_path):
            gt = (np.load(mask_path).astype(np.float32).reshape(IMG_SIZE, IMG_SIZE) > 0).astype(np.float32)
            inter = float((pred * gt).sum())
            dices.append((2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6))
            ious.append((inter + 1e-6) / (((pred + gt) > 0).sum() + 1e-6))

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(x[0], cmap="gray"); ax[0].set_title("Image")
        ax[1].imshow(x[1], cmap="hot"); ax[1].set_title("Anomaly")
        ax[2].imshow(prob, cmap="viridis"); ax[2].set_title("Predicted prob")
        ax[3].imshow(x[0], cmap="gray")
        ax[3].contour(pred, colors="red", linewidths=0.6)
        if gt is not None:
            ax[3].contour(gt, colors="cyan", linewidths=0.6)
        ax[3].set_title("pred (red) / GT (cyan)")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(f"{args.out_dir}/{name}_overlay.png", dpi=80, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {n} predictions to {args.out_dir}/")
    if dices:
        print(f"{len(dices)} slices with GT | "
              f"dice={np.mean(dices):.4f}+-{np.std(dices):.4f} | "
              f"iou={np.mean(ious):.4f}+-{np.std(ious):.4f}")


if __name__ == "__main__":
    main()
