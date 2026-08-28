"""
Builds a per-patient results table comparing the raw MC inference (mean, K=6)
against the trained U-Net (model/segmentation_unet/best.pt), across all
current patients in ./results.

The inference Dice is recomputed directly from the already-saved
anomaly.npy/mask.npy (no model rerun needed -- dice_coeff ignores its `recon`
argument whenever `mse` is passed explicitly, which is how it's always
called here). The U-Net Dice requires one forward pass per slice through the
trained checkpoint.

Writes two CSVs (one row per patient, values averaged over that patient's
slices):
  ./metrics/per_patient_all.csv        -- every patient currently in results/
  ./metrics/per_patient_original800.csv -- only patients 00000-00800 (the
                                            original, smaller BraTS run,
                                            before the 442-patient expansion)

Usage:
    python3 scripts/build_full_results_table.py
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import dice_coeff
from segmentation_unet import IMG_SIZE, SegmentationUNet, list_samples

CHECKPOINT = "./model/segmentation_unet/best.pt"
DATA_DIR = "./results"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    channels = ckpt["channels"]
    hp = ckpt["hparams"]
    mean = np.asarray(ckpt["norm_mean"], dtype=np.float32).reshape(len(channels), 1, 1)
    std = np.asarray(ckpt["norm_std"], dtype=np.float32).reshape(len(channels), 1, 1)

    model = SegmentationUNet(len(channels), int(hp["base_channels"]), float(hp.get("dropout", 0.0))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    samples = list_samples(DATA_DIR)
    print(f"{len(samples)} slices found")

    per_patient_inf = defaultdict(list)
    per_patient_unet = defaultdict(list)

    with torch.no_grad():
        for i, (pid, prefix) in enumerate(samples):
            mask = np.load(f"{prefix}_mask.npy").astype(np.float32).reshape(1, 1, IMG_SIZE, IMG_SIZE)
            anomaly = np.load(f"{prefix}_anomaly.npy").astype(np.float32).reshape(1, 1, IMG_SIZE, IMG_SIZE)

            mask_t = torch.from_numpy((mask > 0).astype(np.float32))
            amap_bin = torch.from_numpy((anomaly > 0.5).astype(np.float32))
            inf_dice = dice_coeff(mask_t, mask_t, mask_t, mse=amap_bin).item()  # real/recon unused when mse given
            per_patient_inf[pid].append(inf_dice)

            x = np.stack([np.load(f"{prefix}_{c}.npy").astype(np.float32).reshape(IMG_SIZE, IMG_SIZE)
                          for c in channels], axis=0)
            x = (x - mean) / std
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            logits = model(x_t)
            pred_bin = (torch.sigmoid(logits) > 0.5).float().cpu()
            unet_dice = dice_coeff(mask_t, mask_t, mask_t, mse=pred_bin).item()
            per_patient_unet[pid].append(unet_dice)

            if i % 200 == 0:
                print(f"[{i}/{len(samples)}] {pid}")

    all_pids = sorted(per_patient_inf.keys())
    original_pids = [p for p in all_pids if int(p) <= 800]

    def write_csv(path, pids):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient", "n_slices", "inference_dice_mean", "unet_dice_mean"])
            for pid in pids:
                w.writerow([
                    pid, len(per_patient_inf[pid]),
                    f"{np.mean(per_patient_inf[pid]):.4f}",
                    f"{np.mean(per_patient_unet[pid]):.4f}",
                    ])

    os.makedirs("./metrics", exist_ok=True)
    write_csv("./metrics/per_patient_all.csv", all_pids)
    write_csv("./metrics/per_patient_original800.csv", original_pids)

    def summarize(pids, label):
        inf = [np.mean(per_patient_inf[p]) for p in pids]
        unet = [np.mean(per_patient_unet[p]) for p in pids]
        print(f"\n{label}: {len(pids)} patients")
        print(f"  inference dice: {np.mean(inf):.4f} +- {np.std(inf):.4f}")
        print(f"  unet dice:      {np.mean(unet):.4f} +- {np.std(unet):.4f}")

    summarize(all_pids, "ALL patients")
    summarize(original_pids, "ORIGINAL (00000-00800) patients")

    print("\nWrote ./metrics/per_patient_all.csv and ./metrics/per_patient_original800.csv")


if __name__ == "__main__":
    main()
