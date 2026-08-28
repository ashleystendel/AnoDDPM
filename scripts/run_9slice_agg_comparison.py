"""
Runs the MC-count x aggregation-method comparison on 9 hand-picked BraTS slices
(one slice per patient -- each patient's own largest-tumour slice, spanning
small to large tumours), for n_samples_unc in {4, 6, 8} and all 4 aggregation
methods (mean/median/min_error/majority_vote) -- 12 combinations per slice,
108 rows total.

For a given (slice, K) pair, the K MC passes are drawn ONCE and all 4
aggregation methods are computed from that same set (reuses
detection._aggregate_mc), so the 4 methods are directly comparable and we
don't waste 4x the compute re-drawing passes per method.

Uses ARGS=30 (NFBS+IXI, orientation/normalization-fix validated -- same
checkpoint used throughout the main ablation results), sample_distance=250.

Reads already-preprocessed image.npy/mask.npy from results_good_slices/ (no
re-preprocessing from raw BraTS needed).

Writes:
  ./results_9slice_comparison/<pid>/<K>/<method>/  -- heatmap png, npys
  ./metrics/agg_comparison_9slices.csv             -- one row per (pid, K, method)

Usage:
    python3 scripts/run_9slice_agg_comparison.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation
from detection import _aggregate_mc, AGGREGATION_METHODS
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from UNet import UNetModel
from helpers import load_checkpoint

SRC_DIR = "./results_good_slices"
OUT_DIR = "./results_9slice_comparison"
SAMPLE_DISTANCE = 250
K_VALUES = [4, 6, 8]

# (patient id, slice index) -- each patient's own largest-tumour slice,
# spanning ~1.1k-5.2k tumour pixels, confirmed visually with the user
CHOSEN_SLICES = [
    ("00375", "2"),
    ("00329", "2"),
    ("00103", "2"),
    ("00576", "0"),
    ("00033", "1"),
    ("00386", "1"),
    ("00164", "1"),
    ("00408", "0"),
    ("00494", "0"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = load_checkpoint("30", False, device)
    args = checkpoint["args"]
    print(f"args{args['arg_num']}, sample_distance={SAMPLE_DISTANCE}")

    unet = UNetModel(
            args["img_size"][0], args["base_channels"], channel_mults=args["channel_mults"], in_channels=1
            )
    unet.load_state_dict(checkpoint["ema"])
    unet.to(device)
    unet.eval()

    betas = get_beta_schedule(args["T"], args["beta_schedule"])
    diff = GaussianDiffusionModel(
            args["img_size"], betas, loss_weight=args["loss_weight"],
            loss_type=args["loss-type"], noise=args["noise_fn"], img_channels=1
            )

    rows = []  # (pid, k, method, dice, ssim, iou, precision, recall, fpr, auc)
    n_total = len(CHOSEN_SLICES) * len(K_VALUES)
    start_time = time.time()
    done = 0

    for pid, sl in CHOSEN_SLICES:
        name = f"{pid}-slice={sl}"
        prefix = f"{SRC_DIR}/{pid}/{sl}/{name}"
        image = torch.from_numpy(np.load(f"{prefix}_image.npy")).to(device).reshape(1, 1, *args["img_size"])
        mask = torch.from_numpy(np.load(f"{prefix}_mask.npy")).to(device).reshape(1, 1, *args["img_size"])

        for k in K_VALUES:
            with torch.no_grad():
                outputs = torch.stack(
                        [
                            diff.forward_backward(
                                    unet, image, see_whole_sequence=None,
                                    t_distance=SAMPLE_DISTANCE, denoise_fn=args["noise_fn"],
                                    )
                            for _ in range(k)
                            ], dim=0
                        )

            agg_results = _aggregate_mc(outputs, image)
            unc = outputs.var(dim=0)

            for method in AGGREGATION_METHODS:
                recon, amap = agg_results[method]

                fpr_curve, tpr_curve, _ = evaluation.ROC_AUC(mask.to(torch.uint8), amap)
                auc = evaluation.AUC_score(fpr_curve, tpr_curve)
                amap_bin = (amap > 0.5).float()

                dice = evaluation.dice_coeff(image, recon.to(device), mask, mse=amap_bin).cpu().item()
                ssim = evaluation.SSIM(
                        image.permute(0, 2, 3, 1).reshape(*args["img_size"], 1),
                        recon.permute(0, 2, 3, 1).reshape(*args["img_size"], 1),
                        )
                precision = evaluation.precision(mask, amap_bin).cpu().item()
                recall = evaluation.recall(mask, amap_bin).cpu().item()
                iou = evaluation.IoU(mask, amap_bin)
                fpr = evaluation.FPR(mask, amap_bin).cpu().item()

                out_path = Path(OUT_DIR) / pid / str(k) / method
                out_path.mkdir(parents=True, exist_ok=True)
                np.save(out_path / f"{name}_image.npy", image.cpu().numpy())
                np.save(out_path / f"{name}_mask.npy", mask.cpu().numpy())
                np.save(out_path / f"{name}_recon.npy", recon.cpu().numpy())
                np.save(out_path / f"{name}_anomaly.npy", amap.cpu().numpy())
                np.save(out_path / f"{name}_unc.npy", unc.cpu().numpy())

                evaluation.heatmap(
                        image, recon.reshape(1, *args["img_size"]).to(device), mask,
                        str(out_path / f"{name}_heatmap.png")
                        )

                rows.append((pid, k, method, dice, ssim, iou, precision, recall, fpr, auc))

            done += 1
            elapsed = time.time() - start_time
            remaining = (n_total - done) * (elapsed / done)
            print(
                    f"[{done}/{n_total}] {name} K={k} done  "
                    f"elapsed={elapsed/60:.1f}m  remaining={remaining/60:.1f}m"
                    )

    os.makedirs("./metrics", exist_ok=True)
    with open("./metrics/agg_comparison_9slices.csv", "w") as f:
        f.write("patient,k,method,dice,ssim,iou,precision,recall,fpr,auc\n")
        for pid, k, method, dice, ssim, iou, precision, recall, fpr, auc in rows:
            f.write(f"{pid},{k},{method},{dice:.4f},{ssim:.4f},{iou:.4f},{precision:.4f},{recall:.4f},{fpr:.4f},{auc:.4f}\n")

    print(f"\nDone. Wrote {len(rows)} rows to ./metrics/agg_comparison_9slices.csv")
    print("\nMean Dice per (K, method):")
    for k in K_VALUES:
        for method in AGGREGATION_METHODS:
            dices = [r[3] for r in rows if r[1] == k and r[2] == method]
            print(f"  K={k:<2} {method:<14} dice={np.mean(dices):.4f} +- {np.std(dices):.4f}")


if __name__ == "__main__":
    main()
