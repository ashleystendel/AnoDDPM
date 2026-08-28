"""
For each of the 9 hand-picked slices, builds one figure in the exact style
already used for single-method uncertainty panels:
  row 0: Image | Ground Truth Mask | MC Reconstruction Variance
  rows 1-4: one row per aggregation method (mean/median/min_error/majority_vote):
            Reconstruction (titled with SSIM) | Anomaly Scaled | Anomaly Thresh (titled with Dice)

All computed at K=6 (the recommended MC count -- best Dice/AUC/IoU/Precision
of the sweep in metrics/agg_comparison_9slices.csv).

Reads from ./results_9slice_comparison/<pid>/6/<method>/ (written by
run_9slice_agg_comparison.py) and ./metrics/agg_comparison_9slices.csv for
the Dice/SSIM scores.

Writes ./presentation_images/agg_comparison/<pid>_agg_grid_k6.png
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = "./results_9slice_comparison"
OUT_DIR = "./presentation_images/agg_comparison"
K = 6
METHODS = ("mean", "median", "min_error", "majority_vote")

CHOSEN_SLICES = [
    ("00375", "2"), ("00329", "2"), ("00103", "2"), ("00576", "0"), ("00033", "1"),
    ("00386", "1"), ("00164", "1"), ("00408", "0"), ("00494", "0"),
]


def load_metrics_table():
    table = {}
    with open("./metrics/agg_comparison_9slices.csv") as f:
        for row in csv.DictReader(f):
            table[(row["patient"], int(row["k"]), row["method"])] = row
    return table


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics_table = load_metrics_table()

    for pid, sl in CHOSEN_SLICES:
        name = f"{pid}-slice={sl}"

        image = np.load(f"{SRC_DIR}/{pid}/{K}/mean/{name}_image.npy").squeeze()
        mask = np.load(f"{SRC_DIR}/{pid}/{K}/mean/{name}_mask.npy").squeeze()
        unc = np.load(f"{SRC_DIR}/{pid}/{K}/mean/{name}_unc.npy").squeeze()
        unc_scaled = (unc / max(unc.max(), 1e-8)) * 2 - 1

        panels = [image, mask, unc_scaled]
        titles = ["Image", "Ground Truth Mask", "MC Reconstruction Variance"]

        for method in METHODS:
            recon = np.load(f"{SRC_DIR}/{pid}/{K}/{method}/{name}_recon.npy").squeeze()
            amap = np.load(f"{SRC_DIR}/{pid}/{K}/{method}/{name}_anomaly.npy").squeeze()
            row = metrics_table[(pid, K, method)]

            amap_scaled = (amap * 2) - 1
            amap_thresh = (amap_scaled > 0).astype(np.float32) * 2 - 1

            panels += [recon, amap_scaled, amap_thresh]
            titles += [
                f"{method} Reconstruction\nSSIM={float(row['ssim']):.3f}",
                f"{method} Anomaly Scaled",
                f"{method} Anomaly Thresh\nDice={float(row['dice']):.3f}",
                ]

        fig, axes = plt.subplots(5, 3, figsize=(12, 19))
        for idx, (ax, panel, title) in enumerate(zip(axes.flat, panels, titles)):
            ax.imshow(panel, cmap="gray", vmin=-1, vmax=1)
            if idx == 0:  # Image panel: overlay tumour contour
                ax.contour(mask > 0, colors="red", linewidths=1.5)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"Patient {pid}, slice {sl}  (K={K} MC passes)", fontsize=14)
        plt.tight_layout()
        out_path = f"{OUT_DIR}/{pid}_agg_grid_k6.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
