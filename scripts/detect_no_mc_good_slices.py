"""
Single-pass (no-MC) detection with ARGS=30 on the exact same slices already
processed with MC in ./results_good_slices/ -- for a fair MC vs. no-MC
comparison on identical inputs.

Reuses the *_image.npy / *_mask.npy already saved there (these are already in
the model's expected normalized/cropped tensor form), so no re-preprocessing
from raw BraTS volumes is needed -- only one forward_backward pass per slice
instead of the 6 used for MC.

Writes to ./results_good_slices_no_mc/ (never touches results_good_slices/ or
results/) and ./metrics/args30_good_slices_no_mc.csv.

Usage:
    python3 scripts/detect_no_mc_good_slices.py
"""
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from UNet import UNetModel
from helpers import load_checkpoint

SRC_DIR = "./results_good_slices"
OUT_DIR = "./results_good_slices_no_mc"
SAMPLE_DISTANCE = 250


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint("30", False, device)
    args = checkpoint["args"]
    print(f"args{args['arg_num']}, sample_distance={SAMPLE_DISTANCE} (no MC, single pass)")

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

    slices = sorted(glob.glob(f"{SRC_DIR}/*/*/*_image.npy"))
    print(f"found {len(slices)} slices to process")

    dice_data, ssim_data, iou_data, precision_data, recall_data, fpr_data, auc_data = [], [], [], [], [], [], []
    start_time = time.time()

    for i, image_path in enumerate(slices):
        prefix = image_path[: -len("_image.npy")]
        name = os.path.basename(prefix)  # e.g. 00107-slice=1
        pid, slice_part = name.split("-slice=")

        image = torch.from_numpy(np.load(image_path)).to(device).reshape(1, 1, *args["img_size"])
        mask = torch.from_numpy(np.load(f"{prefix}_mask.npy")).to(device).reshape(1, 1, *args["img_size"])

        with torch.no_grad():
            recon = diff.forward_backward(
                    unet, image, see_whole_sequence=None, t_distance=SAMPLE_DISTANCE, denoise_fn=args["noise_fn"]
                    )

        mse = (image - recon).square()
        fpr_curve, tpr_curve, _ = evaluation.ROC_AUC(mask.to(torch.uint8), mse)
        auc_data.append(evaluation.AUC_score(fpr_curve, tpr_curve))
        mse_thresh = (mse > 0.5).float()

        dice_data.append(evaluation.dice_coeff(image, recon, mask, mse=mse_thresh).cpu().item())
        ssim_data.append(
                evaluation.SSIM(
                        image.permute(0, 2, 3, 1).reshape(*args["img_size"], 1),
                        recon.permute(0, 2, 3, 1).reshape(*args["img_size"], 1),
                        )
                )
        precision_data.append(evaluation.precision(mask, mse_thresh).cpu().numpy())
        recall_data.append(evaluation.recall(mask, mse_thresh).cpu().numpy())
        iou_data.append(evaluation.IoU(mask, mse_thresh))
        fpr_data.append(evaluation.FPR(mask, mse_thresh).cpu().numpy())

        out_path = Path(OUT_DIR) / pid / slice_part
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / f"{name}_image.npy", image.cpu().numpy())
        np.save(out_path / f"{name}_mask.npy", mask.cpu().numpy())
        np.save(out_path / f"{name}_recon.npy", recon.cpu().numpy())
        np.save(out_path / f"{name}_anomaly.npy", mse.cpu().numpy())

        if i % 20 == 0:
            elapsed = time.time() - start_time
            remaining = (len(slices) - i - 1) * (elapsed / (i + 1)) if i else 0
            print(
                    f"[{i}/{len(slices)}] {name}  dice={dice_data[-1]:.4f}  "
                    f"elapsed={elapsed/3600:.2f}h  remaining={remaining/3600:.2f}h"
                    )

    print("\nOverall (no MC, single pass):")
    print(f"Dice: {np.mean(dice_data):.4f} +- {np.std(dice_data):.4f}")
    print(f"SSIM: {np.mean(ssim_data):.4f} +- {np.std(ssim_data):.4f}")
    print(f"IoU: {np.mean(iou_data):.4f} +- {np.std(iou_data):.4f}")
    print(f"Precision: {np.mean(precision_data):.4f} +- {np.std(precision_data):.4f}")
    print(f"Recall: {np.mean(recall_data):.4f} +- {np.std(recall_data):.4f}")
    print(f"FPR: {np.mean(fpr_data):.4f} +- {np.std(fpr_data):.4f}")
    print(f"AUC: {np.mean(auc_data):.4f} +- {np.std(auc_data):.4f}")

    os.makedirs("./metrics", exist_ok=True)
    with open("./metrics/args30_good_slices_no_mc.csv", "w") as f:
        f.write("dice,ssim,iou,precision,recall,fpr,auc\n")
        for metric in [dice_data, ssim_data, iou_data, precision_data, recall_data, fpr_data, auc_data]:
            f.write(f"{np.mean(metric):.4f} +- {np.std(metric):.4f},")


if __name__ == "__main__":
    main()
