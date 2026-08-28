"""
Runs inference (ARGS=30, K=6 MC, mean aggregation -- same as the main
experiment used throughout the presentation) on the 450 BraTS patients
(00801-01250) that were never processed in the original run (which only
covered 00000-00800).

Same selection rule as before (>500px tumour), but 2 best slices per
patient by tumour area (not 4, and not evenly-spaced) -- picked directly
from the ground-truth mask, no model run needed to decide this part.

442 of 450 patients have >=2 qualifying slices; 8 have none and are
skipped. 884 slices total.

Applies the exact same preprocessing transform as
dataset.AnomalousMRIDataset's custom_slices branch (CenterCrop(235) ->
Resize(img_size) -> ToTensor -> Normalize(0.5,0.5)) to each raw 2D slice
extracted directly from DATASETS/BraTS/raw_cleaned + mask, since these
patients have no pre-extracted per-slice npys yet.

Writes to ./results/<pid>/<0 or 1>/ (same layout/convention as the
original 00000-00800 run, so it can be merged with it) and appends to
./metrics/args30_remaining_brats.csv.

Usage:
    python3 scripts/run_remaining_brats.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from UNet import UNetModel
from helpers import load_checkpoint

SRC_DIR = "./DATASETS/BraTS"
OUT_DIR = "./results"
SAMPLE_DISTANCE = 250
K = 6
MIN_AREA = 500
PATIENT_RANGE = range(801, 1251)


def pick_top2_slices(mask_vol: np.ndarray):
    areas = (mask_vol > 0).sum(axis=(1, 2))
    good_idx = np.where(areas > MIN_AREA)[0]
    if len(good_idx) < 2:
        return []
    top2 = good_idx[np.argsort(areas[good_idx])[-2:]]
    return sorted(top2.tolist())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = load_checkpoint("30", False, device)
    args = checkpoint["args"]
    img_size = args["img_size"]
    print(f"args{args['arg_num']}, sample_distance={SAMPLE_DISTANCE}, K={K}, mean aggregation")

    unet = UNetModel(
            img_size[0], args["base_channels"], channel_mults=args["channel_mults"], in_channels=1
            )
    unet.load_state_dict(checkpoint["ema"])
    unet.to(device)
    unet.eval()

    betas = get_beta_schedule(args["T"], args["beta_schedule"])
    diff = GaussianDiffusionModel(
            img_size, betas, loss_weight=args["loss_weight"],
            loss_type=args["loss-type"], noise=args["noise_fn"], img_channels=1
            )

    transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.CenterCrop(235),
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
            )

    # figure out the full job list up front so we can print an accurate ETA
    jobs = []  # (pid, local_idx, slice_idx)
    for i in PATIENT_RANGE:
        pid = f"{i:05d}"
        mask_path = f"{SRC_DIR}/mask/{pid}.npy"
        if not os.path.exists(mask_path):
            continue
        mask_vol = np.load(mask_path)
        top2 = pick_top2_slices(mask_vol)
        for local_idx, slice_idx in enumerate(top2):
            jobs.append((pid, local_idx, slice_idx))

    print(f"{len(jobs)} slices to process across {len(set(j[0] for j in jobs))} patients")

    heatmap_dir = Path(f'./diffusion-training-images/ARGS={args["arg_num"]}/mean/Anomalous-heatmaps')
    uncertainty_dir = Path(f'./diffusion-training-images/ARGS={args["arg_num"]}/mean/uncertainty')
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    start_time = time.time()
    cur_pid = None
    image_vol = mask_vol = None

    for done, (pid, local_idx, slice_idx) in enumerate(jobs):
        if pid != cur_pid:
            image_vol = np.load(f"{SRC_DIR}/raw_cleaned/{pid}.npy")
            mask_vol = np.load(f"{SRC_DIR}/mask/{pid}.npy")
            cur_pid = pid

        img_slice = image_vol[slice_idx, ...].astype(np.float32)
        mask_slice = mask_vol[slice_idx, ...].astype(np.float32)

        image = transform(img_slice).unsqueeze(0).to(device)
        mask = (transform(mask_slice) > 0).float().unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = torch.stack(
                    [
                        diff.forward_backward(
                                unet, image, see_whole_sequence=None,
                                t_distance=SAMPLE_DISTANCE, denoise_fn=args["noise_fn"],
                                )
                        for _ in range(K)
                        ], dim=0
                    )

        recon = outputs.mean(dim=0)
        amap = (image.unsqueeze(0) - outputs).square().mean(dim=0)
        unc = outputs.var(dim=0)

        fpr_curve, tpr_curve, _ = evaluation.ROC_AUC(mask.to(torch.uint8), amap)
        auc = evaluation.AUC_score(fpr_curve, tpr_curve)
        amap_bin = (amap > 0.5).float()

        dice = evaluation.dice_coeff(image, recon.to(device), mask, mse=amap_bin).cpu().item()
        ssim = evaluation.SSIM(
                image.permute(0, 2, 3, 1).reshape(*img_size, 1),
                recon.permute(0, 2, 3, 1).reshape(*img_size, 1),
                )
        precision = evaluation.precision(mask, amap_bin).cpu().item()
        recall = evaluation.recall(mask, amap_bin).cpu().item()
        iou = evaluation.IoU(mask, amap_bin)
        fpr = evaluation.FPR(mask, amap_bin).cpu().item()

        name = f"{pid}-slice={local_idx}"
        out_path = Path(OUT_DIR) / pid / str(local_idx)
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / f"{name}_image.npy", image.cpu().numpy())
        np.save(out_path / f"{name}_mask.npy", mask.cpu().numpy())
        np.save(out_path / f"{name}_recon.npy", recon.cpu().numpy())
        np.save(out_path / f"{name}_anomaly.npy", amap.cpu().numpy())
        np.save(out_path / f"{name}_unc.npy", unc.cpu().numpy())

        evaluation.heatmap(
                image, recon.reshape(1, *img_size).to(device), mask,
                str(heatmap_dir / f"{name}.png")
                )
        torch.save(
                {"outputs": outputs.cpu(), "unc": unc.cpu()},
                str(uncertainty_dir / f"{name}_pred_x0.pt")
                )

        rows.append((pid, slice_idx, dice, ssim, iou, precision, recall, fpr, auc))

        if done % 10 == 0:
            elapsed = time.time() - start_time
            remaining = (len(jobs) - done - 1) * (elapsed / (done + 1)) if done else 0
            print(
                    f"[{done + 1}/{len(jobs)}] {name} (raw slice {slice_idx})  dice={dice:.4f}  "
                    f"elapsed={elapsed/3600:.2f}h  remaining={remaining/3600:.2f}h"
                    )

    os.makedirs("./metrics", exist_ok=True)
    with open("./metrics/args30_remaining_brats.csv", "w") as f:
        f.write("patient,slice_idx,dice,ssim,iou,precision,recall,fpr,auc\n")
        for pid, slice_idx, dice, ssim, iou, precision, recall, fpr, auc in rows:
            f.write(f"{pid},{slice_idx},{dice:.4f},{ssim:.4f},{iou:.4f},{precision:.4f},{recall:.4f},{fpr:.4f},{auc:.4f}\n")

    dices = [r[2] for r in rows]
    print(f"\nDone. {len(rows)} slices. Overall Dice: {np.mean(dices):.4f} +- {np.std(dices):.4f}")


if __name__ == "__main__":
    main()
