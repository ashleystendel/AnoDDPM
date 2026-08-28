"""
Retrains the refinement U-Net on results_filtered500/ (only slices with
tumor area >= 500px, symlinked from ./results) using the best hyperparameters
already found by the Optuna search (model/segmentation_unet/best.pt), so we
don't need to re-run the full 30-trial search on a very similar dataset.

Same patient-level held-out test set size (30/30) as the previous run, for a
directly comparable number.

Usage:
    python3 scripts/retrain_unet_filtered.py
"""
import os
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation_unet import split_patients, SegmentationDataset, SegmentationUNet, train_model, IMG_SIZE
from evaluation import dice_coeff, IoU, precision, ROC_AUC, AUC_score

DATA_DIR = "./results_filtered500"
CKPT_OUT = "./model/segmentation_unet/best_filtered500.pt"
EPOCHS = 150
N_VAL = 30
N_TEST = 30
SEED = 0

HP = {
    'lr': 0.0006662943794784631, 'base_channels': 32, 'batch_size': 8,
    'dropout': 0.036873790804413206, 'weight_decay': 0.00020170600873882494,
    'pos_weight': 8.698346652292466,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, val_ids, test_ids = split_patients(DATA_DIR, N_VAL, N_TEST, SEED)
    print(f"patients -> train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")

    extra = dict(n_val=N_VAL, n_test=N_TEST, split_seed=SEED, data_dir=DATA_DIR)
    best_val = train_model(
            HP, train_ids, val_ids, DATA_DIR, EPOCHS, device,
            checkpoint_path=CKPT_OUT, extra=extra, seed=SEED,
            )
    print(f"\nRetrained best val dice: {best_val:.4f}")

    ckpt = torch.load(CKPT_OUT, map_location=device, weights_only=False)
    channels = ckpt["channels"]
    model = SegmentationUNet(len(channels), int(HP["base_channels"]), float(HP["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_set = SegmentationDataset(DATA_DIR, test_ids, ckpt["norm_mean"], ckpt["norm_std"], channels)
    loader = DataLoader(test_set, batch_size=1)

    dices, ious, precs, aucs = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = torch.sigmoid(model(x))
            pred_bin = (probs > 0.5).float()
            dices.append(dice_coeff(y, y, y, mse=pred_bin).item())
            ious.append(IoU(y, pred_bin))
            precs.append(precision(y, pred_bin).item())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fpr, tpr, _ = ROC_AUC(y.to(torch.uint8), probs)
                aucs.append(AUC_score(fpr, tpr))

    print(f"\nTEST ({len(dices)} slices, {len(test_ids)} patients):")
    print(f"Dice:      {np.mean(dices):.4f} +- {np.std(dices):.4f}")
    print(f"IoU:       {np.mean(ious):.4f} +- {np.std(ious):.4f}")
    print(f"Precision: {np.mean(precs):.4f} +- {np.std(precs):.4f}")
    print(f"AUC:       {np.nanmean(aucs):.4f} +- {np.nanstd(aucs):.4f}")


if __name__ == "__main__":
    main()
