"""
Short Optuna hyperparameter search for the segmentation U-Net.

Optimises validation Dice over lr, base_channels, batch_size, dropout,
weight_decay and BCE pos_weight. The patient-level train/val/test split is fixed
by --seed; the TEST set is never seen during the search. After the search the
best config is retrained (longer) and evaluated once on the held-out test set.

Usage:
    python3 segmentation_optuna.py --trials 30 --epochs 60
"""
import argparse

import optuna
import torch
from torch.utils.data import DataLoader

from segmentation_unet import (
    CHECKPOINT_DIR, DATA_DIR, SegmentationDataset, SegmentationUNet,
    evaluate, split_patients, train_model,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=60, help="epochs per trial")
    p.add_argument("--final_epochs", type=int, default=200, help="epochs to retrain the best config")
    p.add_argument("--n_val", type=int, default=4)
    p.add_argument("--n_test", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, val_ids, test_ids = split_patients(args.data_dir, args.n_val, args.n_test, args.seed)
    print(f"patients -> train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")

    def objective(trial):
        hp = dict(
                lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                base_channels=trial.suggest_categorical("base_channels", [16, 32, 48]),
                batch_size=trial.suggest_categorical("batch_size", [2, 4, 8]),
                dropout=trial.suggest_float("dropout", 0.0, 0.3),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                pos_weight=trial.suggest_float("pos_weight", 1.0, 20.0),
                )
        return train_model(hp, train_ids, val_ids, args.data_dir, args.epochs,
                           device, checkpoint_path=None, seed=args.seed, verbose=False)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("\nBest val dice:", round(study.best_value, 4))
    print("Best params:", study.best_params)

    # retrain the best config (longer) and evaluate once on the held-out test set
    hp = dict(study.best_params)
    ckpt_path = f"{CHECKPOINT_DIR}/best.pt"
    extra = dict(n_val=args.n_val, n_test=args.n_test, split_seed=args.seed, data_dir=args.data_dir)
    best_val = train_model(hp, train_ids, val_ids, args.data_dir, args.final_epochs,
                           device, ckpt_path, extra=extra, seed=args.seed)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SegmentationUNet(3, hp["base_channels"], hp["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_set = SegmentationDataset(args.data_dir, test_ids, ckpt["norm_mean"], ckpt["norm_std"])
    test_metrics = evaluate(model, DataLoader(test_set, batch_size=int(hp["batch_size"])), device)
    print(f"\nRetrained best val dice: {best_val:.4f}")
    print(f"TEST: dice={test_metrics['dice']:.4f} iou={test_metrics['iou']:.4f}")
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
