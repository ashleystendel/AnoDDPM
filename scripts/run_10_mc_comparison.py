"""
Runs detection.py once per MC count in --n-samples-unc-values, each with --agg-method all,
so mean/median/min_error/majority_vote are evaluated from the SAME set of MC passes per
slice (directly comparable, no extra run-to-run randomness on top of the actual difference
between aggregation formulas):

  python detection.py <ARG_NUM> --uncertainty --n-samples-unc=<K> --save-output --n-slices 1 \\
      --sample-distance 250 --agg-method all

Default ARG_NUM is 30 (NFBS+IXI, orientation/normalization-fix validated -- the checkpoint
used throughout the main ablation results). Override with --arg-num if you need a different one.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
            "--n-samples-unc-values", type=int, nargs="+", default=[4, 6, 8], dest="n_samples_unc_values",
            help="MC counts (K) to sweep, e.g. --n-samples-unc-values 4 6 8"
            )
    parser.add_argument("--arg-num", type=str, default="30", dest="arg_num")
    parser.add_argument(
            "--dataset-path", type=str, default="./DATASETS/BraTS", dest="dataset_path",
            help="Passed through as detection.py's positional dataset-path argument. "
                 "detection.py falls back to its own default (Edinburgh) if this is omitted, "
                 "so pass this explicitly to run on BraTS."
            )
    cli_args = parser.parse_args()

    for k in cli_args.n_samples_unc_values:
        cmd = [
            sys.executable, "detection.py", cli_args.arg_num, cli_args.dataset_path,
            "--uncertainty",
            f"--n-samples-unc={k}",
            "--save-output",
            "--n-slices", "1",
            "--sample-distance", "250",
            "--agg-method", "all",
            ]
        print(f"\n########## n_samples_unc={k} (all 4 agg methods) ##########")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()