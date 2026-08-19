"""
Runs detection.py once per MC aggregation method (mean/median/min_error/majority_vote)
so all four are directly comparable, each with:

  python detection.py 28 --uncertainty --n-samples-unc=<N> --save-output --n-slices 1 \\
      --sample-distance 250 --agg-method <method>
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
AGGREGATION_METHODS = ("mean", "median", "min_error", "majority_vote")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples-unc", type=int, default=4, dest="n_samples_unc")
    cli_args = parser.parse_args()

    for method in AGGREGATION_METHODS:
        cmd = [
            sys.executable, "detection.py", "28",
            "--uncertainty",
            f"--n-samples-unc={cli_args.n_samples_unc}",
            "--save-output",
            "--n-slices", "1",
            "--sample-distance", "250",
            "--agg-method", method,
            ]
        print(f"\n########## agg_method={method} ##########")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()