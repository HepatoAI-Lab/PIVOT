#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute diagnostic metrics without manuscript-specific plotting.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="pivot_score")
    parser.add_argument("--cohort-col", default="split")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    from pivot.evaluation import diagnostic_summary

    summary = diagnostic_summary(
        prediction_csv=args.predictions_csv,
        label_col=args.label_col,
        score_col=args.score_col,
        cohort_col=args.cohort_col,
        threshold=args.threshold,
        n_bootstrap=args.bootstrap,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
