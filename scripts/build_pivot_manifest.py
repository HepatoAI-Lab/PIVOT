#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the patient-level PIVOT training manifest.")
    parser.add_argument("--mri-csv", required=True, help="CSV produced by scripts/preprocess_mri.py.")
    parser.add_argument("--labels-csv", required=True, help="CSV with patient_id, split, and label.")
    parser.add_argument("--slide-embedding-csv", default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sequences", nargs="+", default=["T1WI", "T2WI", "DWI", "ADC", "AP", "PVP", "DP"])
    args = parser.parse_args()

    from pivot.preprocessing import build_patient_manifest

    build_patient_manifest(
        mri_csv=args.mri_csv,
        labels_csv=args.labels_csv,
        slide_embedding_csv=args.slide_embedding_csv,
        output_csv=args.output_csv,
        sequences=tuple(args.sequences),
    )


if __name__ == "__main__":
    main()
