#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare registered tumor-centered MRI tensors for PIVOT.")
    parser.add_argument("--cases-csv", required=True, help="CSV with patient_id, mask_path, and MRI sequence columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--target-spacing", nargs=3, type=float, default=(1.5, 1.5, 3.0))
    parser.add_argument("--crop-size", nargs=3, type=int, default=(96, 160, 160))
    parser.add_argument("--reference-sequence", default="PVP")
    parser.add_argument("--sequences", nargs="+", default=["T1WI", "T2WI", "DWI", "ADC", "AP", "PVP", "DP"])
    args = parser.parse_args()

    import pandas as pd

    from pivot.preprocessing import MRIPreprocessConfig, preprocess_mri_case

    cases = pd.read_csv(args.cases_csv)
    required = {"patient_id", "mask_path", *args.sequences}
    missing = required.difference(cases.columns)
    if missing:
        raise ValueError(f"cases CSV is missing columns: {sorted(missing)}")

    config = MRIPreprocessConfig(
        target_spacing=tuple(args.target_spacing),
        crop_size=tuple(args.crop_size),
        reference_sequence=args.reference_sequence,
    )
    records = []
    for row in cases.itertuples(index=False):
        row_dict = row._asdict()
        patient_id = str(row_dict["patient_id"])
        sequence_paths = {
            seq: row_dict[seq]
            for seq in args.sequences
            if pd.notna(row_dict.get(seq, None)) and str(row_dict[seq]).strip()
        }
        outputs = preprocess_mri_case(
            patient_id=patient_id,
            sequence_paths=sequence_paths,
            mask_path=row_dict["mask_path"],
            output_dir=args.output_dir,
            config=config,
        )
        records.append(outputs)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
