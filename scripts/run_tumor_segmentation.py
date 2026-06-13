#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated tumor segmentation for PIVOT MRI preparation.")
    parser.add_argument("--cases-csv", required=True, help="CSV with patient_id and image_path columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-dir", default="../LiverTumorSegmentor")
    parser.add_argument("--dataset-id", default="666")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    from pivot.preprocessing import run_nnunet_segmentation

    cases = pd.read_csv(args.cases_csv)
    missing = {"patient_id", "image_path"}.difference(cases.columns)
    if missing:
        raise ValueError(f"cases CSV is missing columns: {sorted(missing)}")

    records = []
    for row in cases.itertuples(index=False):
        output_path = run_nnunet_segmentation(
            image_path=row.image_path,
            output_dir=Path(args.output_dir) / str(row.patient_id),
            model_dir=args.model_dir,
            dataset_id=args.dataset_id,
            configuration=args.configuration,
            fold=args.fold,
            device=args.device,
            disable_tta=not args.tta,
        )
        records.append({"patient_id": row.patient_id, "mask_path": str(output_path)})

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
