from __future__ import annotations

import numpy as np
import pandas as pd

from pivot.training.metrics import binary_auc


def threshold_metrics(labels, scores, threshold: float = 0.5) -> dict[str, float]:
    """Compute diagnostic metrics at a prespecified score threshold."""

    y = np.asarray(labels).astype(int)
    s = np.asarray(scores).astype(float)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "threshold": float(threshold),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "accuracy": (tp + tn) / max(1, len(y)),
        "ppv": tp / max(1, tp + fp),
        "npv": tn / max(1, tn + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def bootstrap_auc_ci(labels, scores, n_bootstrap: int = 2000, seed: int = 2026) -> tuple[float, float, float]:
    """Compute AUROC and percentile bootstrap confidence interval."""

    rng = np.random.default_rng(seed)
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores).astype(float)
    auc = binary_auc(y.tolist(), s.tolist())
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(binary_auc(y[idx].tolist(), s[idx].tolist()))
    if not values:
        return float(auc), float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def diagnostic_summary(
    prediction_csv: str,
    label_col: str = "label",
    score_col: str = "pivot_score",
    cohort_col: str = "split",
    threshold: float = 0.5,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Summarize AUROC and threshold metrics by cohort."""

    df = pd.read_csv(prediction_csv)
    rows = []
    cohorts = ["all"] if cohort_col not in df.columns else sorted(df[cohort_col].dropna().unique().tolist())
    for cohort in cohorts:
        sub = df if cohort == "all" else df[df[cohort_col] == cohort]
        auc, ci_low, ci_high = bootstrap_auc_ci(
            sub[label_col].values,
            sub[score_col].values,
            n_bootstrap=n_bootstrap,
        )
        row = {
            "cohort": cohort,
            "n": int(len(sub)),
            "events": int(sub[label_col].sum()),
            "auc": auc,
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
        }
        row.update(threshold_metrics(sub[label_col].values, sub[score_col].values, threshold=threshold))
        rows.append(row)
    return pd.DataFrame(rows)
