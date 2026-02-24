#!/usr/bin/env python3
"""
Combine model evaluation outputs into a single table.

Inputs (typical layout under a single --output-dir):
  - prediction metrics:  <output-dir>/prediction_evaluation/prediction_accuracy_metrics.tsv
  - cross-val summary:   <output-dir>/cross_validation/cv_results_summary.json

This script merges:
  - per-model prediction metrics (TSV written by prediction_accuracy_metrics.py)
  - per-model CV summary metrics (JSON written by cross_validation_evaluation.py)

and writes a combined CSV/TSV with one row per model.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import pandas as pd


def _read_cv_summary_json(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    rows = []
    for model_name, metrics in (data or {}).items():
        if metrics is None:
            continue
        row = {"model_name": model_name}
        # Prefix CV metrics to avoid collisions with prediction metrics
        for k, v in metrics.items():
            row[f"cv_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _infer_default_paths(output_dir: str) -> tuple[str, str]:
    pred_path = os.path.join(output_dir, "prediction_evaluation", "prediction_accuracy_metrics.tsv")
    cv_path = os.path.join(output_dir, "cross_validation", "cv_results_summary.json")
    return pred_path, cv_path


def main():
    ap = argparse.ArgumentParser(description="Combine prediction metrics TSV + CV summary JSON into one CSV/TSV.")
    ap.add_argument("--output-dir", required=True, help="Base output directory from the modeling pipeline.")
    ap.add_argument(
        "--prediction-tsv",
        default=None,
        help="Optional override path to prediction_accuracy_metrics.tsv",
    )
    ap.add_argument(
        "--cv-json",
        default=None,
        help="Optional override path to cv_results_summary.json",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output file path. Default: <output-dir>/combined_model_evaluation_metrics.tsv",
    )
    ap.add_argument(
        "--format",
        choices=["tsv", "csv"],
        default="tsv",
        help="Output format (default: tsv).",
    )
    args = ap.parse_args()

    output_dir = args.output_dir
    pred_default, cv_default = _infer_default_paths(output_dir)
    pred_path = args.prediction_tsv or pred_default
    cv_path = args.cv_json or cv_default

    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(output_dir, f"combined_model_evaluation_metrics.{args.format}")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction metrics TSV not found: {pred_path}")
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"CV summary JSON not found: {cv_path}")

    pred_df = pd.read_csv(pred_path, sep="\t")
    if "model_name" not in pred_df.columns:
        raise ValueError(f"'model_name' column not found in prediction metrics TSV: {pred_path}")

    cv_df = _read_cv_summary_json(cv_path)
    if cv_df.empty:
        # Still write a copy of prediction metrics, but warn via stderr-like print
        print(f"Warning: CV summary JSON contained no model entries: {cv_path}")
        merged = pred_df.copy()
    else:
        merged = pred_df.merge(cv_df, on="model_name", how="outer", validate="one_to_one")

    # Friendly column ordering: model_name first, then prediction metrics, then cv_*
    cols = list(merged.columns)
    pred_cols = [c for c in cols if c != "model_name" and not c.startswith("cv_")]
    cv_cols = [c for c in cols if c.startswith("cv_")]
    merged = merged[["model_name"] + sorted(pred_cols) + sorted(cv_cols)]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if args.format == "csv":
        merged.to_csv(out_path, index=False)
    else:
        merged.to_csv(out_path, sep="\t", index=False)

    print(f"Wrote combined metrics for {len(merged)} model(s) to: {out_path}")


if __name__ == "__main__":
    main()






