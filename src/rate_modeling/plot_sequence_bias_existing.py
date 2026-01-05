#!/usr/bin/env python3
"""
Create improved plots from an existing summary JSON written by
sequence_bias_modeling_sitelevel.py (no refitting).

Usage:
  python plot_sequence_bias_existing.py \
    --summary-json /path/to/sequence_bias_sitelevel_summary.json \
    --output-dir /path/to/out
"""

import os
import sys
import argparse

# Robust import: works both as a package module and as a direct script
try:
    from .sequence_bias_modeling_sitelevel import (
        plot_model_comparison_better,
        plot_positional_heatmap_better,
    )
except ImportError:
    # Fall back to adding this directory to sys.path and import the sibling module
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from sequence_bias_modeling_sitelevel import (
        plot_model_comparison_better,
        plot_positional_heatmap_better,
    )


def main():
    parser = argparse.ArgumentParser(description='Plot improved site-level sequence-bias figures from existing summary JSON')
    parser.add_argument('--summary-json', required=True, help='Path to sequence_bias_sitelevel_summary.json')
    parser.add_argument('--output-dir', required=True, help='Directory to write improved plots')
    args = parser.parse_args()

    import json
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.summary_json) as fh:
        summary = json.load(fh)

    results = summary.get('results', {})
    df_len = int(summary.get('df_len', 1))
    positional_params = summary.get('positional_params', {})

    print('Generating improved plots from summary...')
    plot_model_comparison_better(results, df_len, args.output_dir)
    if positional_params:
        plot_positional_heatmap_better(positional_params, args.output_dir)
    print('Done.')


if __name__ == '__main__':
    main()


