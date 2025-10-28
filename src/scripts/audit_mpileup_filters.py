#!/usr/bin/env python3
"""
Audit mpileup filtering outcomes per sample using the same logic as preprocess_mpileup.py.

For each mpileup (<sample>.txt), this script reproduces filtering decisions and reports counts per reason:
  - depth_filtered
  - low_alt_filtered (alt_count < 3)
  - n_excess_filtered (N > 10x alt)
  - bias_filtered (fails positional bias)
  - kept_mutation (passes bias and min-alt, with mutation type if C>T or G>A)
Additionally breaks down by reference base (C vs G vs other) and reports totals.

Outputs:
  - filters_summary.csv: per-sample counts by reason (and C/G breakdown where relevant)
  - filters_details.json: optional detailed tallies per sample

Usage:
  python audit_mpileup_filters.py --mpileup-dir DIR --out-dir OUT [--percentile 5]
"""

import argparse
import os
import json
from typing import Dict, Any, Tuple


def analyze_read_position_bias(reads: str, alt: str) -> Tuple[float, float]:
    # Lightweight replica: treat as passing bias if no obvious clustering
    # To match preprocess_mpileup.py closely, copy thresholds:
    # Passes if (pvalue == -1 and statistic == -1) or (statistic < 0.25 and pvalue > 0.01)
    # We don't recompute KS here; instead, approximate using spacing.
    alt_positions = []
    other_positions = []
    current_pos = 0
    i = 0
    while i < len(reads):
        c = reads[i]
        if c == '^':
            i += 2
            continue
        if c == '$':
            i += 1
            continue
        if c in '+-':
            i += 1
            num = ''
            while i < len(reads) and reads[i].isdigit():
                num += reads[i]
                i += 1
            i += int(num) if num.isdigit() else 0
            continue
        if c == '*':
            i += 1
            continue
        if c in '.,AGCTagct':
            if alt and c in (alt + alt.lower()):
                alt_positions.append(current_pos)
            else:
                other_positions.append(current_pos)
            current_pos += 1
        i += 1
    # Simple heuristic used in preprocess_mpileup for special cases
    if len(alt_positions) == 0:
        return 1.0, 0.0
    alt_prop = len(alt_positions) / max(1, (len(alt_positions) + len(other_positions)))
    if alt_prop > 0.9:
        return -1.0, -1.0
    if len(alt_positions) < 2:
        return 1.0, 0.0
    alt_positions.sort()
    # Spacing heuristic
    read_length = max(alt_positions[-1] + 1, current_pos)
    expected = read_length / (len(alt_positions) + 1)
    dists = [alt_positions[i+1] - alt_positions[i] for i in range(len(alt_positions)-1)]
    actual = sum(dists) / len(dists)
    if actual > 0.6 * expected:
        return 1.0, 0.0
    return 0.0, 1.0


def count_bases(reads: str, ref: str) -> Tuple[Dict[str, int], str, int]:
    counts = {b: 0 for b in 'ACGTN'}
    i = 0
    while i < len(reads):
        c = reads[i]
        if c == '^':
            i += 2
            continue
        if c == '$':
            i += 1
            continue
        if c in '+-':
            i += 1
            num = ''
            while i < len(reads) and reads[i].isdigit():
                num += reads[i]
                i += 1
            i += int(num) if num.isdigit() else 0
            continue
        if c == '*':
            i += 1
            continue
        base = c.upper()
        if base in counts:
            counts[base] += 1
        i += 1
    alt_total = sum(counts[b] for b in 'ACGT' if b != ref.upper())
    alt_bases = {b: counts[b] for b in 'ACGT' if b != ref.upper()}
    alt = max(alt_bases.items(), key=lambda x: x[1])[0] if alt_bases else None
    return counts, alt, alt_total


def get_depth_threshold(mpileup_path: str, percentile: float) -> int:
    import numpy as np
    depths = []
    with open(mpileup_path) as f:
        for idx, line in enumerate(f):
            if idx % 1000 == 0:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        depths.append(int(parts[3]))
                    except ValueError:
                        pass
    if depths:
        return max(1, int(np.percentile(depths, percentile)))
    return 1


def audit_file(path: str, percentile: float, pos_mask: dict = None) -> Dict[str, Any]:
    depth_thr = get_depth_threshold(path, percentile)
    tallies: Dict[str, int] = {
        'total_lines': 0,
        'depth_filtered': 0,
        'low_alt_filtered': 0,
        'n_excess_filtered': 0,
        'bias_filtered': 0,
        'kept_mutation': 0,
        'kept_ct': 0,
        'kept_ga': 0,
        'kept_other': 0,
        'callable_positions': 0,
    }
    cg_context = {'C_ref': 0, 'G_ref': 0}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            tallies['total_lines'] += 1
            # Optional positional mask (e.g., restrict to genic and non-overlap)
            if pos_mask is not None:
                try:
                    pos1 = int(parts[1])
                except ValueError:
                    continue
                pos0 = pos1 - 1
                if pos0 < 0 or pos0 >= pos_mask.get('length', 0):
                    continue
                if not pos_mask['mask'][pos0]:
                    continue
            try:
                depth = int(parts[3])
            except ValueError:
                continue
            ref = parts[2].upper()
            reads = parts[4]
            # depth filter
            if depth < depth_thr:
                tallies['depth_filtered'] += 1
                continue
            # count bases
            base_counts, alt_base, alt_count = count_bases(reads, ref)
            # Callable if passes coverage and (if alt present) passes N and bias
            callable_ok = False
            if alt_count == 0:
                callable_ok = True
            else:
                if base_counts.get('N', 0) <= alt_count * 10:
                    p, s = analyze_read_position_bias(reads, alt_base or '')
                    callable_ok = ((p == -1 and s == -1) or (s < 0.25 and p > 0.01))
            if callable_ok:
                tallies['callable_positions'] += 1
            # low-alt
            if alt_count < 3:
                tallies['low_alt_filtered'] += 1
                continue
            # N content
            if base_counts.get('N', 0) > alt_count * 10:
                tallies['n_excess_filtered'] += 1
                continue
            # bias
            pvalue, statistic = analyze_read_position_bias(reads, alt_base if alt_base else '')
            passes_bias = (pvalue == -1 and statistic == -1) or (statistic < 0.25 and pvalue > 0.01)
            if not passes_bias:
                tallies['bias_filtered'] += 1
                continue
            # kept
            tallies['kept_mutation'] += 1
            if ref == 'C':
                cg_context['C_ref'] += 1
            elif ref == 'G':
                cg_context['G_ref'] += 1
            mtype = None
            if ref == 'C' and alt_base == 'T':
                mtype = 'ct'
            elif ref == 'G' and alt_base == 'A':
                mtype = 'ga'
            if mtype == 'ct':
                tallies['kept_ct'] += 1
            elif mtype == 'ga':
                tallies['kept_ga'] += 1
            else:
                tallies['kept_other'] += 1
    tallies.update(cg_context)
    tallies['depth_threshold'] = depth_thr
    return tallies


def main() -> None:
    import pandas as pd
    # Optional genome restriction support
    try:
        from modules.parse import SeqContext
        import yaml
    except Exception:
        SeqContext = None  # type: ignore
        yaml = None  # type: ignore
    ap = argparse.ArgumentParser(description='Audit mpileup filter reasons per sample.')
    ap.add_argument('--mpileup-dir', required=True, help='Directory with *.txt mpileup files')
    ap.add_argument('--out-dir', required=True, help='Output directory')
    ap.add_argument('--percentile', type=float, default=5.0, help='Depth percentile for coverage threshold (default: 5)')
    ap.add_argument('--config', type=str, default=None, help='config.yaml to enable genic+overlap mask restriction (optional)')
    ap.add_argument('--restrict-genic', action='store_true', help='Restrict audit to genic, non-overlap positions (requires --config)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    details = {}
    pos_mask = None
    if args.restrict_genic and args.config and SeqContext is not None:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        seqctx = SeqContext(cfg['references']['genomic_fna'], cfg['references']['annotation'])
        # Build a genic mask combined with overlap_mask
        gene_mask = [False] * len(seqctx.genome)
        feats = seqctx.genome_features()
        for _, feat in feats.items():
            for i in range(feat.start, feat.end):
                if 0 <= i < len(gene_mask):
                    gene_mask[i] = True
        combined = [g and m for g, m in zip(gene_mask, seqctx.overlap_mask)]
        pos_mask = {'mask': combined, 'length': len(combined)}
    for fn in sorted(os.listdir(args.mpileup_dir)):
        if not fn.endswith('.txt'):
            continue
        sample = os.path.splitext(fn)[0]
        path = os.path.join(args.mpileup_dir, fn)
        tallies = audit_file(path, args.percentile, pos_mask=pos_mask)
        tallies['sample'] = sample
        rows.append(tallies)
        details[sample] = tallies

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, 'filters_summary.csv')
    df = df[['sample'] + [c for c in df.columns if c != 'sample']]
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(args.out_dir, 'filters_details.json')
    with open(json_path, 'w') as f:
        json.dump(details, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == '__main__':
    main()


