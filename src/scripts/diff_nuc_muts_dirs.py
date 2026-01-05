#!/usr/bin/env python3
"""
Compare two nuc_muts directories (both produced via parse.py) and pinpoint C>T vs G>A differences.

Outputs:
  - per_sample_diff.csv: per-sample deltas (dirB - dirA) for ct/ga sites and alt sums
  - per_gene_diff.csv: per-gene deltas for each sample present in both sets
  - if --config provided: adds gene strand to per_gene_diff to detect strand-skewed drops

Usage:
  python diff_nuc_muts_dirs.py --dir-a A/nuc_muts --dir-b B/nuc_muts --out-dir OUT [--config config.yaml]
"""

import argparse
import json
import os
from typing import Dict, Any, List, Tuple
import pandas as pd


def load_jsons(nuc_dir: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    for fn in os.listdir(nuc_dir):
        if not fn.endswith('.json'):
            continue
        if os.path.splitext(fn)[0] == 'basecounts':
            continue
        path = os.path.join(nuc_dir, fn)
        try:
            with open(path) as f:
                data[os.path.splitext(fn)[0]] = json.load(f)
        except Exception:
            pass
    return data


def summarize_sample(sample_data: Dict[str, Any]) -> Tuple[Dict[str, int], pd.DataFrame]:
    ct_sites = 0
    ga_sites = 0
    ct_alt = 0
    ga_alt = 0
    rows: List[Dict[str, Any]] = []
    for gene_id, gene_obj in sample_data.items():
        muts = gene_obj.get('mutations', {})
        if not isinstance(muts, dict):
            continue
        g_ct_sites = 0
        g_ga_sites = 0
        g_ct_alt = 0
        g_ga_alt = 0
        for key, val in muts.items():
            try:
                pos_str, mtype = key.split('_')
            except ValueError:
                continue
            if mtype not in ('C>T', 'G>A'):
                continue
            try:
                alt_obs = int(val)
            except Exception:
                alt_obs = 1
            if mtype == 'C>T':
                ct_sites += 1
                ct_alt += alt_obs
                g_ct_sites += 1
                g_ct_alt += alt_obs
            else:
                ga_sites += 1
                ga_alt += alt_obs
                g_ga_sites += 1
                g_ga_alt += alt_obs
        if g_ct_sites or g_ga_sites:
            rows.append({
                'gene_id': gene_id,
                'ct_sites': g_ct_sites,
                'ga_sites': g_ga_sites,
                'ct_alt_sum': g_ct_alt,
                'ga_alt_sum': g_ga_alt,
            })
    agg = {'ct_sites': ct_sites, 'ga_sites': ga_sites, 'ct_alt_sum': ct_alt, 'ga_alt_sum': ga_alt}
    return agg, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description='Diff two nuc_muts directories for C>T vs G>A differences')
    ap.add_argument('--dir-a', required=True, help='First nuc_muts dir (baseline)')
    ap.add_argument('--dir-b', required=True, help='Second nuc_muts dir (comparison)')
    ap.add_argument('--out-dir', required=True, help='Output directory')
    ap.add_argument('--config', default=None, help='config.yaml to annotate gene strand (optional)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data_a = load_jsons(args.dir_a)
    data_b = load_jsons(args.dir_b)

    common_samples = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not common_samples:
        print('No common samples found between dirs')
        return

    # Optional strand map
    strand_map: Dict[str, int] = {}
    if args.config:
        try:
            from modules.parse import SeqContext
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            seqctx = SeqContext(cfg['references']['genomic_fna'], cfg['references']['annotation'])
            feats = seqctx.genome_features()
            strand_map = {gid: feat.strand for gid, feat in feats.items()}
        except Exception:
            pass

    per_sample_rows: List[Dict[str, Any]] = []
    per_gene_rows: List[Dict[str, Any]] = []

    for sample in common_samples:
        agg_a, genes_a = summarize_sample(data_a[sample])
        agg_b, genes_b = summarize_sample(data_b[sample])
        per_sample_rows.append({
            'sample': sample,
            'ct_sites_A': agg_a['ct_sites'],
            'ct_sites_B': agg_b['ct_sites'],
            'ct_sites_delta': agg_b['ct_sites'] - agg_a['ct_sites'],
            'ga_sites_A': agg_a['ga_sites'],
            'ga_sites_B': agg_b['ga_sites'],
            'ga_sites_delta': agg_b['ga_sites'] - agg_a['ga_sites'],
            'ct_alt_A': agg_a['ct_alt_sum'],
            'ct_alt_B': agg_b['ct_alt_sum'],
            'ct_alt_delta': agg_b['ct_alt_sum'] - agg_a['ct_alt_sum'],
            'ga_alt_A': agg_a['ga_alt_sum'],
            'ga_alt_B': agg_b['ga_alt_sum'],
            'ga_alt_delta': agg_b['ga_alt_sum'] - agg_a['ga_alt_sum'],
        })
        # Per-gene join
        if not genes_a.empty or not genes_b.empty:
            g = pd.merge(genes_a, genes_b, on='gene_id', how='outer', suffixes=('_A', '_B')).fillna(0)
            g['sample'] = sample
            for col in ['ct_sites', 'ga_sites', 'ct_alt_sum', 'ga_alt_sum']:
                g[f'{col}_delta'] = g[f'{col}_B'] - g[f'{col}_A']
            if strand_map:
                g['strand'] = g['gene_id'].map(strand_map).fillna(0).astype(int)
            per_gene_rows.extend(g.to_dict(orient='records'))

    ps_df = pd.DataFrame(per_sample_rows)
    ps_csv = os.path.join(args.out_dir, 'per_sample_diff.csv')
    ps_df.to_csv(ps_csv, index=False)

    pg_df = pd.DataFrame(per_gene_rows)
    pg_csv = os.path.join(args.out_dir, 'per_gene_diff.csv')
    pg_df.to_csv(pg_csv, index=False)

    print(f'Wrote {ps_csv}')
    print(f'Wrote {pg_csv}')


if __name__ == '__main__':
    main()


