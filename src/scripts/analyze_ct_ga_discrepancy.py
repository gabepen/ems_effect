#!/usr/bin/env python3
"""
Diagnostic script to investigate discrepancies between C>T and G>A mutation rates.

Inputs:
  - Directory of nuc_muts JSONs produced by preprocess_mpileup.py -> provean_effectscore.py
  - Optional mpileup directory (to find *_filtered_positions.json for callable sites)
  - Optional config.yaml (to load genome FASTA/GFF for base lookup when computing callable denominators)

Outputs (written to output directory):
  - per_sample_summary.csv: per-sample aggregates for C>T and G>A
  - per_sample_top_genes.csv: top contributing genes per sample (by sites and alt sums)
  - per_sample_distributions.json: basic distributions (alt counts per site) per sample and type
  - If config and mpileup dir are provided: adds callable denominators (callable_C_sites, callable_G_sites)

Per-sample metrics include:
  - ct_sites, ga_sites: number of unique mutated sites (count of keys) per type
  - ct_alt_sum, ga_alt_sum: sum of observed alt allele counts across sites per type
  - ct_mean_alt_per_site, ga_mean_alt_per_site
  - ct_to_ga_site_ratio, ct_to_ga_alt_ratio
  - callable_C_sites, callable_G_sites (optional)

Usage:
  python analyze_ct_ga_discrepancy.py \
    --nuc-muts-dir /path/to/nuc_muts \
    --out-dir /path/to/out \
    [--mpileup-dir /path/to/mpileups] \
    [--config /path/to/config.yaml]
"""

import argparse
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

import pandas as pd


def list_nuc_mut_jsons(nuc_dir: str) -> List[str]:
    jsons: List[str] = []
    for fn in os.listdir(nuc_dir):
        if not fn.lower().endswith('.json'):
            continue
        stem = os.path.splitext(fn)[0]
        if stem == 'basecounts':
            continue
        jsons.append(os.path.join(nuc_dir, fn))
    return sorted(jsons)


def parse_mutation_key(mut_key: str) -> Tuple[int, str]:
    # mut_key format like "123_C>T" or "456_G>A"; position is 0-based gene-relative in many pipelines
    try:
        pos_str, mtype = mut_key.split('_')
        return int(pos_str), mtype
    except Exception:
        return -1, ''


def analyze_sample_json(json_path: str) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, List[int]]]:
    """Return per-sample aggregates, per-gene table, and distributions per type."""
    with open(json_path, 'r') as f:
        sample_data = json.load(f)

    # Aggregates
    site_counts_by_type: Counter = Counter()
    alt_sums_by_type: Counter = Counter()
    alt_counts_per_site: Dict[str, List[int]] = {'C>T': [], 'G>A': []}

    # Per-gene aggregates for top genes table
    per_gene_rows: List[Dict[str, Any]] = []

    for gene_id, gene_obj in sample_data.items():
        mutations: Dict[str, int] = gene_obj.get('mutations', {})
        if not isinstance(mutations, dict):
            continue
        # Per-gene accumulators
        gene_ct_sites = 0
        gene_ga_sites = 0
        gene_ct_alt_sum = 0
        gene_ga_alt_sum = 0

        for mut_key, val in mutations.items():
            _, mtype = parse_mutation_key(mut_key)
            if mtype not in ('C>T', 'G>A'):
                continue
            # Treat dict value as an alt observation count if numeric, else fallback to 1
            try:
                alt_obs = int(val)
            except Exception:
                alt_obs = 1

            site_counts_by_type[mtype] += 1
            alt_sums_by_type[mtype] += alt_obs
            alt_counts_per_site.setdefault(mtype, []).append(alt_obs)

            if mtype == 'C>T':
                gene_ct_sites += 1
                gene_ct_alt_sum += alt_obs
            else:  # G>A
                gene_ga_sites += 1
                gene_ga_alt_sum += alt_obs

        if gene_ct_sites or gene_ga_sites:
            per_gene_rows.append({
                'gene_id': gene_id,
                'ct_sites': gene_ct_sites,
                'ga_sites': gene_ga_sites,
                'ct_alt_sum': gene_ct_alt_sum,
                'ga_alt_sum': gene_ga_alt_sum,
            })

    per_gene_df = pd.DataFrame(per_gene_rows)

    agg = {
        'ct_sites': int(site_counts_by_type.get('C>T', 0)),
        'ga_sites': int(site_counts_by_type.get('G>A', 0)),
        'ct_alt_sum': int(alt_sums_by_type.get('C>T', 0)),
        'ga_alt_sum': int(alt_sums_by_type.get('G>A', 0)),
    }
    agg['ct_mean_alt_per_site'] = (agg['ct_alt_sum'] / agg['ct_sites']) if agg['ct_sites'] > 0 else 0.0
    agg['ga_mean_alt_per_site'] = (agg['ga_alt_sum'] / agg['ga_sites']) if agg['ga_sites'] > 0 else 0.0
    agg['ct_to_ga_site_ratio'] = (agg['ct_sites'] / agg['ga_sites']) if agg['ga_sites'] > 0 else float('inf')
    agg['ct_to_ga_alt_ratio'] = (agg['ct_alt_sum'] / agg['ga_alt_sum']) if agg['ga_alt_sum'] > 0 else float('inf')

    return agg, per_gene_df, alt_counts_per_site


def load_callable_positions(sample_name: str, mpileup_dir: str) -> Dict[str, List[int]]:
    """Load <sample>_filtered_positions.json produced by preprocess_mpileup.py if present."""
    candidates = [
        os.path.join(mpileup_dir, f"{sample_name}_filtered_positions.json"),
        os.path.join(mpileup_dir, f"{sample_name}_positions.json"),
    ]
    # If name already ends with _filtered, try alternate bases
    if sample_name.endswith('_filtered'):
        base = sample_name[:-9]
        candidates.insert(0, os.path.join(mpileup_dir, f"{sample_name}_positions.json"))
        candidates.append(os.path.join(mpileup_dir, f"{base}_filtered_positions.json"))
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return {}


def compute_callable_denominators(sample_name: str, mpileup_dir: str, config_path: str) -> Tuple[int, int]:
    """Return (callable_C_sites, callable_G_sites) using genome bases at callable positions.
    Requires config.yaml to load genome and annotation via SeqContext.
    """
    try:
        from modules.parse import SeqContext
        import yaml
    except Exception:
        return 0, 0

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    genomic_fna = cfg['references']['genomic_fna']
    annotation = cfg['references']['annotation']
    seqctx = SeqContext(genomic_fna, annotation)
    genome_seq = str(seqctx.genome).upper()
    features = seqctx.genome_features()

    callable_map = load_callable_positions(sample_name, mpileup_dir)
    if not callable_map:
        return 0, 0

    c_count = 0
    g_count = 0
    for gene_id, rel_positions in callable_map.items():
        feat = features.get(gene_id)
        if feat is None:
            continue
        gene_start = feat.start
        gene_end = feat.end
        strand = feat.strand
        for rel0 in rel_positions:
            if strand == 1:
                genome_pos = gene_start + int(rel0)
            else:
                genome_pos = gene_end - 1 - int(rel0)
            if genome_pos < 0 or genome_pos >= len(genome_seq):
                continue
            base = genome_seq[genome_pos]
            if base == 'C':
                c_count += 1
            elif base == 'G':
                g_count += 1
    return c_count, g_count


def main() -> None:
    ap = argparse.ArgumentParser(description='Diagnose C>T vs G>A discrepancies from nuc_muts JSONs.')
    ap.add_argument('--nuc-muts-dir', required=True, help='Directory containing nuc_muts JSONs')
    ap.add_argument('--out-dir', required=True, help='Output directory for reports')
    ap.add_argument('--mpileup-dir', default=None, help='Directory containing *_filtered_positions.json (optional)')
    ap.add_argument('--config', default=None, help='config.yaml with references (optional, enables callable denominators)')
    ap.add_argument('--top-n', type=int, default=15, help='Number of top genes to emit per sample (default: 15)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    json_paths = list_nuc_mut_jsons(args.nuc_muts_dir)
    if not json_paths:
        print(f"No nuc_muts JSONs found in {args.nuc_muts_dir}")
        return

    per_sample_rows: List[Dict[str, Any]] = []
    top_gene_rows: List[Dict[str, Any]] = []
    distributions: Dict[str, Dict[str, List[int]]] = {}

    for path in json_paths:
        sample = os.path.splitext(os.path.basename(path))[0]
        agg, per_gene_df, alt_counts_per_site = analyze_sample_json(path)
        agg['sample'] = sample

        # Optional callable denominators
        if args.mpileup_dir and args.config:
            try:
                c_sites, g_sites = compute_callable_denominators(sample, args.mpileup_dir, args.config)
                agg['callable_C_sites'] = int(c_sites)
                agg['callable_G_sites'] = int(g_sites)
                # Provide simple per-ref normalized rates for quick comparison
                agg['ct_sites_per_callable_C'] = (agg['ct_sites'] / c_sites) if c_sites > 0 else 0.0
                agg['ga_sites_per_callable_G'] = (agg['ga_sites'] / g_sites) if g_sites > 0 else 0.0
                agg['ct_alt_per_callable_C'] = (agg['ct_alt_sum'] / c_sites) if c_sites > 0 else 0.0
                agg['ga_alt_per_callable_G'] = (agg['ga_alt_sum'] / g_sites) if g_sites > 0 else 0.0
            except Exception:
                pass

        per_sample_rows.append(agg)

        # Top genes by sites and by alt sum
        if not per_gene_df.empty:
            # Add sample column
            per_gene_df['sample'] = sample
            # Top by site counts
            per_gene_df['site_total'] = per_gene_df['ct_sites'] + per_gene_df['ga_sites']
            top_sites = per_gene_df.sort_values('site_total', ascending=False).head(args.top_n)
            top_genes_sites = top_sites[['sample', 'gene_id', 'ct_sites', 'ga_sites', 'ct_alt_sum', 'ga_alt_sum']]
            top_gene_rows.extend(top_genes_sites.to_dict(orient='records'))
            # Top by alt sum
            per_gene_df['alt_total'] = per_gene_df['ct_alt_sum'] + per_gene_df['ga_alt_sum']
            top_alts = per_gene_df.sort_values('alt_total', ascending=False).head(args.top_n)
            top_genes_alts = top_alts[['sample', 'gene_id', 'ct_sites', 'ga_sites', 'ct_alt_sum', 'ga_alt_sum']]
            top_gene_rows.extend(top_genes_alts.to_dict(orient='records'))

        distributions[sample] = {
            'C>T': alt_counts_per_site.get('C>T', []),
            'G>A': alt_counts_per_site.get('G>A', []),
        }

    # Write outputs
    summary_df = pd.DataFrame(per_sample_rows)
    summary_csv = os.path.join(args.out_dir, 'per_sample_summary.csv')
    summary_df = summary_df[['sample'] + [c for c in summary_df.columns if c != 'sample']]
    summary_df.to_csv(summary_csv, index=False)

    top_genes_df = pd.DataFrame(top_gene_rows)
    top_genes_csv = os.path.join(args.out_dir, 'per_sample_top_genes.csv')
    top_genes_df.to_csv(top_genes_csv, index=False)

    dist_json = os.path.join(args.out_dir, 'per_sample_distributions.json')
    with open(dist_json, 'w') as f:
        json.dump(distributions, f, indent=2)

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {top_genes_csv}")
    print(f"Wrote: {dist_json}")


if __name__ == '__main__':
    main()


