#!/usr/bin/env python3
"""
Collect supplemental mutation tables: per-sample and per-gene mutation counts and rates.

This script generates two tables:
1. Per-sample table with total, synonymous, and non-synonymous mutation counts and rates
2. Per-gene table with total, synonymous, and non-synonymous mutation counts and rates

Uses existing functions from estimate_rates.py to load and process data.
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to import from rate_modeling
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

from rate_modeling.estimate_rates import (
    load_site_level_with_category,
    create_gene_windows
)


def map_sites_to_genes(site_df: pd.DataFrame, gene_windows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map sites to genes based on position overlap using efficient interval search.
    
    Args:
        site_df: DataFrame with columns chrom, pos, and other site data
        gene_windows_df: DataFrame with columns chrom, start, end, gene_id, gene_name, strand
    
    Returns:
        DataFrame with gene_id, gene_name, strand added to site_df (NaN for intergenic sites)
    """
    print("Mapping sites to genes...")
    
    # Prepare gene windows
    genes = gene_windows_df[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand']].copy()
    
    # Process each chromosome separately
    result_parts = []
    
    for chrom in site_df['chrom'].unique():
        chrom_sites = site_df[site_df['chrom'] == chrom].copy()
        chrom_genes = genes[genes['chrom'] == chrom].copy()
        
        if len(chrom_genes) == 0:
            # No genes on this chromosome
            chrom_sites['gene_id'] = None
            chrom_sites['gene_name'] = None
            chrom_sites['strand'] = None
            result_parts.append(chrom_sites)
            continue
        
        # Sort genes by start position for efficient search
        chrom_genes = chrom_genes.sort_values('start').reset_index(drop=True)
        
        # Initialize gene columns
        chrom_sites['gene_id'] = None
        chrom_sites['gene_name'] = None
        chrom_sites['strand'] = None
        
        # Use numpy for fully vectorized search
        site_positions = chrom_sites['pos'].values[:, np.newaxis]  # Shape: (n_sites, 1)
        gene_starts = chrom_genes['start'].values[np.newaxis, :]  # Shape: (1, n_genes)
        gene_ends = chrom_genes['end'].values[np.newaxis, :]       # Shape: (1, n_genes)
        
        # Create boolean matrix: (n_sites, n_genes) - True if site is within gene bounds
        # Broadcasting: (n_sites, 1) >= (1, n_genes) -> (n_sites, n_genes)
        overlap_matrix = (site_positions >= gene_starts) & (site_positions <= gene_ends)
        
        # For each site, find first matching gene (argmax finds first True)
        # Use argmax along gene axis, but need to handle no-match case
        gene_indices = np.argmax(overlap_matrix, axis=1)
        has_match = overlap_matrix[np.arange(len(site_positions)), gene_indices]
        
        # Only assign gene info where there's a match
        match_mask = has_match
        if match_mask.any():
            chrom_sites.loc[match_mask, 'gene_id'] = chrom_genes.iloc[gene_indices[match_mask]]['gene_id'].values
            chrom_sites.loc[match_mask, 'gene_name'] = chrom_genes.iloc[gene_indices[match_mask]]['gene_name'].values
            chrom_sites.loc[match_mask, 'strand'] = chrom_genes.iloc[gene_indices[match_mask]]['strand'].values
        
        result_parts.append(chrom_sites)
    
    # Combine all chromosomes
    result_df = pd.concat(result_parts, ignore_index=True)
    
    n_mapped = result_df['gene_id'].notna().sum()
    print(f"Mapped {n_mapped:,} sites to genes (out of {len(result_df):,} total sites)")
    
    return result_df


def aggregate_per_sample(site_df: pd.DataFrame, per_sample_rates_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate mutation counts and rates per sample.
    
    Args:
        site_df: DataFrame with columns sample, category, ems_count, depth, is_control
        per_sample_rates_df: Optional DataFrame with per-sample rates from estimate_rates.py output
                            (columns: sample, site_glm_rate, site_glm_CI_low, site_glm_CI_high)
    
    Returns:
        DataFrame with per-sample aggregated counts and rates
    """
    print("Aggregating per-sample data...")
    
    # Group by sample and category
    grouped = site_df.groupby(['sample', 'category'], as_index=False).agg({
        'ems_count': 'sum',
        'depth': 'sum',
        'is_control': 'first'  # Should be same for all rows of a sample
    })
    
    # Pivot to get syn and non_syn as separate columns
    mutations_pivot = grouped.pivot_table(
        index='sample',
        columns='category',
        values='ems_count',
        fill_value=0
    )
    depth_pivot = grouped.pivot_table(
        index='sample',
        columns='category',
        values='depth',
        fill_value=0
    )
    
    # Get is_control (should be same for all categories of a sample)
    is_control_map = grouped.groupby('sample')['is_control'].first()
    
    # Combine into result DataFrame
    result = pd.DataFrame(index=mutations_pivot.index)
    result['sample'] = result.index
    
    # Total mutations and depth
    result['total_mutations'] = mutations_pivot.sum(axis=1)
    result['total_depth'] = depth_pivot.sum(axis=1)
    
    # Use rate from estimate_rates.py output if available, otherwise calculate simple rate
    if per_sample_rates_df is not None and not per_sample_rates_df.empty:
        # Merge rates from output file
        rates_map = per_sample_rates_df.set_index('sample')['site_glm_rate']
        result['mutation_rate'] = result['sample'].map(rates_map)
        # Fill missing with simple rate
        missing = result['mutation_rate'].isna()
        if missing.any():
            result.loc[missing, 'mutation_rate'] = (
                result.loc[missing, 'total_mutations'] / result.loc[missing, 'total_depth']
            )
    else:
        result['mutation_rate'] = result['total_mutations'] / result['total_depth']
    
    # Synonymous
    if 'synonymous' in mutations_pivot.columns:
        result['syn_mutations'] = mutations_pivot['synonymous']
        result['syn_depth'] = depth_pivot['synonymous']
        result['syn_rate'] = result['syn_mutations'] / result['syn_depth'].replace(0, np.nan)
    else:
        result['syn_mutations'] = 0
        result['syn_depth'] = 0
        result['syn_rate'] = np.nan
    
    # Non-synonymous
    if 'non_synonymous' in mutations_pivot.columns:
        result['non_syn_mutations'] = mutations_pivot['non_synonymous']
        result['non_syn_depth'] = depth_pivot['non_synonymous']
        result['non_syn_rate'] = result['non_syn_mutations'] / result['non_syn_depth'].replace(0, np.nan)
    else:
        result['non_syn_mutations'] = 0
        result['non_syn_depth'] = 0
        result['non_syn_rate'] = np.nan
    
    # Add is_control
    result['is_control'] = result['sample'].map(is_control_map)
    
    # Reset index and reorder columns
    result = result.reset_index(drop=True)
    result = result[[
        'sample', 'total_mutations', 'total_depth', 'mutation_rate',
        'syn_mutations', 'syn_depth', 'syn_rate',
        'non_syn_mutations', 'non_syn_depth', 'non_syn_rate',
        'is_control'
    ]]
    
    print(f"Aggregated data for {len(result)} samples")
    return result


def aggregate_per_gene(site_df_with_genes: pd.DataFrame, gene_windows_df: pd.DataFrame, 
                      gene_rates_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate mutation counts and rates per gene.
    
    Args:
        site_df_with_genes: DataFrame with columns gene_id, category, ems_count, depth, sample
        gene_windows_df: DataFrame with gene metadata (gene_id, gene_name, chrom, start, end, strand)
        gene_rates_df: Optional DataFrame with per-gene rates from estimate_rates.py output
                      (columns: gene_id, rate_treated, rate_control, total_mutations, total_depth)
    
    Returns:
        DataFrame with per-gene aggregated counts and rates
    """
    print("Aggregating per-gene data...")
    
    # Filter to only sites that are in genes (have gene_id)
    gene_sites = site_df_with_genes[site_df_with_genes['gene_id'].notna()].copy()
    
    if len(gene_sites) == 0:
        print("Warning: No sites mapped to genes")
        return pd.DataFrame()
    
    # Group by gene_id and category
    grouped = gene_sites.groupby(['gene_id', 'category'], as_index=False).agg({
        'ems_count': 'sum',
        'depth': 'sum',
        'sample': 'nunique'  # Count unique samples per gene-category combination
    })
    
    # Pivot to get syn and non_syn as separate columns
    mutations_pivot = grouped.pivot_table(
        index='gene_id',
        columns='category',
        values='ems_count',
        fill_value=0
    )
    depth_pivot = grouped.pivot_table(
        index='gene_id',
        columns='category',
        values='depth',
        fill_value=0
    )
    
    # Count number of unique samples per gene (across all categories)
    n_samples = gene_sites.groupby('gene_id')['sample'].nunique()
    
    # Combine into result DataFrame
    # Create result with gene_id as index first (for proper alignment)
    result = pd.DataFrame(index=mutations_pivot.index)
    result['gene_id'] = result.index
    
    # Total mutations and depth - assign as Series (pandas will align by index)
    result['total_mutations'] = mutations_pivot.sum(axis=1)
    result['total_depth'] = depth_pivot.sum(axis=1)
    
    # Use rate from estimate_rates.py output if available
    if gene_rates_df is not None and not gene_rates_df.empty:
        # Prefer calculating from total_mutations/total_depth (overall rate across all samples)
        # Otherwise use rate_treated, then rate_control as fallbacks
        if 'total_mutations' in gene_rates_df.columns and 'total_depth' in gene_rates_df.columns:
            gene_rates_df = gene_rates_df.copy()
            gene_rates_df['calc_rate'] = gene_rates_df['total_mutations'] / gene_rates_df['total_depth'].replace(0, np.nan)
            rates_map = gene_rates_df.set_index('gene_id')['calc_rate']
        elif 'rate_treated' in gene_rates_df.columns:
            rates_map = gene_rates_df.set_index('gene_id')['rate_treated']
        elif 'rate_control' in gene_rates_df.columns:
            rates_map = gene_rates_df.set_index('gene_id')['rate_control']
        else:
            rates_map = None
        
        if rates_map is not None:
            result['mutation_rate'] = result['gene_id'].map(rates_map)
            # Fill missing with simple rate
            missing = result['mutation_rate'].isna()
            if missing.any():
                result.loc[missing, 'mutation_rate'] = (
                    result.loc[missing, 'total_mutations'] / result.loc[missing, 'total_depth']
                )
        else:
            result['mutation_rate'] = result['total_mutations'] / result['total_depth']
    else:
        result['mutation_rate'] = result['total_mutations'] / result['total_depth']
    
    # Synonymous - assign while index is still gene_id
    if 'synonymous' in mutations_pivot.columns:
        result['syn_mutations'] = mutations_pivot['synonymous']
        result['syn_depth'] = depth_pivot['synonymous']
        result['syn_rate'] = result['syn_mutations'] / result['syn_depth'].replace(0, np.nan)
    else:
        result['syn_mutations'] = 0
        result['syn_depth'] = 0
        result['syn_rate'] = np.nan
    
    # Non-synonymous - assign while index is still gene_id
    if 'non_synonymous' in mutations_pivot.columns:
        result['non_syn_mutations'] = mutations_pivot['non_synonymous']
        result['non_syn_depth'] = depth_pivot['non_synonymous']
        result['non_syn_rate'] = result['non_syn_mutations'] / result['non_syn_depth'].replace(0, np.nan)
    else:
        result['non_syn_mutations'] = 0
        result['non_syn_depth'] = 0
        result['non_syn_rate'] = np.nan
    
    # Add number of samples (map by index which is gene_id)
    result['n_samples'] = result.index.map(n_samples).fillna(0).astype(int)
    
    # Merge with gene metadata - convert to regular merge to preserve all columns
    gene_metadata = gene_windows_df[['gene_id', 'gene_name', 'chrom', 'start', 'end', 'strand']].copy()
    
    # Reset index to make gene_id a regular column before merging
    result = result.reset_index(drop=True)
    
    # Merge metadata (left join - keep all genes with mutation data)
    result = result.merge(gene_metadata, on='gene_id', how='left')
    
    # Reorder columns - ensure all data columns are preserved
    expected_cols = [
        'gene_id', 'gene_name', 'chrom', 'start', 'end', 'strand',
        'total_mutations', 'total_depth', 'mutation_rate',
        'syn_mutations', 'syn_depth', 'syn_rate',
        'non_syn_mutations', 'non_syn_depth', 'non_syn_rate',
        'n_samples'
    ]
    # Only include columns that exist, preserve order
    result_cols = [col for col in expected_cols if col in result.columns]
    # Add any remaining columns at the end
    remaining_cols = [col for col in result.columns if col not in result_cols]
    result = result[result_cols + remaining_cols]
    
    # Sort by gene_id
    result = result.sort_values('gene_id').reset_index(drop=True)
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['total_mutations', 'total_depth', 'mutation_rate',
                    'syn_mutations', 'syn_depth', 'syn_rate',
                    'non_syn_mutations', 'non_syn_depth', 'non_syn_rate']
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')
    
    print(f"Aggregated data for {len(result)} genes")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Collect supplemental mutation tables: per-sample and per-gene counts and rates"
    )
    parser.add_argument(
        "--counts-dir",
        required=True,
        help="Directory containing .counts files"
    )
    parser.add_argument(
        "--gff-file",
        required=True,
        help="GFF file for gene/CDS annotation"
    )
    parser.add_argument(
        "--genome-fasta",
        required=True,
        help="Genome FASTA file for codon context"
    )
    parser.add_argument(
        "--exclusion-mask",
        type=str,
        default=None,
        help="Optional exclusion mask file"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for tables"
    )
    parser.add_argument(
        "--codon-table",
        type=str,
        default=None,
        help="Optional codon table JSON file"
    )
    parser.add_argument(
        "--estimate-rates-output-dir",
        type=str,
        default=None,
        help="Directory containing estimate_rates.py output files (site_level_glm_per_sample.tsv, gene_mutation_rates.tsv)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load site-level data with category annotations
    print("=" * 60)
    print("Step 1: Loading site-level data with categories...")
    print("=" * 60)
    site_df = load_site_level_with_category(
        counts_dir=args.counts_dir,
        gff_file=args.gff_file,
        genome_fasta=args.genome_fasta,
        exclusion_mask=args.exclusion_mask,
        codon_table_path=args.codon_table
    )
    
    if site_df.empty:
        print("Error: No site-level data loaded")
        return 1
    
    # Step 2: Load gene windows
    print("\n" + "=" * 60)
    print("Step 2: Loading gene windows...")
    print("=" * 60)
    gene_windows_df = create_gene_windows(
        gff_file=args.gff_file,
        genome_fasta=args.genome_fasta,
        feature_type="gene"
    )
    
    if gene_windows_df.empty:
        print("Warning: No gene windows found")
    
    # Step 3: Load rates from estimate_rates.py output if available
    per_sample_rates_df = None
    gene_rates_df = None
    
    if args.estimate_rates_output_dir:
        print("\n" + "=" * 60)
        print("Step 3: Loading rates from estimate_rates.py output...")
        print("=" * 60)
        
        # Load per-sample rates
        per_sample_rates_path = os.path.join(args.estimate_rates_output_dir, "site_level_glm_per_sample.tsv")
        if os.path.exists(per_sample_rates_path):
            per_sample_rates_df = pd.read_csv(per_sample_rates_path, sep="\t")
            print(f"Loaded per-sample rates from: {per_sample_rates_path}")
            print(f"  Found rates for {len(per_sample_rates_df)} samples")
        else:
            print(f"Warning: Per-sample rates file not found: {per_sample_rates_path}")
        
        # Load per-gene rates (check both possible locations)
        gene_rates_paths = [
            os.path.join(args.estimate_rates_output_dir, "5mer_normalized_gene_windows", "gene_mutation_rates.tsv"),
            os.path.join(args.estimate_rates_output_dir, "gene_mutation_rates.tsv")
        ]
        
        for gene_rates_path in gene_rates_paths:
            if os.path.exists(gene_rates_path):
                gene_rates_df = pd.read_csv(gene_rates_path, sep="\t")
                print(f"Loaded per-gene rates from: {gene_rates_path}")
                print(f"  Found rates for {len(gene_rates_df)} genes")
                break
        
        if gene_rates_df is None:
            print("Warning: Per-gene rates file not found in expected locations")
    
    # Step 4: Map sites to genes
    print("\n" + "=" * 60)
    print("Step 4: Mapping sites to genes...")
    print("=" * 60)
    site_df_with_genes = map_sites_to_genes(site_df, gene_windows_df)
    
    # Step 5: Aggregate per-sample
    print("\n" + "=" * 60)
    print("Step 5: Aggregating per-sample data...")
    print("=" * 60)
    per_sample_df = aggregate_per_sample(site_df, per_sample_rates_df)
    
    # Step 6: Aggregate per-gene
    print("\n" + "=" * 60)
    print("Step 6: Aggregating per-gene data...")
    print("=" * 60)
    per_gene_df = aggregate_per_gene(site_df_with_genes, gene_windows_df, gene_rates_df)
    
    # Step 7: Write output files
    print("\n" + "=" * 60)
    print("Step 7: Writing output files...")
    print("=" * 60)
    
    per_sample_path = os.path.join(args.output_dir, "per_sample_mutation_table.tsv")
    per_sample_df.to_csv(per_sample_path, sep="\t", index=False)
    print(f"Per-sample table saved to: {per_sample_path}")
    print(f"  Rows: {len(per_sample_df)}")
    
    if not per_gene_df.empty:
        per_gene_path = os.path.join(args.output_dir, "per_gene_mutation_table.tsv")
        per_gene_df.to_csv(per_gene_path, sep="\t", index=False)
        print(f"Per-gene table saved to: {per_gene_path}")
        print(f"  Rows: {len(per_gene_df)}")
    else:
        print("Warning: Per-gene table is empty, not writing file")
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

