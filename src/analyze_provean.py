import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Any
import argparse
import requests
import yaml
from upsetplot import plot as upsetplot
from upsetplot import from_memberships
import operator
from statsmodels.stats.multitest import multipletests
import pdb
import matplotlib.gridspec as gridspec
from scipy.stats import percentileofscore
import ast
from modules import parse
from matplotlib.ticker import ScalarFormatter


def load_gene_info_cache(cache_file: str) -> Dict[str, Dict]:
    """Load cached gene info from JSON file."""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading gene info cache: {e}")
    return {}

def save_gene_info_cache(cache: Dict[str, Dict], cache_file: str) -> None:
    """Save gene info cache to JSON file."""
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Error saving gene info cache: {e}")

def lookup_gene_info(ncbi_id: str, cache_file: str = "gene_info_cache.json") -> Dict[str, Any]:
    """Look up gene information from NCBI's E-utils API with caching."""
    
    # Load cache
    cache = load_gene_info_cache(cache_file)
    
    #print(f"Looking up gene {ncbi_id} in cache")
    
    # Check cache first
    if str(ncbi_id) in cache:
        #print(f"Found gene {ncbi_id} in cache")
        return cache[str(ncbi_id)]
    
    print(f"Looking up gene {ncbi_id} from NCBI")
    # If not in cache, look up from NCBI
    server = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ext = f"/esummary.fcgi?db=gene&id={ncbi_id}&retmode=json"
    
    try:
        response = requests.get(server + ext)
        print(f"Response: {response}")
        print(f"Response.ok: {response.ok}")
        print(f"URL: {server + ext}")
        
        if response.ok:
            data = response.json()
            if 'result' in data and str(ncbi_id) in data['result']:
                # Store full response in cache
                cache[str(ncbi_id)] = data['result'][str(ncbi_id)]
                # Save updated cache
                save_gene_info_cache(cache, cache_file)
                
                # Return formatted info
                gene_info = data['result'][str(ncbi_id)]
                return {
                    'gene_id': str(ncbi_id),
                    'name': gene_info.get('name', ''),
                    'description': gene_info.get('description', ''),
                    'summary': gene_info.get('summary', '')
                }
    except Exception as e:
        print(f"Error looking up gene {ncbi_id}: {e}")
        print(f"Full URL that failed: {server + ext}")
    
    return None

def build_wd_to_ncbi_map(gff_path: str) -> dict:
    """
    Build a mapping from WD gene IDs to NCBI Gene IDs using the GFF file.
    Returns: dict mapping WDxxxxxx -> ncbi_numeric_id (as string)
    """
    mapping = {}
    print(f"Building WD to NCBI mapping from {gff_path}")
    
    try:
        with open(gff_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] != 'gene':
                    continue
                attrs = dict(attr.split('=') for attr in parts[8].split(';') if '=' in attr)
                wd_id = attrs.get('ID', None)
                # Remove 'gene-' prefix if present
                if wd_id and wd_id.startswith('gene-'):
                    wd_id = wd_id[5:]  # Remove 'gene-' prefix
                dbxref = attrs.get('Dbxref', '')
                ncbi_id = None
                if 'GeneID:' in dbxref:
                    ncbi_id = dbxref.split('GeneID:')[1]
                if wd_id and ncbi_id:
                    mapping[wd_id] = ncbi_id
        
        print(f"Found {len(mapping)} WD to NCBI mappings")
        print("First few mappings:")
        for i, (wd, ncbi) in enumerate(mapping.items()):
            if i < 5:  # Print first 5 mappings
                print(f"{wd} -> {ncbi}")
            else:
                break
                
        return mapping
    except Exception as e:
        print(f"Error building WD to NCBI mapping: {e}")
        return {}


def load_sample_results(results_dir: str, gene_lengths: Dict[str, int], output_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all sample results and convert to DataFrames."""
    sample_dfs = {}
    
    # First load gene statistics to get coverage information
    gene_stats_file = os.path.join(results_dir, 'gene_statistics.json')
    gene_coverage = {}
    if os.path.exists(gene_stats_file):
        with open(gene_stats_file) as f:
            gene_stats = json.load(f)
            # Calculate average coverage across samples for each gene
            for sample, stats in gene_stats.items():
                for gene_id, gene_data in stats.items():
                    if gene_id not in gene_coverage:
                        gene_coverage[gene_id] = []
                    gene_coverage[gene_id].append(gene_data.get('average_coverage', 0))
            
            # Calculate mean coverage for each gene
            gene_coverage = {gene: np.mean(covs) for gene, covs in gene_coverage.items()}
    
    # Load both per-sample results and pooled control/treated results
    result_files = list(Path(results_dir).glob('*_results.json'))
    result_files += list(Path(results_dir).glob('permuted_provean_results_*.json'))
    
    for file in result_files:
        sample = file.stem.replace('_results', '')
        # Map pooled outputs to simpler sample names for downstream logic
        if sample == 'permuted_provean_results_controls':
            sample = 'NT_controls'
        elif sample == 'permuted_provean_results_treated':
            sample = 'EMS_treated'
        
        with open(file) as f:
            data = json.load(f)
        
        # Convert to DataFrame
        rows = []
        for gene_id, values in data.items():
            
            # Calculate p-value (two-tailed) for effect score
            perm_scores = values['permutation']['scores']
            effect_score = values['effect']
            if len(perm_scores) <= 10:
                pvalue_more = None
                pvalue_less = None
            else:
                pvalue_more, pvalue_less = calculate_summed_effect_pvalue(effect_score, perm_scores)

            # Calculate p-values for deleterious ratio
            tm = values.get('mutation_stats', {}).get('total_mutations', 0)
            dm = values.get('mutation_stats', {}).get('deleterious_count', 0)
            if tm and tm > 0:
                deleterious_ratio = dm / tm
            else:
                deleterious_ratio = np.nan
            
            pvalue_ratio_more = None
            pvalue_ratio_less = None
            # Calculate deleterious ratios for each permutation using their own total mutations
            perm_del_counts = values['permutation'].get('deleterious_counts', [])
            perm_total_mutations = values['permutation'].get('total_mutations', [])
            if len(perm_scores) > 10 and perm_del_counts and perm_total_mutations and len(perm_del_counts) == len(perm_total_mutations):
                perm_ratios = [
                    (dc / tm_i) if tm_i and tm_i > 0 else 0.0
                    for dc, tm_i in zip(perm_del_counts, perm_total_mutations)
                ]
                if not np.isnan(deleterious_ratio):
                    pvalue_ratio_more, pvalue_ratio_less = calculate_deleterious_ratio_pvalues(deleterious_ratio, perm_ratios)
            
            # Calculate p-values for non-synonymous mutation counts
            obs_non_syn = values.get('mutation_stats', {}).get('non_syn_muts', np.nan)
            perm_non_syn = values['permutation'].get('non_syn_counts', None)
            pvalue_nonsyn_more = None
            pvalue_nonsyn_less = None
            if isinstance(obs_non_syn, (int, float)) and not np.isnan(obs_non_syn) and perm_non_syn and len(perm_non_syn) > 10:
                # Two-tailed empirical p-values: counts are naturally one-sided higher=more nonsyn
                n_more = sum(x > obs_non_syn for x in perm_non_syn)
                n_less = sum(x < obs_non_syn for x in perm_non_syn)
                pvalue_nonsyn_more = n_more / len(perm_non_syn)
                pvalue_nonsyn_less = n_less / len(perm_non_syn)
            
            # Derive syn_muts if present; else compute from total - non_syn if available
            obs_syn = values.get('mutation_stats', {}).get('syn_muts', np.nan)
            if (isinstance(obs_syn, (int, float)) and np.isnan(obs_syn)) and isinstance(obs_non_syn, (int, float)) and isinstance(tm, (int, float)):
                obs_syn = (tm - obs_non_syn) if (tm is not None and obs_non_syn is not None) else np.nan
            # dN/dS per gene when syn > 0
            dn_ds = (obs_non_syn / obs_syn) if isinstance(obs_non_syn, (int, float)) and isinstance(obs_syn, (int, float)) and obs_syn not in (0, 0.0) else np.nan
            # dN/dS permutation-based p-values from perm_non and perm_total
            pvalue_dnds_more = None
            pvalue_dnds_less = None
            perm_total = values['permutation'].get('total_mutations', None)
            if perm_non_syn and perm_total and len(perm_non_syn) == len(perm_total) and len(perm_non_syn) > 10 and isinstance(dn_ds, (int, float)) and not np.isnan(dn_ds):
                _non = np.array(perm_non_syn, dtype=float)
                _tot = np.array(perm_total, dtype=float)
                _syn = _tot - _non
                with np.errstate(divide='ignore', invalid='ignore'):
                    _dn = np.divide(_non, _syn)
                _dn = _dn[np.isfinite(_dn)]
                if _dn.size > 10:
                    n_more = int(np.sum(_dn > dn_ds))
                    n_less = int(np.sum(_dn < dn_ds))
                    pvalue_dnds_more = n_more / _dn.size
                    pvalue_dnds_less = n_less / _dn.size

            row = {
                'gene_id': gene_id,
                'effect_score': effect_score,
                'pvalue_more': pvalue_more,
                'pvalue_less': pvalue_less,
                'pvalue_ratio_more': pvalue_ratio_more,
                'pvalue_ratio_less': pvalue_ratio_less,
                'pvalue_nonsyn_more': pvalue_nonsyn_more,
                'pvalue_nonsyn_less': pvalue_nonsyn_less,
                'pvalue_dnds_more': pvalue_dnds_more,
                'pvalue_dnds_less': pvalue_dnds_less,
                'permutation_mean': values['permutation'].get('mean', 0),
                'permutation_std': values['permutation'].get('std', 0),
                'n_permutations': len(perm_scores),
                'deleterious_mutations': dm,
                'total_mutations': tm,
                'deleterious_ratio': deleterious_ratio,
                'non_syn_muts': obs_non_syn,
                'syn_muts': obs_syn,
                'dn_ds': dn_ds,
                'avg_cov': gene_coverage.get(gene_id, 0)  # Use average coverage from gene statistics
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Calculate normalized scores using both gene lengths and average coverage
        df['effect_norm'] = df.apply(lambda x: x['effect_score'] / (gene_lengths.get(x['gene_id'], 1) * max(x['avg_cov'], 1)), axis=1)
        
        # Apply FDR corrections
        valid_pvals_more = df['pvalue_more'].notna()
        if valid_pvals_more.any():
            df.loc[valid_pvals_more, 'pvalue_more_fdr'] = multipletests(df.loc[valid_pvals_more, 'pvalue_more'], method='fdr_bh')[1]
        
        valid_pvals_less = df['pvalue_less'].notna()
        if valid_pvals_less.any():
            df.loc[valid_pvals_less, 'pvalue_less_fdr'] = multipletests(df.loc[valid_pvals_less, 'pvalue_less'], method='fdr_bh')[1]
        
        # Re-enabled: deleterious ratio FDR corrections
        valid_ratio_more = df['pvalue_ratio_more'].notna()
        if valid_ratio_more.any():
            df.loc[valid_ratio_more, 'pvalue_ratio_more_fdr'] = multipletests(df.loc[valid_ratio_more, 'pvalue_ratio_more'], method='fdr_bh')[1]
        
        valid_ratio_less = df['pvalue_ratio_less'].notna()
        if valid_ratio_less.any():
            df.loc[valid_ratio_less, 'pvalue_ratio_less_fdr'] = multipletests(df.loc[valid_ratio_less, 'pvalue_ratio_less'], method='fdr_bh')[1] 
        
        # New: non-synonymous count FDR corrections
        valid_ns_more = df['pvalue_nonsyn_more'].notna()
        if valid_ns_more.any():
            df.loc[valid_ns_more, 'pvalue_nonsyn_more_fdr'] = multipletests(df.loc[valid_ns_more, 'pvalue_nonsyn_more'], method='fdr_bh')[1]
        valid_ns_less = df['pvalue_nonsyn_less'].notna()
        if valid_ns_less.any():
            df.loc[valid_ns_less, 'pvalue_nonsyn_less_fdr'] = multipletests(df.loc[valid_ns_less, 'pvalue_nonsyn_less'], method='fdr_bh')[1]
        
        # New: dN/dS FDR corrections
        if 'pvalue_dnds_more' in df.columns:
            valid_dnds_more = df['pvalue_dnds_more'].notna()
            if valid_dnds_more.any():
                df.loc[valid_dnds_more, 'pvalue_dnds_more_fdr'] = multipletests(df.loc[valid_dnds_more, 'pvalue_dnds_more'], method='fdr_bh')[1]
        if 'pvalue_dnds_less' in df.columns:
            valid_dnds_less = df['pvalue_dnds_less'].notna()
            if valid_dnds_less.any():
                df.loc[valid_dnds_less, 'pvalue_dnds_less_fdr'] = multipletests(df.loc[valid_dnds_less, 'pvalue_dnds_less'], method='fdr_bh')[1]
        
        sample_dfs[sample] = df
        
    return sample_dfs

def calculate_summed_effect_pvalue(observed_value: float, null_distribution: List[float]) -> float:
    """Calculate proper two-tailed empirical p-value.
    
    Returns the fraction of the null distribution that is as or more extreme than
    the observed value in either direction.
    """
    if len(null_distribution) <= 10:
        return None
    
    # Count values more extreme in either tail
    n_more_effect = sum(x < observed_value for x in null_distribution)
    n_less_effect = sum(x > observed_value for x in null_distribution)
    
    # Calculate p-value as fraction of distribution more extreme than observed
    pvalue_more = n_more_effect / len(null_distribution)
    pvalue_less = n_less_effect / len(null_distribution)
    
    return pvalue_more, pvalue_less

# Re-enabled: deleterious ratio p-value calculation
def calculate_deleterious_ratio_pvalues(observed_ratio: float, perm_ratios: List[float]) -> tuple[float, float]:
    """Calculate p-values based on deleterious ratio distribution.
    
    For deleterious ratios:
    - Higher ratios indicate more deleterious mutations
    - P-value is fraction of background ratios that are more extreme
    - Low p-values indicate unusually high or low ratios
    """
    if len(perm_ratios) <= 10:
        return None, None
    
    n_more_deleterious = sum(x > observed_ratio for x in perm_ratios)
    n_less_deleterious = sum(x < observed_ratio for x in perm_ratios)
    
    pvalue_more = n_more_deleterious / len(perm_ratios)
    pvalue_less = n_less_deleterious / len(perm_ratios)
    return pvalue_more, pvalue_less

def create_significant_genes_summary(df: pd.DataFrame, sample: str, output_dir: str, cache_file: str = None, gene_strands: Dict[str, str] = None, pvalue_threshold: float = 0.05) -> None:
    """Create a summary CSV file of significant genes."""
    # Filter out None/NaN p-values
    df_valid = df.dropna(subset=['pvalue_more_fdr', 'pvalue_less_fdr'])
    
    # Filter for significant genes (both deleterious and protective)
    deleterious = (df_valid['pvalue_more_fdr'] < pvalue_threshold) & (df_valid['effect_score'] < 0)
    protective = (df_valid['pvalue_less_fdr'] < pvalue_threshold) & (df_valid['effect_score'] > 0)
    sig_genes = df_valid[deleterious | protective].copy()
    
    # Add strand information if available
    if gene_strands:
        sig_genes['strand'] = sig_genes['gene_id'].map(gene_strands)
    
    # Sort by absolute effect score (strongest effects first)
    sig_genes['abs_effect'] = abs(sig_genes['effect_score'])
    sig_genes = sig_genes.sort_values('abs_effect', ascending=False)
    
    # Select and rename columns for the summary
    summary_columns = [
        'gene_id',
        'effect_score',
        'effect_norm',
        'pvalue_more',
        'pvalue_less',
        'pvalue_more_fdr',
        'pvalue_less_fdr',
        'permutation_mean',
        'permutation_std',
        'n_permutations',
        'deleterious_mutations',
        'total_mutations'
        # 'deleterious_ratio'  # COMMENTED OUT FOR NOW
    ]
    
    # Add strand column if available
    if gene_strands:
        summary_columns.append('strand')
    
    summary_df = sig_genes[summary_columns].copy()
    
    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        'effect_score': 'raw_effect_score',
        'effect_norm': 'length_normalized_score',
        'permutation_mean': 'permutation_mean_score',
        'permutation_std': 'permutation_std_dev',
        'n_permutations': 'number_of_permutations',
        # 'deleterious_ratio': 'deleterious_mutation_ratio'  # COMMENTED OUT FOR NOW
    })
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"{sample}_significant_genes.csv")
    summary_df.to_csv(output_file, index=False)
    
    # Get gene info for each significant gene using provided cache file
    for gene_id in sig_genes['gene_id']:
        gene_info = lookup_gene_info(gene_id, cache_file)
        if gene_info:
            sig_genes.loc[sig_genes['gene_id'] == gene_id, 'gene_name'] = gene_info['name']
            sig_genes.loc[sig_genes['gene_id'] == gene_id, 'gene_description'] = gene_info['description']
            sig_genes.loc[sig_genes['gene_id'] == gene_id, 'gene_summary'] = gene_info['summary']
    
    # Make a new summary file with expanded gene info
    gene_summary_columns = {
        'gene_id': sig_genes['gene_id'],
        'gene_name': sig_genes.get('gene_name', ''),
        'gene_description': sig_genes.get('gene_description', ''),
        'effect_score': sig_genes['effect_score'],
        'effect_norm': sig_genes['effect_norm'],
        # 'deleterious_ratio': sig_genes['deleterious_ratio'],  # COMMENTED OUT FOR NOW
        'pvalue_more': sig_genes['pvalue_more'],
        'pvalue_less': sig_genes['pvalue_less'],
        'pvalue_more_fdr': sig_genes['pvalue_more_fdr'],
        'pvalue_less_fdr': sig_genes['pvalue_less_fdr']
    }
    
    # Add strand information if available
    if gene_strands:
        gene_summary_columns['strand'] = sig_genes.get('strand', '')
    
    gene_summary_df = pd.DataFrame(gene_summary_columns)

    # Sort by effect score (most negative first)
    gene_summary_df = gene_summary_df.sort_values('effect_score')

    # Save gene info summary
    gene_info_file = os.path.join(output_dir, f"{sample}_gene_info_summary.csv")
    gene_summary_df.to_csv(gene_info_file, index=False)
    
    # Print summary
    print(f"\nSignificant genes summary for {sample}:")
    print(f"Total significant genes: {len(sig_genes)}")
    print(f"Strongest effect: {sig_genes['effect_score'].min():.2f}")
    print(f"Summary saved to: {output_file}")

def analyze_sample(df: pd.DataFrame, sample: str, output_dir: str, cache_file: str, gene_lengths: Dict[str, int], gene_strands: Dict[str, str] = None) -> None:
    """Analyze a single sample's results."""
    sample_dir = os.path.join(output_dir, sample)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create plots
    plot_effect_relationships(df, gene_lengths, sample, sample_dir)
    # plot_effect_relationships_ratio(df, gene_lengths, sample, sample_dir)  # COMMENTED OUT FOR NOW
    
            # Add method comparison
        # compare_scoring_methods(df, gene_lengths, sample, sample_dir)  # COMMENTED OUT FOR NOW
    
    # New mirrored metric plots
    plot_gene_metrics_vs_length(df, gene_lengths, sample, sample_dir)
    
    # Create significant genes summary
    create_significant_genes_summary(df, sample, sample_dir, cache_file, gene_strands)
    
    # List significant genes
    list_significant_genes(df, sample, sample_dir, cache_file, gene_strands)

def plot_combined_pvalue_density(sample_dfs: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Create combined histogram plots of p-values for all samples."""
    plt.figure(figsize=(12, 8))

    # Collect p-values for controls and treated samples
    control_pvals_more = []
    treated_pvals_more = []
    control_pvals_less = []
    treated_pvals_less = []
    control_pvals_ratio_more = []
    treated_pvals_ratio_more = []
    control_pvals_ratio_less = []
    treated_pvals_ratio_less = []
    # FDR
    control_pvals_more_fdr = []
    treated_pvals_more_fdr = []
    control_pvals_less_fdr = []
    treated_pvals_less_fdr = []
    control_pvals_ratio_more_fdr = []
    treated_pvals_ratio_more_fdr = []
    control_pvals_ratio_less_fdr = []
    treated_pvals_ratio_less_fdr = []

    for sample, df in sample_dfs.items():
        if 'NT' in sample:
            control_pvals_more.extend(df['pvalue_more'].dropna().values)
            control_pvals_less.extend(df['pvalue_less'].dropna().values)
            control_pvals_ratio_more.extend(df['pvalue_ratio_more'].dropna().values)
            control_pvals_ratio_less.extend(df['pvalue_ratio_less'].dropna().values)
            if 'pvalue_more_fdr' in df:
                control_pvals_more_fdr.extend(df['pvalue_more_fdr'].dropna().values)
            if 'pvalue_less_fdr' in df:
                control_pvals_less_fdr.extend(df['pvalue_less_fdr'].dropna().values)
            if 'pvalue_ratio_more_fdr' in df:
                control_pvals_ratio_more_fdr.extend(df['pvalue_ratio_more_fdr'].dropna().values)
            if 'pvalue_ratio_less_fdr' in df:
                control_pvals_ratio_less_fdr.extend(df['pvalue_ratio_less_fdr'].dropna().values)
        else:
            treated_pvals_more.extend(df['pvalue_more'].dropna().values)
            treated_pvals_less.extend(df['pvalue_less'].dropna().values)
            treated_pvals_ratio_more.extend(df['pvalue_ratio_more'].dropna().values)
            treated_pvals_ratio_less.extend(df['pvalue_ratio_less'].dropna().values)
            if 'pvalue_more_fdr' in df:
                treated_pvals_more_fdr.extend(df['pvalue_more_fdr'].dropna().values)
            if 'pvalue_less_fdr' in df:
                treated_pvals_less_fdr.extend(df['pvalue_less_fdr'].dropna().values)
            if 'pvalue_ratio_more_fdr' in df:
                treated_pvals_ratio_more_fdr.extend(df['pvalue_ratio_more_fdr'].dropna().values)
            if 'pvalue_ratio_less_fdr' in df:
                treated_pvals_ratio_less_fdr.extend(df['pvalue_ratio_less_fdr'].dropna().values)

    # Example: plot for effect_more
    bins = np.linspace(0, 1, 51)
    plt.hist([control_pvals_more, treated_pvals_more], 
             bins=bins,
             label=['Control', 'EMS Treated'],
             stacked=True,
             alpha=0.7,
             color=['lightgrey', 'darkorange'])
    plt.axvline(x=0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05 threshold')
    plt.xlabel('p-value (effect_more)')
    plt.ylabel('Count')
    plt.title('P-value Distribution (effect_more) Across Samples')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_pvalue_histogram_effect_more.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Repeat for other pvalue types as needed, or remove this function if you use the new create_pvalue_distribution_plots

def analyze_shared_genes(sample_dfs: Dict[str, pd.DataFrame], output_dir: str, gene_strands: Dict[str, str] = None, pvalue_threshold: float = 0.05) -> None:
    """Analyze shared significant genes between samples."""
    
    print("\nStarting shared gene analysis...")
    
    # Separate control and treated samples first
    control_samples = []
    treated_samples = []
    
    # Identify treated and control samples
    for sample in sample_dfs:
        if 'NT' in sample:
            control_samples.append(sample)
        elif '7d' not in sample:  # Only add non-7d treated samples
            treated_samples.append(sample)
    
    print(f"Control samples: {control_samples}")
    print(f"Treated samples (excluding 7d): {treated_samples}")
    
    # Get significant genes for each treated sample
    treated_sig_genes = []
    for sample in treated_samples:
        df = sample_dfs[sample]
        # Get significant genes
        sig_genes = set(df[df['pvalue_more_fdr'] < pvalue_threshold]['gene_id'])
        print(f"Sample {sample} has {len(sig_genes)} significant genes")
        treated_sig_genes.append(sig_genes)
    
    # Find genes present in ALL treated samples
    if treated_sig_genes:
        genes_in_all_treated = set.intersection(*treated_sig_genes)
        print(f"\nFound {len(genes_in_all_treated)} genes significant in all treated samples")
        
        # Debug: print first few genes
        if genes_in_all_treated:
            print("First few shared genes:", list(genes_in_all_treated)[:5])
    else:
        genes_in_all_treated = set()
        print("No treated samples to analyze")
    
    # Create merged mutation data for these genes
    merged_data = []
    for gene in genes_in_all_treated:
        # Initialize counters
        total_mutations = 0
        deleterious_mutations = 0
        effect_scores = []
        pvalues = []
        
        # Count mutations across all treated samples
        for sample in treated_samples:
            df = sample_dfs[sample]
            gene_row = df[df['gene_id'] == gene].iloc[0]
            total_mutations += gene_row['total_mutations']
            deleterious_mutations += gene_row['deleterious_mutations']  # Use actual count instead of calculating
            effect_scores.append(gene_row['effect_score'])
            pvalues.append(gene_row['pvalue_more_fdr'])
        
        # Count control samples where this gene is significant
        control_count = sum(1 for control in control_samples 
                          if gene in set(sample_dfs[control][sample_dfs[control]['pvalue_more_fdr'] < pvalue_threshold]['gene_id']))
        
        gene_data = {
            'gene_id': gene,
            'total_mutations': total_mutations,
            'deleterious_mutations': deleterious_mutations,
            # 'deleterious_ratio': deleterious_mutations / total_mutations if total_mutations > 0 else 0,  # COMMENTED OUT FOR NOW
            'mean_effect_score': np.mean(effect_scores),
            'mean_pvalue': np.mean(pvalues),
            'control_samples_count': control_count
        }
        
        # Add strand information if available
        if gene_strands and gene in gene_strands:
            gene_data['strand'] = gene_strands[gene]
        
        merged_data.append(gene_data)
    
    # Create and save DataFrame
    if merged_data:
        print("\nCreating merged mutation analysis DataFrame...")
        df = pd.DataFrame(merged_data)
        df = df.sort_values('mean_effect_score')  # Sort by effect score
        output_file = os.path.join(output_dir, 'merged_mutation_analysis.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved merged analysis to: {output_file}")
        print(f"DataFrame shape: {df.shape}")
    else:
        print("\nNo merged data to save!")

def plot_effect_relationships(df: pd.DataFrame, gene_lengths: Dict[str, int], sample: str, output_dir: str) -> None:
    """Create plots showing relationships between effect score and gene properties."""
    # Define significance using FDR corrected p-values
    deleterious = (df['pvalue_more_fdr'] < 0.05)
    protective = (df['pvalue_less_fdr'] < 0.05)
    nonsig = ~(deleterious | protective)

    # Get gene lengths for plotting
    gene_lens = df['gene_id'].map(lambda x: gene_lengths.get(x, 1))

    # Fixed bounds and single-axis with outlier markers
    x_min = 75
    x_max = 8000.0
    y_main_low, y_main_high = -700.0, 10.0

    # Masks
    within = (gene_lens >= x_min) & (gene_lens <= x_max) & \
             (df['effect_score'] >= y_main_low) & (df['effect_score'] <= y_main_high)
    below = (gene_lens >= x_min) & (gene_lens <= x_max) & (df['effect_score'] < y_main_low)
    above = (gene_lens >= x_min) & (gene_lens <= x_max) & (df['effect_score'] > y_main_high)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Inliers
    ax.scatter(gene_lens[nonsig & within], df[nonsig & within]['effect_score'],
               alpha=0.3, color='grey', label='Non-significant')
    ax.scatter(gene_lens[deleterious & within], df[deleterious & within]['effect_score'],
               alpha=0.7, color='darkorange', label='Deleterious')
    ax.scatter(gene_lens[protective & within], df[protective & within]['effect_score'],
               alpha=0.7, color='blue', label='Neutral/Protective')

    # Outlier markers aggregated by x-bins and offset from bounds for readability
    pad = 0.03 * (y_main_high - y_main_low)
    bins = np.logspace(np.log10(x_min), np.log10(x_max), 25)
    centers = np.sqrt(bins[:-1] * bins[1:])
    # Below-range aggregated markers
    if below.any():
        counts_below, _ = np.histogram(gene_lens[below], bins=bins)
        mask = counts_below > 0
        sizes = 20 + 18 * np.sqrt(counts_below[mask])
        ax.scatter(centers[mask], np.full(mask.sum(), y_main_low + pad),
                   marker='v', s=sizes, facecolors='none', edgecolors='black', linewidths=0.6,
                   alpha=0.9, label='Below range', zorder=5)
    # Above-range aggregated markers
    if above.any():
        counts_above, _ = np.histogram(gene_lens[above], bins=bins)
        mask = counts_above > 0
        sizes = 20 + 18 * np.sqrt(counts_above[mask])
        ax.scatter(centers[mask], np.full(mask.sum(), y_main_high - pad),
                   marker='^', s=sizes, facecolors='none', edgecolors='black', linewidths=0.6,
                   alpha=0.9, label='Above range', zorder=5)

    # Annotate counts
    n_below = int(below.sum())
    n_above = int(above.sum())
    if n_below:
        ax.text(0.99, 0.02, f"{n_below} below", transform=ax.transAxes, fontsize=12,
                va='bottom', ha='right', color='black')
    if n_above:
        ax.text(0.01, 0.98, f"{n_above} above", transform=ax.transAxes, fontsize=12,
                va='top', ha='left', color='black')

    ax.set_xlabel('Gene Length (bp)', fontsize=16)
    ax.set_ylabel('Effect Score', fontsize=16)
    ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_main_low, y_main_high)
    # Force non-scientific tick labels on log axis
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Move larger legend into bottom-left where whitespace is available
    ax.legend(loc='lower left', frameon=True, fontsize=13, fancybox=True, framealpha=0.85, borderpad=0.7)

    ax.set_title('Effect Score vs Gene Length - Treated', fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{sample}_effect_relationships.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_significance_counts(sample_dfs: Dict[str, pd.DataFrame], output_dir: str, cache_file: str = None) -> None:
    """Create table showing how many treated/control samples each gene is significant in."""
    
    # Load gene info cache
    try:
        with open(cache_file) as f:
            gene_info = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Gene info cache not found at {cache_file}")
        gene_info = {}
    
    # Separate control and treated samples
    control_samples = []
    treated_samples = []
    
    for sample in sample_dfs:
        if 'NT' in sample:
            control_samples.append(sample)
        elif '7d' not in sample:  # Exclude 7d samples
            treated_samples.append(sample)
    
    print(f"Found {len(treated_samples)} treated samples and {len(control_samples)} control samples")
    
    # Get genes significant by effect score measure (using FDR corrected p-values)
    all_genes_low_p = set()
    all_genes_high_p = set()
    # all_genes_high_ratio = set()  # COMMENTED OUT FOR NOW
    # all_genes_low_ratio = set()   # COMMENTED OUT FOR NOW
    
    for df in sample_dfs.values():
        # Effect score based (using FDR corrected p-values)
        all_genes_low_p.update(set(df[df['pvalue_more_fdr'] < 0.05]['gene_id']))
        all_genes_high_p.update(set(df[df['pvalue_less_fdr'] > (1 - 0.05)]['gene_id']))
        # Ratio based (already FDR corrected) - COMMENTED OUT FOR NOW
        # all_genes_high_ratio.update(set(df[df['pvalue_ratio_more_fdr'] < 0.05]['gene_id']))
        # all_genes_low_ratio.update(set(df[df['pvalue_ratio_less_fdr'] > 0.99]['gene_id']))

def plot_merged_significance(df: pd.DataFrame, output_dir: str, analysis_type: str = 'deleterious') -> None:
    """Create scatter plots of merged mutation data colored by significance count."""
    
    # Plot 1: Effect Score Analysis
    plt.figure(figsize=(12, 8))
    max_treated = df['treated_samples_significant'].max()
    norm = plt.Normalize(1, max_treated)
    cmap = plt.cm.viridis
    
    # Create scatter plot for effect scores
    scatter = plt.scatter(df['total_mutations'], 
                         df['mean_effect_score'],
                         c=df['treated_samples_significant'],
                         cmap=cmap,
                         norm=norm,
                         alpha=0.6,
                         s=50)
    
    plt.xlabel('Total Mutations (All Treated Samples)')
    plt.ylabel('Mean Effect Score')
    title_prefix = 'Protective' if 'protective' in analysis_type else 'Deleterious'
    plt.title(f'{title_prefix} Effect Score Analysis Across Treated Samples')
    plt.xscale('log')  # Log scale for total mutations
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Treated Samples Significant In')
    
    # Save effect score plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'merged_effect_score_{analysis_type}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Deleterious Ratio Analysis
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot for deleterious ratios
    scatter = plt.scatter(df['total_mutations'], 
                         df['deleterious_ratio'],
                         c=df['treated_samples_significant'],
                         cmap=cmap,
                         norm=norm,
                         alpha=0.6,
                         s=50)
    
    plt.xlabel('Total Mutations (All Treated Samples)')
    plt.ylabel('Deleterious Ratio')
    plt.title(f'{title_prefix} Mutation Ratio Analysis Across Treated Samples')
    plt.xscale('log')  # Log scale for total mutations
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Treated Samples Significant In')
    
    # Save deleterious ratio plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'merged_mutation_ratio_{analysis_type}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_volcano(df: pd.DataFrame, title: str, score_col: str = 'effect_score', 
                output_dir: str = None, sample: str = None, output_prefix: str = '') -> None:
    """Create volcano plot of effect scores vs -log10(pvalue)."""
    plt.figure(figsize=(10, 8))
    
    # Filter out None/NaN p-values
    df_valid = df.dropna(subset=['pvalue_more_fdr'])
    
    # Define significance using FDR corrected p-values
    deleterious = (df_valid['pvalue_more_fdr'] < 0.05) & (df_valid['effect_score'] < 0)
    protective = (df_valid['pvalue_less_fdr'] < 0.05) & (df_valid['effect_score'] > 0)
    nonsig = ~(deleterious | protective)
    
    # Plot points
    plt.scatter(df_valid[nonsig][score_col], 
               -np.log10(df_valid[nonsig]['pvalue_more_fdr']),
               alpha=0.5, color='grey', label='Non-significant')
    
    plt.scatter(df_valid[deleterious][score_col],
                -np.log10(df_valid[deleterious]['pvalue_more_fdr']),
                alpha=0.7, color='red', label='Deleterious (p < 0.05)')
    
    plt.scatter(df_valid[protective][score_col],
                -np.log10(df_valid[protective]['pvalue_less_fdr']),
                alpha=0.7, color='blue', label='Protective (p > 0.95)')
    
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.5)
    plt.xlabel(f'{score_col}')
    plt.ylabel('-log10(p-value)')
    plt.title(f'{title}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_prefix}volcano.png'), dpi=300, bbox_inches='tight')
    plt.close()

def list_significant_genes(df: pd.DataFrame, sample: str, output_dir: str, cache_file: str = None, gene_strands: Dict[str, str] = None) -> None:
    """List genes with significant FDR-corrected p-values in 4 separate categories."""
    
    # Add gene descriptions
    gene_info_cache = load_gene_info_cache(cache_file) if cache_file else {}
    
    # Function to add gene descriptions to a dataframe
    def add_gene_info(df_subset):
        result_df = df_subset.copy()
        result_df['gene_description'] = ''
        
        for i, row in result_df.iterrows():
            gene_id = row['gene_id']
            if str(gene_id) in gene_info_cache:
                result_df.at[i, 'gene_description'] = gene_info_cache[str(gene_id)].get('description', '')
            else:
                # Try to look up if not in cache
                gene_info = lookup_gene_info(gene_id, cache_file) if cache_file else None
                if gene_info:
                    result_df.at[i, 'gene_description'] = gene_info.get('description', '')
        
        # Add strand information if available
        if gene_strands:
            result_df['strand'] = result_df['gene_id'].map(gene_strands)
        
        # Select and reorder columns based on what's available
        base_columns = [
            'gene_id', 
            'gene_description', 
            'effect_score',
            'deleterious_mutations',
            'total_mutations',
            'deleterious_ratio'
        ]
        
        # Add strand column if available
        if gene_strands:
            base_columns.append('strand')
        
        # Add p-value columns that exist
        available_pval_cols = [col for col in ['pvalue_more_fdr', 'pvalue_less_fdr', 'pvalue_ratio_more_fdr', 'pvalue_ratio_less_fdr', 'pvalue_nonsyn_more_fdr', 'pvalue_nonsyn_less_fdr'] 
                              if col in result_df.columns]
        
        columns = base_columns + available_pval_cols
        
        return result_df[columns]
    
    # Check which FDR columns exist
    has_effect_fdr = 'pvalue_more_fdr' in df.columns and 'pvalue_less_fdr' in df.columns
    has_ratio_fdr = 'pvalue_ratio_more_fdr' in df.columns and 'pvalue_ratio_less_fdr' in df.columns
    
    # Initialize empty DataFrames
    effect_more_df = pd.DataFrame()
    effect_less_df = pd.DataFrame()
    ratio_more_df = pd.DataFrame()
    ratio_less_df = pd.DataFrame()
    nonsyn_more_df = pd.DataFrame()
    nonsyn_less_df = pd.DataFrame()
    
    # 1. Genes with significant pvalue_more_fdr < 0.05 (deleterious effect)
    if has_effect_fdr:
        significant_effect_more = df[df['pvalue_more_fdr'] < 0.05].dropna(subset=['pvalue_more_fdr'])
        effect_more_df = add_gene_info(significant_effect_more)
        if not effect_more_df.empty:
            effect_more_df = effect_more_df.sort_values('pvalue_more_fdr')
    
    # 2. Genes with significant pvalue_less_fdr < 0.05 (protective effect)  
    if has_effect_fdr:
        significant_effect_less = df[df['pvalue_less_fdr'] < 0.05].dropna(subset=['pvalue_less_fdr'])
        effect_less_df = add_gene_info(significant_effect_less)
        if not effect_less_df.empty:
            effect_less_df = effect_less_df.sort_values('pvalue_less_fdr')
    
    # 3. Genes with significant pvalue_ratio_more_fdr < 0.05 (high deleterious ratio)
    if has_ratio_fdr:
        significant_ratio_more = df[df['pvalue_ratio_more_fdr'] < 0.05].dropna(subset=['pvalue_ratio_more_fdr'])
        ratio_more_df = add_gene_info(significant_ratio_more)
        if not ratio_more_df.empty:
            ratio_more_df = ratio_more_df.sort_values('pvalue_ratio_more_fdr')
    
    # 4. Genes with significant pvalue_ratio_less_fdr < 0.05 (low deleterious ratio)
    if has_ratio_fdr:
        significant_ratio_less = df[df['pvalue_ratio_less_fdr'] < 0.05].dropna(subset=['pvalue_ratio_less_fdr'])
        ratio_less_df = add_gene_info(significant_ratio_less)
        if not ratio_less_df.empty:
            ratio_less_df = ratio_less_df.sort_values('pvalue_ratio_less_fdr')
    
    # 5. Non-synonymous count significance (more and less)
    if 'pvalue_nonsyn_more_fdr' in df.columns:
        significant_ns_more = df[df['pvalue_nonsyn_more_fdr'] < 0.05].dropna(subset=['pvalue_nonsyn_more_fdr'])
        nonsyn_more_df = add_gene_info(significant_ns_more)
        if not nonsyn_more_df.empty and 'pvalue_nonsyn_more_fdr' in nonsyn_more_df.columns:
            nonsyn_more_df = nonsyn_more_df.sort_values('pvalue_nonsyn_more_fdr')
    if 'pvalue_nonsyn_less_fdr' in df.columns:
        significant_ns_less = df[df['pvalue_nonsyn_less_fdr'] < 0.05].dropna(subset=['pvalue_nonsyn_less_fdr'])
        nonsyn_less_df = add_gene_info(significant_ns_less)
        if not nonsyn_less_df.empty and 'pvalue_nonsyn_less_fdr' in nonsyn_less_df.columns:
            nonsyn_less_df = nonsyn_less_df.sort_values('pvalue_nonsyn_less_fdr')
    
    # Save to CSV files
    effect_more_file = os.path.join(output_dir, f"{sample}_significant_effect_more_genes.csv")
    effect_less_file = os.path.join(output_dir, f"{sample}_significant_effect_less_genes.csv")
    ratio_more_file = os.path.join(output_dir, f"{sample}_significant_ratio_more_genes.csv")
    ratio_less_file = os.path.join(output_dir, f"{sample}_significant_ratio_less_genes.csv")
    nonsyn_more_file = os.path.join(output_dir, f"{sample}_significant_nonsyn_more_genes.csv")
    nonsyn_less_file = os.path.join(output_dir, f"{sample}_significant_nonsyn_less_genes.csv")
    
    effect_more_df.to_csv(effect_more_file, index=False)
    effect_less_df.to_csv(effect_less_file, index=False)
    ratio_more_df.to_csv(ratio_more_file, index=False)
    ratio_less_df.to_csv(ratio_less_file, index=False)
    nonsyn_more_df.to_csv(nonsyn_more_file, index=False)
    nonsyn_less_df.to_csv(nonsyn_less_file, index=False)
    
    print(f"\nSignificant genes for {sample}:")
    print(f"Effect more significant (pvalue_more_fdr < 0.05): {len(effect_more_df)}")
    print(f"Effect less significant (pvalue_less_fdr < 0.05): {len(effect_less_df)}")
    print(f"Ratio more significant (pvalue_ratio_more_fdr < 0.05): {len(ratio_more_df)}")
    print(f"Ratio less significant (pvalue_ratio_less_fdr < 0.05): {len(ratio_less_df)}")
    print(f"Non-syn more significant (pvalue_nonsyn_more_fdr < 0.05): {len(nonsyn_more_df)}")
    print(f"Non-syn less significant (pvalue_nonsyn_less_fdr < 0.05): {len(nonsyn_less_df)}")
    
    if not has_effect_fdr:
        print("Warning: Effect FDR columns not found in DataFrame")
    if not has_ratio_fdr:
        print("Warning: Ratio FDR columns not found in DataFrame")

def generate_summary_tables(sample_dfs: Dict[str, pd.DataFrame], output_dir: str, cache_file: str = None, gene_strands: Dict[str, str] = None) -> None:
    """Generate comprehensive summary tables of significant genes."""
    os.makedirs(os.path.join(output_dir, "summary_tables"), exist_ok=True)
    summary_dir = os.path.join(output_dir, "summary_tables")
    
    # Load gene info cache
    gene_info_cache = load_gene_info_cache(cache_file) if cache_file else {}
    
    # Only consider treated samples (exclude controls, e.g., those with 'NT' in the name)
    treated_sample_dfs = {sample: df for sample, df in sample_dfs.items() if 'NT' not in sample}

    multi_sample_deleterious = {}
    multi_sample_neutral = {}
    # multi_sample_ratio_high = {}  # COMMENTED OUT FOR NOW
    # multi_sample_ratio_low = {}   # COMMENTED OUT FOR NOW

    for sample, df in treated_sample_dfs.items():
        for gene_id in df[(df['pvalue_more_fdr'] < 0.05)]['gene_id']:
            multi_sample_deleterious.setdefault(gene_id, []).append(sample)
        for gene_id in df[(df['pvalue_less_fdr'] < 0.05)]['gene_id']:
            multi_sample_neutral.setdefault(gene_id, []).append(sample)
        # for gene_id in df[df['pvalue_ratio_more_fdr'] < 0.05]['gene_id']:  # COMMENTED OUT FOR NOW
        #     multi_sample_ratio_high.setdefault(gene_id, []).append(sample)
        # for gene_id in df[df['pvalue_ratio_less_fdr'] < 0.05]['gene_id']:  # COMMENTED OUT FOR NOW
        #     multi_sample_ratio_low.setdefault(gene_id, []).append(sample)

    def get_gene_metrics(gene_id, samples, sample_dfs, value_fields):
        """Helper to get average metrics for a gene across samples."""
        values = {field: [] for field in value_fields}
        total_mutations = []
        for sample in samples:
            df = sample_dfs[sample]
            row = df[df['gene_id'] == gene_id]
            if not row.empty:
                for field in value_fields:
                    val = row.iloc[0][field]
                    if pd.notnull(val):
                        values[field].append(val)
                # Collect total_mutations for this gene/sample
                tm = row.iloc[0].get('total_mutations', None)
                if pd.notnull(tm):
                    total_mutations.append(tm)
        # Compute averages
        avg = {f'avg_{field}': np.mean(values[field]) if values[field] else np.nan for field in value_fields}
        avg['avg_total_mutations'] = np.mean(total_mutations) if total_mutations else np.nan
        return avg

    def write_multi_sample_table(gene_dict, filename, gene_info_cache, sample_dfs, value_fields):
        rows = []
        for gene_id, samples in gene_dict.items():
            row = {
                'gene_id': gene_id,
                'num_samples': len(samples),
            }
            if gene_info_cache and gene_id in gene_info_cache:
                row['gene_name'] = gene_info_cache[gene_id].get('name', '')
                row['gene_description'] = gene_info_cache[gene_id].get('description', '')
            # Add strand information if available
            if gene_strands and gene_id in gene_strands:
                row['strand'] = gene_strands[gene_id]
            avg_metrics = get_gene_metrics(gene_id, samples, sample_dfs, value_fields)
            row.update(avg_metrics)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values('num_samples', ascending=False)
            df.to_csv(os.path.join(summary_dir, filename), index=False)
            print(f"Multi-sample summary table written: {filename} ({len(df)} genes)")

    # Write tables with appropriate value fields, using only treated_sample_dfs
    write_multi_sample_table(
        multi_sample_deleterious, "multi_sample_deleterious_effect.csv", gene_info_cache, treated_sample_dfs,
        value_fields=['effect_score', 'pvalue_more_fdr']
    )
    write_multi_sample_table(
        multi_sample_neutral, "multi_sample_neutral_effect.csv", gene_info_cache, treated_sample_dfs,
        value_fields=['effect_score', 'pvalue_less_fdr']
    )
    # write_multi_sample_table(  # COMMENTED OUT FOR NOW
    #     multi_sample_ratio_high, "multi_sample_high_ratio.csv", gene_info_cache, treated_sample_dfs,
    #     value_fields=['deleterious_ratio', 'pvalue_ratio_more_fdr']
    # )
    # write_multi_sample_table(  # COMMENTED OUT FOR NOW
    #     multi_sample_ratio_low, "multi_sample_low_ratio.csv", gene_info_cache, treated_sample_dfs,
    #     value_fields=['deleterious_ratio', 'pvalue_ratio_less_fdr']
    # )

def create_pvalue_distribution_plots(sample_dfs: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Create multi-panel distribution plots for p-values."""
    # Create output directory
    dist_plot_dir = os.path.join(output_dir, "distribution_plots")
    os.makedirs(dist_plot_dir, exist_ok=True)
    
    # Collect p-values from all samples
    control_pvals = {
        'effect_more': [],
        'effect_less': [],
        'effect_more_fdr': [],
        'effect_less_fdr': [],
        'ratio_more': [],
        'ratio_less': [],
        'ratio_more_fdr': [],
        'ratio_less_fdr': []
    }
    
    treated_pvals = {
        'effect_more': [],
        'effect_less': [],
        'effect_more_fdr': [],
        'effect_less_fdr': [],
        'ratio_more': [],
        'ratio_less': [],
        'ratio_more_fdr': [],
        'ratio_less_fdr': []
    }
    
    # Collect p-values from all samples
    for sample, df in sample_dfs.items():
        target_dict = control_pvals if 'NT' in sample else treated_pvals
        
        # Effect p-values: collect independently of ratio columns
        if 'pvalue_more' in df.columns:
            target_dict['effect_more'].extend(df['pvalue_more'].dropna().values)
        if 'pvalue_less' in df.columns:
            target_dict['effect_less'].extend(df['pvalue_less'].dropna().values)
        if 'pvalue_more_fdr' in df.columns:
            target_dict['effect_more_fdr'].extend(df['pvalue_more_fdr'].dropna().values)
        if 'pvalue_less_fdr' in df.columns:
            target_dict['effect_less_fdr'].extend(df['pvalue_less_fdr'].dropna().values)
        
        # Ratio p-values: only if present (these may be absent or all NaN)
        if 'pvalue_ratio_more' in df.columns:
            target_dict['ratio_more'].extend(df['pvalue_ratio_more'].dropna().values)
        if 'pvalue_ratio_less' in df.columns:
            target_dict['ratio_less'].extend(df['pvalue_ratio_less'].dropna().values)
        if 'pvalue_ratio_more_fdr' in df.columns:
            target_dict['ratio_more_fdr'].extend(df['pvalue_ratio_more_fdr'].dropna().values)
        if 'pvalue_ratio_less_fdr' in df.columns:
            target_dict['ratio_less_fdr'].extend(df['pvalue_ratio_less_fdr'].dropna().values)
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    bins = np.linspace(0, 1, 51)  # 50 bins from 0 to 1
    
    # Plot titles and data
    plot_configs = [
        # Row 1: Raw p-values
        {'ax': axes[0, 0], 'data': [control_pvals['effect_more'], treated_pvals['effect_more']], 
         'title': 'Effect Score (More Deleterious)'},
        {'ax': axes[0, 1], 'data': [control_pvals['effect_less'], treated_pvals['effect_less']], 
         'title': 'Effect Score (Less Deleterious)'},
        {'ax': axes[0, 2], 'data': [control_pvals['ratio_more'], treated_pvals['ratio_more']], 
         'title': 'Deleterious Ratio (More)'},
        {'ax': axes[0, 3], 'data': [control_pvals['ratio_less'], treated_pvals['ratio_less']], 
         'title': 'Deleterious Ratio (Less)'},
        
        # Row 2: FDR corrected p-values
        {'ax': axes[1, 0], 'data': [control_pvals['effect_more_fdr'], treated_pvals['effect_more_fdr']], 
         'title': 'Effect Score (More) - FDR Corrected'},
        {'ax': axes[1, 1], 'data': [control_pvals['effect_less_fdr'], treated_pvals['effect_less_fdr']], 
         'title': 'Effect Score (Less) - FDR Corrected'},
        {'ax': axes[1, 2], 'data': [control_pvals['ratio_more_fdr'], treated_pvals['ratio_more_fdr']], 
         'title': 'Deleterious Ratio (More) - FDR Corrected'},
        {'ax': axes[1, 3], 'data': [control_pvals['ratio_less_fdr'], treated_pvals['ratio_less_fdr']], 
         'title': 'Deleterious Ratio (Less) - FDR Corrected'}
    ]
    
    # Create each subplot
    for config in plot_configs:
        ax = config['ax']
        ax.hist(config['data'], bins=bins, 
                label=['Control', 'EMS Treated'],
                stacked=True, alpha=0.7,
                color=['lightgrey', 'darkorange'])
        
        # Add significance threshold line
        ax.axvline(x=0.05, color='r', linestyle='--', alpha=0.5)
        
        ax.set_title(config['title'])
        ax.set_xlabel('p-value')
        ax.set_ylabel('Count')
    
    # Add a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle('P-value Distributions Across Samples', fontsize=16)
    
    # Save the figure
    plt.savefig(os.path.join(dist_plot_dir, 'pvalue_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create individual plots for each p-value type
    for pval_type in ['effect_more', 'effect_less', 'ratio_more', 'ratio_less']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Raw p-values
        ax1.hist([control_pvals[pval_type], treated_pvals[pval_type]], 
                bins=bins, label=['Control', 'EMS Treated'],
                stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
        ax1.axvline(x=0.05, color='r', linestyle='--', alpha=0.5)
        ax1.set_title(f'{pval_type.replace("_", " ").title()} - Raw')
        ax1.set_xlabel('p-value')
        ax1.set_ylabel('Count')
        
        # FDR corrected p-values
        ax2.hist([control_pvals[f'{pval_type}_fdr'], treated_pvals[f'{pval_type}_fdr']], 
                bins=bins, label=['Control', 'EMS Treated'],
                stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
        ax2.axvline(x=0.05, color='r', linestyle='--', alpha=0.5)
        ax2.set_title(f'{pval_type.replace("_", " ").title()} - FDR Corrected')
        ax2.set_xlabel('p-value')
        ax2.set_ylabel('Count')
        
        # Add legend
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(dist_plot_dir, f'{pval_type}_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"P-value distribution plots saved to: {dist_plot_dir}")

# --- New diagnostics: genome-wide non-syn fraction and codon-position distributions ---

def compute_global_nonsyn_fractions(results_dir: str, output_dir: str) -> None:
    """Compute genome-wide non-synonymous fraction for observed vs permuted (controls and treated).
    Robustly finds permuted result files regardless of exact naming and saves summary + histograms.
    """
    results_path = Path(results_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def pad_and_add(acc: List[float], arr: List[float]) -> List[float]:
        a = np.array(acc, dtype=float)
        b = np.array(arr, dtype=float)
        if a.size == 0:
            return b.tolist()
        if b.size == 0:
            return a.tolist()
        if a.size < b.size:
            a = np.pad(a, (0, b.size - a.size))
        elif b.size < a.size:
            b = np.pad(b, (0, a.size - b.size))
        return (a + b).tolist()

    # Find candidate permuted result files
    candidates = list(results_path.glob('permuted_provean*/*.json'))  # if placed in subdir
    candidates += list(results_path.glob('permuted_provean*.json'))
    # Fallback to known names
    candidates += [results_path / 'permuted_provean_results_controls.json',
                   results_path / 'permuted_provean_results_treated.json']
    candidates = [p for p in candidates if p.exists()]

    # Group files into controls/treated by filename
    control_files = []
    treated_files = []
    for p in candidates:
        name = p.name.lower()
        if any(tag in name for tag in ['control', 'controls', 'nt']):
            control_files.append(p)
        if any(tag in name for tag in ['treated', 'ems']):
            treated_files.append(p)

    def load_and_merge(files: List[Path]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for p in files:
            with open(p) as f:
                data = json.load(f)
            for gene_id, vals in data.items():
                if gene_id not in merged:
                    merged[gene_id] = vals
                else:
                    # Merge mutation_stats
                    ms_a = merged[gene_id].get('mutation_stats', {})
                    ms_b = vals.get('mutation_stats', {})
                    for k in ['non_syn_muts', 'syn_muts', 'total_mutations']:
                        ms_a[k] = (ms_a.get(k, 0) or 0) + (ms_b.get(k, 0) or 0)
                    merged[gene_id]['mutation_stats'] = ms_a
                    # Merge permutation arrays (sum element-wise with padding)
                    pa = merged[gene_id].get('permutation', {})
                    pb = vals.get('permutation', {})
                    for k in ['non_syn_counts', 'total_mutations']:
                        a = pa.get(k, [])
                        b = pb.get(k, [])
                        pa[k] = pad_and_add(a, b)
                    merged[gene_id]['permutation'] = pa
        return merged

    controls = load_and_merge(control_files) if control_files else {}
    treated = load_and_merge(treated_files) if treated_files else {}

    def summarize(group: Dict[str, Any]) -> Dict[str, Any]:
        if not group:
            return {}
        obs_non_syn = 0
        obs_total = 0
        perm_non_syn_sums: List[float] = []
        perm_total_sums: List[float] = []
        for _, vals in group.items():
            ms = vals.get('mutation_stats', {})
            obs_non_syn += ms.get('non_syn_muts', 0) or 0
            if 'syn_muts' in ms:
                obs_total += (ms.get('non_syn_muts', 0) or 0) + (ms.get('syn_muts', 0) or 0)
            else:
                obs_total += ms.get('total_mutations', 0) or 0
            perm_ns = vals.get('permutation', {}).get('non_syn_counts', [])
            perm_tm = vals.get('permutation', {}).get('total_mutations', [])
            perm_non_syn_sums = pad_and_add(perm_non_syn_sums, perm_ns)
            perm_total_sums = pad_and_add(perm_total_sums, perm_tm)
        obs_frac = (obs_non_syn / obs_total) if obs_total else np.nan
        perm_frac = None
        if len(perm_total_sums) and np.any(np.array(perm_total_sums) > 0):
            denom = np.array(perm_total_sums, dtype=float)
            numer = np.array(perm_non_syn_sums, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                pf = np.where(denom > 0, numer / denom, np.nan)
            perm_frac = pf
        return {
            'observed_fraction': obs_frac,
            'permuted_fractions': perm_frac.tolist() if isinstance(perm_frac, np.ndarray) else []
        }

    ctrl_summary = summarize(controls)
    treat_summary = summarize(treated)

    # Save summary CSV and simple hist plots
    summ_rows = []
    for label, summ in [('controls', ctrl_summary), ('treated', treat_summary)]:
        if not summ:
            continue
        pf = np.array(summ.get('permuted_fractions', []), dtype=float)
        pf = pf[~np.isnan(pf)] if pf.size else pf
        row = {
            'group': label,
            'observed_fraction': summ.get('observed_fraction', np.nan),
            'permuted_mean': float(np.nanmean(pf)) if pf.size else np.nan,
            'permuted_sd': float(np.nanstd(pf)) if pf.size else np.nan,
            'permuted_q05': float(np.nanpercentile(pf, 5)) if pf.size else np.nan,
            'permuted_q50': float(np.nanpercentile(pf, 50)) if pf.size else np.nan,
            'permuted_q95': float(np.nanpercentile(pf, 95)) if pf.size else np.nan,
        }
        summ_rows.append(row)
        if pf.size:
            plt.figure(figsize=(6, 4))
            plt.hist(pf, bins=40, color='lightgrey', edgecolor='k')
            if not np.isnan(row['observed_fraction']):
                plt.axvline(row['observed_fraction'], color='red', linestyle='--', label='Observed')
            plt.xlabel('Genome-wide non-syn fraction')
            plt.ylabel('Count')
            plt.title(f'Permuted vs Observed non-syn fraction ({label})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'nonsyn_fraction_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()

    if summ_rows:
        pd.DataFrame(summ_rows).to_csv(Path(output_dir) / 'nonsyn_fraction_summary.csv', index=False)
        print(f"Non-syn fraction summary written: {Path(output_dir) / 'nonsyn_fraction_summary.csv'}")

def plot_codon_position_distributions(results_dir: str, config: Dict[str, Any], output_dir: str) -> None:
    """Plot codon-position distributions (1/2/3) for observed vs expected-null (controls and treated)."""
    results_path = Path(results_dir)
    out_dir = Path(output_dir)
    
    # Derive nuc_muts directory (sibling of results)
    nuc_muts_dir = results_path.parent / 'nuc_muts'
    if not nuc_muts_dir.exists():
        print(f"Warning: nuc_muts directory not found at {nuc_muts_dir}; skipping codon-position plot")
        return
    
    # Initialize genome context
    wolgenome = parse.SeqContext(config['references']['genomic_fna'], config['references']['annotation'])
    features = wolgenome.genome_features()
    
    def observed_codon_pos_counts(json_files: List[Path]) -> np.ndarray:
        counts = np.zeros(3, dtype=int)
        for js in json_files:
            with open(js) as jf:
                mut_dict = json.load(jf)
                for gene_id, gene_data in mut_dict.items():
                    if gene_id not in features:
                        continue
                    for mut_key in gene_data.get('mutations', {}).keys():
                        pos_str, mut_type = mut_key.split('_')
                        ref_base = mut_type.split('>')[0]
                        if ref_base not in ('C', 'G'):
                            continue
                        # Observed JSONs encode gene_pos as 1-based
                        gene_pos_1b = int(pos_str)
                        codon_pos = (gene_pos_1b - 1) % 3  # 0,1,2
                        counts[codon_pos] += 1
        return counts
    
    def expected_codon_pos_counts(json_files: List[Path]) -> np.ndarray:
        # Build observed 5mer counts for these files (C/G-centered only)
        kmer_counts: Dict[str, int] = {}
        flank = 2
        genome_seq = str(wolgenome.genome).upper()
        for js in json_files:
            with open(js) as jf:
                mut_dict = json.load(jf)
                for gene_id, gene_data in mut_dict.items():
                    if gene_id not in features:
                        continue
                    feat = features[gene_id]
                    for mut_key in gene_data.get('mutations', {}).keys():
                        pos_str, mut_type = mut_key.split('_')
                        ref_base = mut_type.split('>')[0]
                        if ref_base not in ('C', 'G'):
                            continue
                        gene_pos_1b = int(pos_str)
                        # Map to genome position (0-based)
                        if feat.strand == 1:
                            genome_pos = feat.start + gene_pos_1b - 1
                        else:
                            genome_pos = feat.end - gene_pos_1b
                        if genome_pos - flank < 0 or genome_pos + flank >= len(genome_seq):
                            continue
                        kmer = genome_seq[genome_pos - flank: genome_pos + flank + 1]
                        if len(kmer) == 5:
                            # For reverse strand, normalize to coding direction (reverse complement)
                            if feat.strand == -1:
                                from Bio.Seq import Seq
                                kmer = str(Seq(kmer).reverse_complement())
                            if kmer[2] in ('C', 'G'):
                                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
        # Build site pool once and compute codon-pos proportions per 5-mer
        pool: Dict[str, List[tuple]] = {}
        flank = 2
        for gene_id, feat in features.items():
            gene_start = feat.start
            gene_end = feat.end
            strand = feat.strand
            gene_len = gene_end - gene_start
            for i in range(gene_len):
                if strand == 1:
                    genome_pos = gene_start + i
                    gene_pos = i + 1
                else:
                    genome_pos = gene_end - 1 - i
                    gene_pos = gene_end - genome_pos
                if genome_pos - flank < 0 or genome_pos + flank >= len(genome_seq):
                    continue
                kmer = genome_seq[genome_pos - flank: genome_pos + flank + 1]
                if len(kmer) != 5:
                    continue
                # Normalize to coding direction for keying
                if strand == -1:
                    from Bio.Seq import Seq
                    kmer_key = str(Seq(kmer).reverse_complement())
                else:
                    kmer_key = kmer
                if kmer_key[2] not in ('C', 'G'):
                    continue
                pool.setdefault(kmer_key, []).append((gene_id, gene_pos, genome_pos, strand))
        
        # For each 5-mer, compute codon-position distribution among its sites
        expected = np.zeros(3, dtype=float)
        for kmer, count in kmer_counts.items():
            sites = pool.get(kmer, [])
            if not sites:
                continue
            codon_bins = np.zeros(3, dtype=float)
            for _, gene_pos, _, _ in sites:
                codon_bins[(gene_pos - 1) % 3] += 1
            codon_props = codon_bins / codon_bins.sum() if codon_bins.sum() > 0 else codon_bins
            expected += count * codon_props
        return expected
    
    # Partition JSONs into groups
    json_files = list(nuc_muts_dir.glob('*.json'))
    control_files = [p for p in json_files if 'EMS' not in p.stem]
    treated_files = [p for p in json_files if 'EMS' in p.stem and '7d' not in p.stem]
    
    for label, files in [('controls', control_files), ('treated', treated_files)]:
        if not files:
            print(f"No nuc mutation JSONs for {label}; skipping codon-position plot")
            continue
        obs = observed_codon_pos_counts(files).astype(float)
        exp = expected_codon_pos_counts(files).astype(float)
        # Normalize to fractions
        obs_frac = obs / obs.sum() if obs.sum() else np.array([np.nan, np.nan, np.nan])
        exp_frac = exp / exp.sum() if exp.sum() else np.array([np.nan, np.nan, np.nan])
        # Plot side-by-side bars
        plt.figure(figsize=(6,4))
        indices = np.arange(3)
        width = 0.35
        plt.bar(indices - width/2, obs_frac, width=width, color='darkorange', label='Observed')
        plt.bar(indices + width/2, exp_frac, width=width, color='lightgrey', label='Expected (null)')
        plt.xticks(indices, ['Codon1', 'Codon2', 'Codon3'])
        plt.ylabel('Fraction of C/G-centered mutations')
        plt.title(f'Codon-position distribution ({label})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'codon_position_distribution_{label}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Codon-position distribution plot saved for {label}")

def load_gene_lengths_and_strands(gff_path: str) -> tuple[Dict[str, int], Dict[str, str]]:
    """Load gene lengths and strands from GFF file into lookup dictionaries."""
    gene_lengths = {}
    gene_strands = {}
    try:
        with open(gff_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] != 'gene':
                    continue
                    
                # Calculate gene length
                start = int(parts[3])
                end = int(parts[4])
                length = end - start + 1
                
                # Extract strand information (7th column)
                strand = parts[6]
                
                # Extract gene ID from attributes
                attrs = dict(attr.split('=') for attr in parts[8].split(';') if '=' in attr)
                dbxref = attrs.get('Dbxref', '')
                if 'GeneID:' in dbxref:
                    gene_id = dbxref.split('GeneID:')[1]
                    gene_lengths[gene_id] = length
                    gene_strands[gene_id] = strand
                    
        return gene_lengths, gene_strands
    except Exception as e:
        print(f"Error loading GFF file: {e}")
        return {}, {}

def load_gene_lengths(gff_path: str) -> Dict[str, int]:
    """Load gene lengths from GFF file into a lookup dictionary."""
    gene_lengths, _ = load_gene_lengths_and_strands(gff_path)
    return gene_lengths



def load_module_assignments(module_file: str) -> pd.DataFrame:
    """Load gene module assignments from transcriptomic data."""
    return pd.read_csv(module_file, sep='\t')

def load_expression_data(expression_file: str) -> pd.DataFrame:
    """Load expression data from DESeq2 results file."""
    return pd.read_csv(expression_file, sep='\t')

def plot_effect_coverage_by_module(df: pd.DataFrame, module_df: pd.DataFrame, output_dir: str, wd_to_gid_map: Dict[str, str], gene_lengths: Dict[str, int]) -> None:
    """Create scatter plots of effect scores and deleterious ratios against gene length, colored by module."""
    # Create reverse mapping (GID to WD)
    gid_to_wd_map = {v: k for k, v in wd_to_gid_map.items()}
    
    # Add WD IDs to the effect scores DataFrame
    df['wd_id'] = df['gene_id'].map(gid_to_wd_map)
    
    # Add gene lengths
    df['gene_length'] = df['gene_id'].map(gene_lengths)
    
    # Merge effect scores with module assignments using WD IDs
    merged_df = pd.merge(df, module_df, left_on='wd_id', right_on='gene', how='inner')
    
    if merged_df.empty:
        return
    
    # Drop any rows with NaN values in the columns we'll plot
    plot_df = merged_df.dropna(subset=['gene_length', 'effect_score', 'deleterious_ratio', 'module'])
    
    if plot_df.empty:
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Effect Score vs Gene Length
    for module in sorted(plot_df['module'].unique()):
        module_data = plot_df[plot_df['module'] == module]
        if not module_data.empty:
            ax1.scatter(module_data['gene_length'].astype(float), 
                       module_data['effect_score'].astype(float),
                       label=f'Module {module}',
                       alpha=0.6)
    
    ax1.set_xlabel('Gene Length (bp)')
    ax1.set_ylabel('Effect Score')
    ax1.set_title('Effect Score vs Gene Length by Module')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')  # Log scale for gene length
    
    # Plot 2: Deleterious Ratio vs Gene Length
    for module in sorted(plot_df['module'].unique()):
        module_data = plot_df[plot_df['module'] == module]
        if not module_data.empty:
            ax2.scatter(module_data['gene_length'].astype(float), 
                       module_data['deleterious_ratio'].astype(float),
                       label=f'Module {module}',
                       alpha=0.6)
    
    ax2.set_xlabel('Gene Length (bp)')
    ax2.set_ylabel('Deleterious Ratio')
    ax2.set_title('Deleterious Ratio vs Gene Length by Module')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')  # Log scale for gene length
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effect_coverage_by_module.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_effect_by_expression(df: pd.DataFrame, expression_df: pd.DataFrame, output_dir: str, wd_to_gid_map: Dict[str, str]) -> None:
    """Create scatter plots of effect scores and deleterious ratios against log2FoldChange, colored by significance."""
    # Create reverse mapping (GID to WD)
    gid_to_wd_map = {v: k for k, v in wd_to_gid_map.items()}
    
    # Add WD IDs to the effect scores DataFrame
    df['wd_id'] = df['gene_id'].map(gid_to_wd_map)
    
    # Merge effect scores with expression data using WD IDs
    merged_df = pd.merge(df, expression_df, left_on='wd_id', right_on='gene', how='inner')
    
    if merged_df.empty:
        return
    
    # Drop any rows with NaN values in the columns we'll plot
    plot_df = merged_df.dropna(subset=['log2FoldChange', 'effect_norm', 'deleterious_ratio', 'padj'])
    
    if plot_df.empty:
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define significance
    significant = plot_df['padj'] < 0.05
    nonsig = ~significant
    
    # Plot 1: Normalized Effect Score vs log2FoldChange
    ax1.scatter(plot_df[nonsig]['baseMean'], 
                plot_df[nonsig]['effect_norm'],
                color='grey', alpha=0.3, label='Non-significant')
    ax1.scatter(plot_df[significant]['baseMean'], 
                plot_df[significant]['effect_norm'],
                color='red', alpha=0.6, label='Significant (padj < 0.05)')
    
    ax1.set_xlabel('baseMean')
    ax1.set_ylabel('Length-Normalized Effect Score')
    ax1.set_title('Normalized Effect Score vs Expression Change')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Deleterious Ratio vs log2FoldChange
    ax2.scatter(plot_df[nonsig]['baseMean'], 
                plot_df[nonsig]['deleterious_ratio'],
                color='grey', alpha=0.3, label='Non-significant')
    ax2.scatter(plot_df[significant]['baseMean'], 
                plot_df[significant]['deleterious_ratio'],
                color='red', alpha=0.6, label='Significant (padj < 0.05)')
    
    ax2.set_xlabel('baseMean')
    ax2.set_ylabel('Deleterious Ratio')
    ax2.set_title('Deleterious Ratio vs Expression Change')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effect_by_expression.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_expression_vs_mutations(df: pd.DataFrame, expression_df: pd.DataFrame, output_dir: str, wd_to_gid_map: Dict[str, str], gene_lengths: Dict[str, int]) -> None:
    """Create scatter plots comparing baseMean expression to total mutations normalized by gene length."""
    # Create reverse mapping (GID to WD)
    gid_to_wd_map = {v: k for k, v in wd_to_gid_map.items()}
    
    # Add WD IDs to the effect scores DataFrame
    df['wd_id'] = df['gene_id'].map(gid_to_wd_map)
    
    # Add gene lengths
    df['gene_length'] = df['gene_id'].map(gene_lengths)
    
    # Merge effect scores with expression data using WD IDs
    merged_df = pd.merge(df, expression_df, left_on='wd_id', right_on='gene', how='inner')
    
    if merged_df.empty:
        return
    
    # Drop any rows with NaN values in the columns we'll plot
    plot_df = merged_df.dropna(subset=['baseMean', 'total_mutations', 'gene_length', 'padj'])
    
    if plot_df.empty:
        return
    
    # Calculate mutations per base pair
    plot_df['mutations_per_bp'] = plot_df['total_mutations'] / plot_df['gene_length']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define significance
    # Print one row of plot_df to inspect the data
    print("\nExample row from plot_df:")
    print(plot_df.iloc[0].to_string())
    print()  
    
    significant = plot_df['padj'] < 0.05
    nonsig = ~significant
    
    # Plot 1: baseMean vs Total Mutations
    ax1.scatter(plot_df[nonsig]['baseMean'], 
                plot_df[nonsig]['total_mutations'],
                color='grey', alpha=0.3, label='Non-significant')
    ax1.scatter(plot_df[significant]['baseMean'], 
                plot_df[significant]['total_mutations'],
                color='red', alpha=0.6, label='Significant (padj < 0.05)')
    
    ax1.set_xlabel('baseMean Expression')
    ax1.set_ylabel('Total Mutations')
    ax1.set_title('Expression vs Total Mutations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')  # Log scale for baseMean
    
    # Plot 2: baseMean vs Mutations per bp
    ax2.scatter(plot_df[nonsig]['baseMean'], 
                plot_df[nonsig]['mutations_per_bp'],
                color='grey', alpha=0.3, label='Non-significant')
    ax2.scatter(plot_df[significant]['baseMean'], 
                plot_df[significant]['mutations_per_bp'],
                color='red', alpha=0.6, label='Significant (padj < 0.05)')
    
    ax2.set_xlabel('baseMean Expression')
    ax2.set_ylabel('Mutations per bp')
    ax2.set_title('Expression vs Mutations per bp')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')  # Log scale for baseMean
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expression_vs_mutations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def perform_dn_ds_analysis(results_dir: str, output_dir: str) -> None:
    """Create a new analysis folder with per-gene observed vs permutation-average dN/dS.

    Reads pooled JSONs written by the PROVEAN pipeline:
    - permuted_provean_results_controls.json
    - permuted_provean_results_treated.json

    For each gene and group (controls/treated) it computes:
    - Observed counts: non_syn, syn, total
    - Observed dN/dS = non_syn / syn
    - Permutation averages: mean(non_syn_counts), mean(syn_counts), mean(total_mutations)
    - Permutation dN/dS summary: mean, median, sd, q05, q50, q95 over ratios per permutation

    Saves per-group CSVs and generates comparison plots.
    """
    results_path = Path(results_dir)
    out_dir = Path(output_dir) / 'dn_ds_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_group(name: str) -> dict:
        # Try exact filename first
        exact = results_path / f'permuted_provean_results_{name}.json'
        if exact.exists():
            with open(exact) as f:
                print(f"Using pooled file: {exact}")
                return json.load(f)
        # Tag-based fallback within results_path
        tags = ['treated', 'ems'] if name == 'treated' else ['control', 'controls', 'nt']
        files = list(results_path.glob('*.json'))
        candidates = [p for p in files if any(t in p.name.lower() for t in tags)]
        if candidates:
            preferred = [p for p in candidates if p.name.lower().startswith('permuted_provean')]
            fp = preferred[0] if preferred else candidates[0]
            with open(fp) as f:
                print(f"Using pooled file (tag match): {fp}")
                return json.load(f)
        print(f"Warning: pooled results not found for {name} in {results_path}")
        return None

    def summarize_group(group_data: dict, label: str) -> pd.DataFrame:
        rows = []
        for gene_id, vals in group_data.items():
            ms = vals.get('mutation_stats', {})
            obs_non_syn = ms.get('non_syn_muts', np.nan)
            obs_syn = ms.get('syn_muts', np.nan)
            # Prefer explicit syn_muts; if missing, leave NaN to avoid miscount
            with np.errstate(divide='ignore', invalid='ignore'):
                obs_dn_ds = (obs_non_syn / obs_syn) if (pd.notnull(obs_non_syn) and pd.notnull(obs_syn) and obs_syn not in (0, 0.0)) else np.nan

            perm = vals.get('permutation', {})
            perm_non = np.array(perm.get('non_syn_counts', []), dtype=float)
            perm_tot = np.array(perm.get('total_mutations', []), dtype=float)
            # Derive perm_syn from syn_counts if present; otherwise use total - non_syn
            if 'syn_counts' in perm:
                perm_syn = np.array(perm.get('syn_counts', []), dtype=float)
            else:
                perm_syn = perm_tot - perm_non if perm_non.size and perm_tot.size else np.array([])

            valid_ratio_mask = (perm_syn > 0)
            perm_dn_ds_vec = np.divide(perm_non[valid_ratio_mask], perm_syn[valid_ratio_mask]) if perm_non.size and perm_syn.size else np.array([])

            perm_mean_non = float(np.nanmean(perm_non)) if perm_non.size else np.nan
            perm_mean_syn = float(np.nanmean(perm_syn)) if perm_syn.size else np.nan
            perm_mean_tot = float(np.nanmean(perm_tot)) if perm_tot.size else np.nan

            def nq(a, q):
                return float(np.nanpercentile(a, q)) if a.size else np.nan

            row = {
                'group': label,
                'gene_id': gene_id,
                'obs_non_syn': obs_non_syn,
                'obs_syn': obs_syn,
                'obs_dn_ds': float(obs_dn_ds) if pd.notnull(obs_dn_ds) else np.nan,
                'perm_mean_non_syn': perm_mean_non,
                'perm_mean_syn': perm_mean_syn,
                'perm_mean_total_mut': perm_mean_tot,
                'perm_dn_ds_mean': float(np.nanmean(perm_dn_ds_vec)) if perm_dn_ds_vec.size else np.nan,
                'perm_dn_ds_median': float(np.nanmedian(perm_dn_ds_vec)) if perm_dn_ds_vec.size else np.nan,
                'perm_dn_ds_sd': float(np.nanstd(perm_dn_ds_vec)) if perm_dn_ds_vec.size else np.nan,
                'perm_dn_ds_q05': nq(perm_dn_ds_vec, 5),
                'perm_dn_ds_q50': nq(perm_dn_ds_vec, 50),
                'perm_dn_ds_q95': nq(perm_dn_ds_vec, 95),
                'n_perms_with_syn>0': int(valid_ratio_mask.sum()) if perm_syn.size else 0,
                'n_perms_total': int(len(perm_syn)) if perm_syn is not None else 0
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.sort_values(['group', 'gene_id'])
        return df

    controls = load_group('controls')
    treated = load_group('treated')

    ctrl_df = summarize_group(controls, 'controls') if controls else pd.DataFrame()
    treat_df = summarize_group(treated, 'treated') if treated else pd.DataFrame()

    if not ctrl_df.empty:
        ctrl_path = out_dir / 'gene_dn_ds_summary_controls.csv'
        ctrl_df.to_csv(ctrl_path, index=False)
        print(f"Wrote: {ctrl_path}")
    if not treat_df.empty:
        treat_path = out_dir / 'gene_dn_ds_summary_treated.csv'
        treat_df.to_csv(treat_path, index=False)
        print(f"Wrote: {treat_path}")

    # Combined comparison (treated vs controls)
    if not ctrl_df.empty and not treat_df.empty:
        merged = pd.merge(ctrl_df.add_prefix('ctrl_'), treat_df.add_prefix('treat_'), left_on='ctrl_gene_id', right_on='treat_gene_id', how='inner')
        # Simplify columns
        merged['gene_id'] = merged['ctrl_gene_id']
        merged['obs_dn_ds_delta_treat_minus_ctrl'] = merged['treat_obs_dn_ds'] - merged['ctrl_obs_dn_ds']
        merged['perm_dn_ds_mean_delta_treat_minus_ctrl'] = merged['treat_perm_dn_ds_mean'] - merged['ctrl_perm_dn_ds_mean']
        comp_cols = [
            'gene_id',
            'ctrl_obs_non_syn', 'ctrl_obs_syn', 'ctrl_obs_dn_ds',
            'treat_obs_non_syn', 'treat_obs_syn', 'treat_obs_dn_ds',
            'obs_dn_ds_delta_treat_minus_ctrl',
            'ctrl_perm_mean_non_syn', 'ctrl_perm_mean_syn', 'ctrl_perm_dn_ds_mean',
            'treat_perm_mean_non_syn', 'treat_perm_mean_syn', 'treat_perm_dn_ds_mean',
            'perm_dn_ds_mean_delta_treat_minus_ctrl'
        ]
        # Some columns may be missing if DF empty; filter by presence
        comp_cols = [c for c in comp_cols if c in merged.columns]
        comp_df = merged[comp_cols].copy()
        comp_path = out_dir / 'dn_ds_comparison_controls_vs_treated.csv'
        comp_df.to_csv(comp_path, index=False)
        print(f"Wrote: {comp_path}")

        # Plots
        plt.figure(figsize=(7,6))
        plt.scatter(treat_df['obs_dn_ds'], treat_df['perm_dn_ds_mean'], alpha=0.6, color='darkorange', label='Treated genes')
        lims = [0, np.nanmax([treat_df['obs_dn_ds'].max(), treat_df['perm_dn_ds_mean'].max()])]
        lims = [0, lims[1] if np.isfinite(lims[1]) and lims[1] > 0 else 1]
        plt.plot(lims, lims, 'k--', alpha=0.4, label='y=x')
        plt.xlabel('Observed dN/dS (treated)')
        plt.ylabel('Permutation mean dN/dS (treated)')
        plt.title('Observed vs permutation dN/dS per gene (treated)')
        plt.tight_layout()
        plt.savefig(out_dir / 'scatter_obs_vs_perm_dn_ds_treated.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(7,6))
        plt.scatter(ctrl_df['obs_dn_ds'], ctrl_df['perm_dn_ds_mean'], alpha=0.6, color='slategray', label='Control genes')
        lims = [0, np.nanmax([ctrl_df['obs_dn_ds'].max(), ctrl_df['perm_dn_ds_mean'].max()])]
        lims = [0, lims[1] if np.isfinite(lims[1]) and lims[1] > 0 else 1]
        plt.plot(lims, lims, 'k--', alpha=0.4, label='y=x')
        plt.xlabel('Observed dN/dS (controls)')
        plt.ylabel('Permutation mean dN/dS (controls)')
        plt.title('Observed vs permutation dN/dS per gene (controls)')
        plt.tight_layout()
        plt.savefig(out_dir / 'scatter_obs_vs_perm_dn_ds_controls.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Observed treated vs observed controls scatter
        plt.figure(figsize=(7,6))
        merged_simple = pd.merge(ctrl_df[['gene_id','obs_dn_ds']], treat_df[['gene_id','obs_dn_ds']], on='gene_id', suffixes=('_ctrl','_treat'))
        plt.scatter(merged_simple['obs_dn_ds_ctrl'], merged_simple['obs_dn_ds_treat'], alpha=0.6, color='teal')
        lim = np.nanmax([merged_simple['obs_dn_ds_ctrl'].max(), merged_simple['obs_dn_ds_treat'].max()])
        lim = lim if np.isfinite(lim) and lim > 0 else 1
        plt.plot([0, lim], [0, lim], 'k--', alpha=0.4)
        plt.xlabel('Observed dN/dS (controls)')
        plt.ylabel('Observed dN/dS (treated)')
        plt.title('Observed dN/dS per gene: treated vs controls')
        plt.tight_layout()
        plt.savefig(out_dir / 'scatter_obs_dn_ds_treated_vs_controls.png', dpi=300, bbox_inches='tight')
        plt.close()

    else:
        # If only one group available, still make its scatter against permutations
        df = treat_df if not treat_df.empty else ctrl_df
        label = 'treated' if not treat_df.empty else 'controls'
        if not df.empty:
            plt.figure(figsize=(7,6))
            plt.scatter(df['obs_dn_ds'], df['perm_dn_ds_mean'], alpha=0.6, color='darkorange' if label=='treated' else 'slategray')
            lims = [0, np.nanmax([df['obs_dn_ds'].max(), df['perm_dn_ds_mean'].max()])]
            lims = [0, lims[1] if np.isfinite(lims[1]) and lims[1] > 0 else 1]
            plt.plot(lims, lims, 'k--', alpha=0.4)
            plt.xlabel(f'Observed dN/dS ({label})')
            plt.ylabel(f'Permutation mean dN/dS ({label})')
            plt.title(f'Observed vs permutation dN/dS per gene ({label})')
            plt.tight_layout()
            plt.savefig(out_dir / f'scatter_obs_vs_perm_dn_ds_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()

def perform_simple_counts_analysis(results_dir: str, output_dir: str) -> None:
    """Generate simple tables and totals comparison as requested:
    - Per gene: observed syn and non-syn counts and permutation means.
    - Genome-wide: observed total AA-changing mutations vs distribution across permutations.
    """
    results_path = Path(results_dir)
    out_dir = Path(output_dir) / 'simple_counts_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_group(name: str) -> dict:
        # Try exact filename first
        exact = results_path / f'permuted_provean_results_{name}.json'
        if exact.exists():
            with open(exact) as f:
                print(f"Using pooled file: {exact}")
                return json.load(f)
        # Tag-based fallback
        tags = ['treated', 'ems'] if name == 'treated' else ['control', 'controls', 'nt']
        files = list(results_path.glob('*.json'))
        candidates = [p for p in files if any(t in p.name.lower() for t in tags)]
        if candidates:
            preferred = [p for p in candidates if p.name.lower().startswith('permuted_provean')]
            fp = preferred[0] if preferred else candidates[0]
            with open(fp) as f:
                print(f"Using pooled file (tag match): {fp}")
                return json.load(f)
        print(f"Warning: pooled results not found for {name} in {results_path}")
        return None

    def summarize_counts(group_data: dict, label: str) -> pd.DataFrame:
        rows = []
        for gene_id, vals in group_data.items():
            ms = vals.get('mutation_stats', {})
            obs_non = ms.get('non_syn_muts', np.nan)
            obs_syn = ms.get('syn_muts', np.nan)
            perm = vals.get('permutation', {})
            perm_non = np.array(perm.get('non_syn_counts', []), dtype=float)
            perm_tot = np.array(perm.get('total_mutations', []), dtype=float)
            if 'syn_counts' in perm:
                perm_syn = np.array(perm.get('syn_counts', []), dtype=float)
            else:
                perm_syn = perm_tot - perm_non if perm_non.size and perm_tot.size else np.array([])
            row = {
                'group': label,
                'gene_id': gene_id,
                'observed_syn': obs_syn,
                'observed_non_syn': obs_non,
                'perm_mean_syn': float(np.nanmean(perm_syn)) if perm_syn.size else np.nan,
                'perm_mean_non_syn': float(np.nanmean(perm_non)) if perm_non.size else np.nan
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.sort_values(['group', 'gene_id'])
        return df

    def pad_and_add(acc: np.ndarray, arr: np.ndarray) -> np.ndarray:
        if acc.size == 0:
            return arr.copy()
        if arr.size == 0:
            return acc
        if acc.size < arr.size:
            acc = np.pad(acc, (0, arr.size - acc.size))
        elif arr.size < acc.size:
            arr = np.pad(arr, (0, acc.size - arr.size))
        return acc + arr

    def summarize_totals_distribution(group_data: dict, label: str) -> None:
        # Observed genome-wide AA-changing total (align basis with syn/non per-site counts)
        obs_total = 0
        for _, vals in group_data.items():
            ms = vals.get('mutation_stats', {})
            syn = (ms.get('syn_muts', None))
            non = (ms.get('non_syn_muts', None))
            if syn is not None and non is not None:
                obs_total += (syn or 0) + (non or 0)
            else:
                # Fallback
                obs_total += (ms.get('total_mutations', 0) or 0)
        # Sum per-permutation totals across genes (use syn + non site-level counts)
        summed_perm = np.array([], dtype=float)
        for _, vals in group_data.items():
            perm = vals.get('permutation', {})
            perm_non = np.array(perm.get('non_syn_counts', []), dtype=float)
            perm_syn = np.array(perm.get('syn_counts', []), dtype=float)
            if not perm_syn.size:
                # Fallback if syn_counts not present
                perm_tot = np.array(perm.get('total_mutations', []), dtype=float)
                if perm_non.size and perm_tot.size:
                    perm_syn = perm_tot - perm_non
            if perm_non.size:
                per_gene_total = perm_non + (perm_syn if perm_syn.size else 0)
                summed_perm = pad_and_add(summed_perm, per_gene_total)
        # Write CSV of distribution
        if summed_perm.size:
            dist_df = pd.DataFrame({'perm_index': np.arange(1, len(summed_perm) + 1), 'total_mutations': summed_perm})
            csv_path = out_dir / f'genome_total_mutations_distribution_{label}.csv'
            dist_df.to_csv(csv_path, index=False)
            print(f"Wrote: {csv_path}")
            # Plot histogram with observed marker
            plt.figure(figsize=(7,5))
            plt.hist(summed_perm, bins=40, color='lightgrey', edgecolor='k')
            plt.axvline(obs_total, color='red', linestyle='--', label='Observed total')
            plt.xlabel('Genome-wide total AA-changing mutations per permutation')
            plt.ylabel('Count')
            plt.title(f'Total mutations distribution vs observed ({label})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'genome_total_mutations_distribution_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()
            # Quick summary CSV
            stats_row = {
                'group': label,
                'observed_total': obs_total,
                'perm_mean': float(np.nanmean(summed_perm)),
                'perm_sd': float(np.nanstd(summed_perm)),
                'perm_q05': float(np.nanpercentile(summed_perm, 5)),
                'perm_q50': float(np.nanpercentile(summed_perm, 50)),
                'perm_q95': float(np.nanpercentile(summed_perm, 95)),
                'n_permutations': int(len(summed_perm))
            }
            pd.DataFrame([stats_row]).to_csv(out_dir / f'genome_total_mutations_summary_{label}.csv', index=False)

    def summarize_nonsyn_distribution(group_data: dict, label: str) -> None:
        # Observed genome-wide non-synonymous total
        obs_non_syn = 0
        for _, vals in group_data.items():
            ms = vals.get('mutation_stats', {})
            obs_non_syn += (ms.get('non_syn_muts', 0) or 0)
        # Sum per-permutation non-syn counts across genes
        summed_perm_non = np.array([], dtype=float)
        for _, vals in group_data.items():
            perm_non = np.array(vals.get('permutation', {}).get('non_syn_counts', []), dtype=float)
            if perm_non.size:
                summed_perm_non = pad_and_add(summed_perm_non, perm_non)
        if summed_perm_non.size:
            dist_df = pd.DataFrame({'perm_index': np.arange(1, len(summed_perm_non) + 1), 'non_syn_counts': summed_perm_non})
            csv_path = out_dir / f'genome_nonsyn_counts_distribution_{label}.csv'
            dist_df.to_csv(csv_path, index=False)
            print(f"Wrote: {csv_path}")
            # Plot histogram with observed marker
            plt.figure(figsize=(7,5))
            plt.hist(summed_perm_non, bins=40, color='lightgrey', edgecolor='k')
            plt.axvline(obs_non_syn, color='red', linestyle='--', label='Observed non-syn total')
            plt.xlabel('Genome-wide non-synonymous mutations per permutation')
            plt.ylabel('Count')
            plt.title(f'Non-synonymous distribution vs observed ({label})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'genome_nonsyn_counts_distribution_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()
            # Summary CSV
            stats_row = {
                'group': label,
                'observed_nonsyn_total': obs_non_syn,
                'perm_mean': float(np.nanmean(summed_perm_non)),
                'perm_sd': float(np.nanstd(summed_perm_non)),
                'perm_q05': float(np.nanpercentile(summed_perm_non, 5)),
                'perm_q50': float(np.nanpercentile(summed_perm_non, 50)),
                'perm_q95': float(np.nanpercentile(summed_perm_non, 95)),
                'n_permutations': int(len(summed_perm_non))
            }
            pd.DataFrame([stats_row]).to_csv(out_dir / f'genome_nonsyn_counts_summary_{label}.csv', index=False)

    # Load groups
    controls = load_group('controls')
    treated = load_group('treated')

    if controls:
        ctrl_df = summarize_counts(controls, 'controls')
        ctrl_path = out_dir / 'simple_counts_controls.csv'
        ctrl_df.to_csv(ctrl_path, index=False)
        print(f"Wrote: {ctrl_path}")
        summarize_totals_distribution(controls, 'controls')
        summarize_nonsyn_distribution(controls, 'controls')
    if treated:
        treat_df = summarize_counts(treated, 'treated')
        treat_path = out_dir / 'simple_counts_treated.csv'
        treat_df.to_csv(treat_path, index=False)
        print(f"Wrote: {treat_path}")
        summarize_totals_distribution(treated, 'treated')
        summarize_nonsyn_distribution(treated, 'treated')

def verify_site_level_consistency(results_dir: str, output_dir: str) -> None:
	"""Compare genome-wide unique-site totals (from nuc_muts union) to results JSON observed totals and permutation totals.
	Writes a CSV per group and prints a short summary.
	"""
	results_path = Path(results_dir)
	out_dir = Path(output_dir) / 'consistency_checks'
	out_dir.mkdir(parents=True, exist_ok=True)
	
	# Locate nuc_muts sibling
	nuc_muts_dir = results_path.parent / 'nuc_muts'
	if not nuc_muts_dir.exists():
		print(f"Warning: nuc_muts directory not found at {nuc_muts_dir}; skipping consistency check")
		return
	
	def load_results_group(name: str) -> dict:
		fp = results_path / f'permuted_provean_results_{name}.json'
		if not fp.exists():
			return None
		with open(fp) as f:
			return json.load(f)
	
	def group_nuc_files(group: str) -> list[Path]:
		files = list(nuc_muts_dir.glob('*.json'))
		if group == 'controls':
			return [p for p in files if 'EMS' not in p.stem]
		else:
			return [p for p in files if 'EMS' in p.stem and '7d' not in p.stem]
	
	def compute_unique_site_union(json_files: list[Path]) -> int:
		seen = set()
		for js in json_files:
			with open(js) as f:
				mut_dict = json.load(f)
			for gene_id, gene_data in mut_dict.items():
				for mut_key in gene_data.get('mutations', {}).keys():
					pos_str, mut_type = mut_key.split('_')
					# Restrict to EMS-type C>T/G>A on C/G center to match null
					if mut_type not in ('C>T', 'G>A'):
						continue
					seen.add((gene_id, int(pos_str)))  # pos_str is 0-based
		return len(seen)
	
	def summarize_group(group_name: str):
		group_results = load_results_group(group_name)
		if not group_results:
			return None
		# Observed site-level total from results JSON (syn+non)
		obs_syn = 0
		obs_non = 0
		for _, vals in group_results.items():
			ms = vals.get('mutation_stats', {})
			obs_syn += (ms.get('syn_muts', 0) or 0)
			obs_non += (ms.get('non_syn_muts', 0) or 0)
		obs_total_results = obs_syn + obs_non
		# Permutation site-level totals per permutation = syn_counts + non_syn_counts
		perm_sum = None
		for _, vals in group_results.items():
			perm = vals.get('permutation', {})
			non = np.array(perm.get('non_syn_counts', []), dtype=float)
			syn = np.array(perm.get('syn_counts', []), dtype=float)
			if not syn.size:
				tot = np.array(perm.get('total_mutations', []), dtype=float)
				syn = (tot - non) if tot.size and non.size else np.array([])
			if non.size:
				totals = non + (syn if syn.size else 0)
				if perm_sum is None:
					perm_sum = totals.copy()
				else:
					# Pad shorter array then add
					if perm_sum.size < totals.size:
						perm_sum = np.pad(perm_sum, (0, totals.size - perm_sum.size))
					elif totals.size < perm_sum.size:
						totals = np.pad(totals, (0, perm_sum.size - totals.size))
					perm_sum = perm_sum + totals
		perm_mean = float(np.nanmean(perm_sum)) if isinstance(perm_sum, np.ndarray) else np.nan
		# Unique-site union from nuc_muts
		union_files = group_nuc_files(group_name)
		union_sites = compute_unique_site_union(union_files) if union_files else np.nan
		return {
			'group': group_name,
			'unique_sites_union_nuc_muts': union_sites,
			'observed_total_results_json': obs_total_results,
			'permutation_total_mean': perm_mean
		}
	
	rows = []
	for group in ['controls', 'treated']:
		s = summarize_group(group)
		if s:
			rows.append(s)
	if rows:
		df = pd.DataFrame(rows)
		csv_path = out_dir / 'site_level_consistency_summary.csv'
		df.to_csv(csv_path, index=False)
		print(f"Wrote: {csv_path}")

def compute_global_dn_ds(results_dir: str, output_dir: str) -> None:
    """Compute genome-wide dN/dS for observed vs permuted (controls and treated),
    writing summary CSV and histograms similar to non-syn fraction plots.
    """
    results_path = Path(results_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def pad_and_add(acc: List[float], arr: List[float]) -> List[float]:
        a = np.array(acc, dtype=float)
        b = np.array(arr, dtype=float)
        if a.size == 0:
            return b.tolist()
        if b.size == 0:
            return a.tolist()
        if a.size < b.size:
            a = np.pad(a, (0, b.size - a.size))
        elif b.size < a.size:
            b = np.pad(b, (0, a.size - b.size))
        return (a + b).tolist()

    # Find pooled results
    ctrl_fp = results_path / 'permuted_provean_results_controls.json'
    treat_fp = results_path / 'permuted_provean_results_treated.json'

    def load_group(fp: Path, group_name: str) -> Dict[str, Any]:
        if fp.exists():
            with open(fp) as f:
                print(f"Using pooled file: {fp}")
                return json.load(f)
        # Tag-based fallback
        tags = ['treated', 'ems'] if group_name == 'treated' else ['control', 'controls', 'nt']
        files = list(results_path.glob('*.json'))
        candidates = [p for p in files if any(t in p.name.lower() for t in tags)]
        if candidates:
            preferred = [p for p in candidates if p.name.lower().startswith('permuted_provean')]
            pick = preferred[0] if preferred else candidates[0]
            with open(pick) as f:
                print(f"Using pooled file (tag match): {pick}")
                return json.load(f)
        print(f"Warning: pooled results not found for {group_name} in {results_path}")
        return {}

    def summarize(group: Dict[str, Any]) -> Dict[str, Any]:
        if not group:
            return {}
        # Observed totals
        obs_non = 0
        obs_syn = 0
        # Permutation sums per-permutation index
        perm_non_sums: List[float] = []
        perm_syn_sums: List[float] = []
        perm_tot_sums: List[float] = []  # fallback if syn_counts missing
        for _, vals in group.items():
            ms = vals.get('mutation_stats', {})
            obs_non += (ms.get('non_syn_muts', 0) or 0)
            obs_syn += (ms.get('syn_muts', 0) or 0)
            perm = vals.get('permutation', {})
            perm_non = perm.get('non_syn_counts', []) or []
            perm_syn = perm.get('syn_counts', []) or []
            perm_tot = perm.get('total_mutations', []) or []
            perm_non_sums = pad_and_add(perm_non_sums, perm_non)
            # Prefer explicit syn_counts; else derive from total - non
            if perm_syn:
                perm_syn_sums = pad_and_add(perm_syn_sums, perm_syn)
            elif perm_tot:
                perm_tot_sums = pad_and_add(perm_tot_sums, perm_tot)
        # If syn not directly summed, derive from totals - non
        if (not perm_syn_sums) and perm_tot_sums:
            a = np.array(perm_tot_sums, dtype=float)
            b = np.array(perm_non_sums, dtype=float)
            if b.size < a.size:
                b = np.pad(b, (0, a.size - b.size))
            elif a.size < b.size:
                a = np.pad(a, (0, b.size - a.size))
            perm_syn_sums = (a - b).tolist()
        # Compute ratios
        obs_dn_ds = (obs_non / obs_syn) if obs_syn not in (0, 0.0) else np.nan
        perm_dn_ds = []
        if perm_non_sums and perm_syn_sums:
            non = np.array(perm_non_sums, dtype=float)
            syn = np.array(perm_syn_sums, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(syn > 0, non / syn, np.nan)
            perm_dn_ds = r[~np.isnan(r)].tolist()
        return {
            'observed_dn_ds': obs_dn_ds,
            'permuted_dn_ds': perm_dn_ds
        }

    ctrl = load_group(ctrl_fp, 'controls')
    treat = load_group(treat_fp, 'treated')
    ctrl_summary = summarize(ctrl)
    treat_summary = summarize(treat)

    # Save summary + plots
    rows = []
    for label, summ in [('controls', ctrl_summary), ('treated', treat_summary)]:
        if not summ:
            continue
        perm_vec = np.array(summ.get('permuted_dn_ds', []), dtype=float)
        perm_vec = perm_vec[~np.isnan(perm_vec)] if perm_vec.size else perm_vec
        row = {
            'group': label,
            'observed_dn_ds': float(summ.get('observed_dn_ds')) if summ.get('observed_dn_ds') is not None and not np.isnan(summ.get('observed_dn_ds')) else np.nan,
            'permuted_mean': float(np.nanmean(perm_vec)) if perm_vec.size else np.nan,
            'permuted_sd': float(np.nanstd(perm_vec)) if perm_vec.size else np.nan,
            'permuted_q05': float(np.nanpercentile(perm_vec, 5)) if perm_vec.size else np.nan,
            'permuted_q50': float(np.nanpercentile(perm_vec, 50)) if perm_vec.size else np.nan,
            'permuted_q95': float(np.nanpercentile(perm_vec, 95)) if perm_vec.size else np.nan,
        }
        rows.append(row)
        if perm_vec.size:
            plt.figure(figsize=(6, 4))
            plt.hist(perm_vec, bins=40, color='lightgrey', edgecolor='k')
            if not np.isnan(row['observed_dn_ds']):
                plt.axvline(row['observed_dn_ds'], color='red', linestyle='--', label='Observed')
            plt.xlabel('Genome-wide dN/dS')
            plt.ylabel('Count')
            plt.title(f'Permuted vs Observed dN/dS ({label})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'dn_ds_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / 'dn_ds_summary.csv', index=False)
        print(f"dN/dS summary written: {out_dir / 'dn_ds_summary.csv'}")


def plot_per_gene_dn_ds_histograms(results_dir: str, output_dir: str, max_genes_per_group: int = 50, min_perm: int = 20) -> None:
    """For each group, plot per-gene permutation dN/dS histograms with observed marker
    for up to max_genes_per_group genes (prioritized by observed total mutations).
    """
    results_path = Path(results_dir)
    out_dir = Path(output_dir) / 'per_gene_dn_ds'
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_group(name: str) -> dict:
        # Try exact filename first
        exact = results_path / f'permuted_provean_results_{name}.json'
        if not exact.exists():
            tags = ['treated', 'ems'] if name == 'treated' else ['control', 'controls', 'nt']
            files = list(results_path.glob('*.json'))
            candidates = [p for p in files if any(t in p.name.lower() for t in tags)]
            if candidates:
                preferred = [p for p in candidates if p.name.lower().startswith('permuted_provean')]
                exact = preferred[0] if preferred else candidates[0]
        if not exact.exists():
            print(f"Warning: pooled results not found for {name} in {results_path}")
            return {}
        with open(exact) as f:
            print(f"Using pooled file: {exact}")
            return json.load(f)

    def per_gene_dn_ds(group_data: dict) -> pd.DataFrame:
        rows = []
        for gene_id, vals in group_data.items():
            ms = vals.get('mutation_stats', {})
            obs_non = ms.get('non_syn_muts', 0) or 0
            obs_syn = ms.get('syn_muts', 0) or 0
            obs_dn_ds = (obs_non / obs_syn) if obs_syn not in (0, 0.0) else np.nan
            perm = vals.get('permutation', {})
            perm_non = np.array(perm.get('non_syn_counts', []), dtype=float)
            perm_tot = np.array(perm.get('total_mutations', []), dtype=float)
            if 'syn_counts' in perm:
                perm_syn = np.array(perm.get('syn_counts', []), dtype=float)
            else:
                perm_syn = perm_tot - perm_non if perm_non.size and perm_tot.size else np.array([])
            mask = (perm_syn > 0) if perm_syn.size else np.array([], dtype=bool)
            dn_ds_vec = (perm_non[mask] / perm_syn[mask]) if mask.size else np.array([])
            rows.append({
                'gene_id': gene_id,
                'obs_total': obs_non + obs_syn,
                'obs_dn_ds': float(obs_dn_ds) if not np.isnan(obs_dn_ds) else np.nan,
                'perm_dn_ds': dn_ds_vec
            })
        df = pd.DataFrame(rows)
        return df

    for label in ['controls', 'treated']:
        data = load_group(label)
        if not data:
            continue
        df = per_gene_dn_ds(data)
        # Filter genes with enough permutation support
        df = df[df['perm_dn_ds'].apply(lambda a: isinstance(a, np.ndarray) and a.size >= min_perm)]
        if df.empty:
            continue
        # Take top genes by observed totals
        df = df.sort_values('obs_total', ascending=False).head(max_genes_per_group)
        for _, row in df.iterrows():
            gene_id = row['gene_id']
            perm_vec = row['perm_dn_ds']
            obs = row['obs_dn_ds']
            if perm_vec.size == 0:
                continue
            plt.figure(figsize=(6, 4))
            plt.hist(perm_vec, bins=40, color='lightgrey', edgecolor='k')
            if not (obs is None or np.isnan(obs)):
                plt.axvline(obs, color='red', linestyle='--', label='Observed')
            plt.xlabel('Per-gene dN/dS')
            plt.ylabel('Count')
            plt.title(f'{gene_id} dN/dS ({label})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'{gene_id}_dn_ds_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()

def plot_per_gene_dn_ds_vs_deleterious_ratio(results_dir: str, output_dir: str) -> None:
    """Create per-gene scatter comparing observed dN/dS to observed deleterious ratio,
    overlaid against permutation mean expectations for both controls and treated.

    Outputs per group:
      - per_gene_dn_ds_vs_delratio_{group}.png
      - per_gene_dn_ds_vs_delratio_{group}.csv
    """
    results_path = Path(results_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_group(name: str) -> dict:
        exact = results_path / f'permuted_provean_results_{name}.json'
        if not exact.exists():
            tags = ['treated', 'ems'] if name == 'treated' else ['control', 'controls', 'nt']
            files = list(results_path.glob('*.json'))
            candidates = [p for p in files if any(t in p.name.lower() for t in tags)]
            if candidates:
                preferred = [p for p in candidates if p.name.lower().startswith('permuted_provean')]
                exact = preferred[0] if preferred else candidates[0]
        if not exact.exists():
            print(f"Warning: pooled results not found for {name} in {results_path}")
            return {}
        with open(exact) as f:
            print(f"Using pooled file: {exact}")
            return json.load(f)

    def summarize_group(group_data: dict) -> pd.DataFrame:
        rows = []
        for gene_id, vals in group_data.items():
            ms = vals.get('mutation_stats', {})
            obs_non = ms.get('non_syn_muts', 0) or 0
            obs_syn = ms.get('syn_muts', 0) or 0
            obs_tot = obs_non + obs_syn
            obs_dn_ds = (obs_non / obs_syn) if obs_syn not in (0, 0.0) else np.nan
            obs_del_ratio = (obs_non / obs_tot) if obs_tot not in (0, 0.0) else np.nan
            perm = vals.get('permutation', {})
            perm_non = np.array(perm.get('non_syn_counts', []), dtype=float)
            if 'syn_counts' in perm:
                perm_syn = np.array(perm.get('syn_counts', []), dtype=float)
            else:
                perm_tot = np.array(perm.get('total_mutations', []), dtype=float)
                perm_syn = perm_tot - perm_non if perm_non.size and perm_tot.size else np.array([])
            # Compute per-permutation ratios
            if perm_non.size and perm_syn.size:
                with np.errstate(divide='ignore', invalid='ignore'):
                    perm_dn_ds_vec = np.where(perm_syn > 0, perm_non / perm_syn, np.nan)
                    perm_del_ratio_vec = np.where((perm_non + perm_syn) > 0, perm_non / (perm_non + perm_syn), np.nan)
                perm_mean_dn_ds = float(np.nanmean(perm_dn_ds_vec)) if np.any(~np.isnan(perm_dn_ds_vec)) else np.nan
                perm_mean_del_ratio = float(np.nanmean(perm_del_ratio_vec)) if np.any(~np.isnan(perm_del_ratio_vec)) else np.nan
                n_perm = int(len(perm_dn_ds_vec))
            else:
                perm_mean_dn_ds = np.nan
                perm_mean_del_ratio = np.nan
                n_perm = 0
            rows.append({
                'gene_id': gene_id,
                'obs_non': obs_non,
                'obs_syn': obs_syn,
                'obs_total': obs_tot,
                'obs_dn_ds': obs_dn_ds,
                'obs_del_ratio': obs_del_ratio,
                'perm_mean_dn_ds': perm_mean_dn_ds,
                'perm_mean_del_ratio': perm_mean_del_ratio,
                'n_perms': n_perm
            })
        df = pd.DataFrame(rows)
        return df

    for label in ['controls', 'treated']:
        data = load_group(label)
        if not data:
            continue
        df = summarize_group(data)
        if df.empty:
            continue
        # Save CSV
        csv_path = out_dir / f'per_gene_dn_ds_vs_delratio_{label}.csv'
        df.to_csv(csv_path, index=False)
        # Plot: overlay permutation means (grey) and observed (colored by delta dN/dS)
        plt.figure(figsize=(8, 6))
        # Reference expectation cloud
        ref_df = df.dropna(subset=['perm_mean_del_ratio', 'perm_mean_dn_ds'])
        if not ref_df.empty:
            plt.scatter(ref_df['perm_mean_del_ratio'], ref_df['perm_mean_dn_ds'],
                        color='lightgrey', alpha=0.6, s=12, label='Permutation mean (per gene)')
        # Observed, colored by delta
        obs_df = df.dropna(subset=['obs_del_ratio', 'obs_dn_ds'])
        if not obs_df.empty:
            delta = obs_df['obs_dn_ds'] - obs_df['perm_mean_dn_ds']
            sc = plt.scatter(obs_df['obs_del_ratio'], obs_df['obs_dn_ds'],
                             c=delta, cmap='coolwarm', alpha=0.7, s=14, edgecolors='none')
            cbar = plt.colorbar(sc)
            cbar.set_label('Δ dN/dS (obs - perm mean)')
        plt.xlabel('Observed deleterious ratio (non_syn / total)')
        plt.ylabel('Observed dN/dS (non_syn / syn)')
        plt.title(f'Per-gene dN/dS vs deleterious ratio ({label})')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(out_dir / f'per_gene_dn_ds_vs_delratio_{label}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dn_ds_pvalue_distribution_plots(results_dir: str, output_dir: str, min_perms: int = 10) -> None:
    """Create distribution plots of empirical p-values for per-gene dN/dS using pooled
    control and treated permutation outputs.

    Uses non_syn and total counts per permutation (syn = total - non_syn).
    Autodetects pooled files in results_dir by filename tags.
    Applies observed-count filters: obs_syn >= 3 and obs_total >= 5.
    """
    from pathlib import Path

    def pick_group_file(dir_path: Path, group: str) -> dict:
        files = list(dir_path.glob('*.json'))
        if not files:
            print(f"Warning: no JSON files found in {dir_path}")
            return {}
        # Exact expected name first
        exact = dir_path / f'permuted_provean_results_{group}.json'
        if exact.exists():
            with open(exact) as f:
                print(f"Using pooled file: {exact}")
                return json.load(f)
        # Tag-based fallback
        tag_pos = ['treated', 'ems'] if group == 'treated' else ['control', 'controls', 'nt']
        candidates = [p for p in files if any(t in p.name.lower() for t in tag_pos)]
        if candidates:
            # Prefer ones starting with 'permuted_provean' to avoid per-sample
            preferred = [p for p in candidates if p.name.lower().startswith('permuted_provean')]
            fp = preferred[0] if preferred else candidates[0]
            with open(fp) as f:
                print(f"Using pooled file (tag match): {fp}")
                return json.load(f)
        print(f"Warning: no pooled JSON matched tags for {group} in {dir_path}")
        return {}

    dirp = Path(results_dir)
    controls = pick_group_file(dirp, 'controls')
    treated = pick_group_file(dirp, 'treated')

    def per_gene_pvals(group_data: dict) -> tuple[list, list]:
        p_more_list: list[float] = []
        p_less_list: list[float] = []
        eps = 1e-9
        min_obs_syn = 3
        min_obs_total = 5
        for _, vals in group_data.items():
            ms = vals.get('mutation_stats', {})
            obs_non = ms.get('non_syn_muts', None)
            obs_tot = ms.get('total_mutations', None)
            if obs_tot is None and (ms.get('syn_muts', None) is not None) and (obs_non is not None):
                obs_tot = (ms.get('syn_muts') or 0) + (obs_non or 0)
            if obs_non is None or obs_tot is None:
                continue
            obs_syn = max(obs_tot - obs_non, 0)
            # Enforce observed count thresholds
            if obs_syn < min_obs_syn or obs_tot < min_obs_total:
                continue
            obs_dn_ds = obs_non / max(obs_syn, eps)

            perm = vals.get('permutation', {})
            perm_non = np.array(perm.get('non_syn_counts', []), dtype=float)
            perm_tot = np.array(perm.get('total_mutations', []), dtype=float)
            if perm_non.size == 0 or perm_tot.size == 0:
                continue
            perm_syn = perm_tot - perm_non
            with np.errstate(divide='ignore', invalid='ignore'):
                dn_ds_vec = np.divide(perm_non, np.maximum(perm_syn, eps))
            dn_ds_vec = dn_ds_vec[np.isfinite(dn_ds_vec)]
            if dn_ds_vec.size < min_perms:
                continue
            n_more = int(np.sum(dn_ds_vec > obs_dn_ds))
            n_less = int(np.sum(dn_ds_vec < obs_dn_ds))
            p_more_list.append(n_more / dn_ds_vec.size)
            p_less_list.append(n_less / dn_ds_vec.size)
        return p_more_list, p_less_list

    c_more, c_less = per_gene_pvals(controls) if controls else ([], [])
    t_more, t_less = per_gene_pvals(treated) if treated else ([], [])

    c_more_fdr = multipletests(c_more, method='fdr_bh')[1].tolist() if len(c_more) else []
    c_less_fdr = multipletests(c_less, method='fdr_bh')[1].tolist() if len(c_less) else []
    t_more_fdr = multipletests(t_more, method='fdr_bh')[1].tolist() if len(t_more) else []
    t_less_fdr = multipletests(t_less, method='fdr_bh')[1].tolist() if len(t_less) else []

    dist_plot_dir = os.path.join(output_dir, "distribution_plots")
    os.makedirs(dist_plot_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bins = np.linspace(0, 1, 51)

    axes[0, 0].hist([c_more, t_more], bins=bins, label=['Control', 'EMS Treated'],
                    stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
    axes[0, 0].axvline(0.05, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('dN/dS p-values (more)')
    axes[0, 0].set_xlabel('p-value')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].hist([c_less, t_less], bins=bins, label=['Control', 'EMS Treated'],
                    stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
    axes[0, 1].axvline(0.05, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('dN/dS p-values (less)')
    axes[0, 1].set_xlabel('p-value')
    axes[0, 1].set_ylabel('Count')

    axes[1, 0].hist([c_more_fdr, t_more_fdr], bins=bins, label=['Control', 'EMS Treated'],
                    stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
    axes[1, 0].axvline(0.05, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('dN/dS p-values (more) - FDR corrected')
    axes[1, 0].set_xlabel('FDR')
    axes[1, 0].set_ylabel('Count')

    axes[1, 1].hist([c_less_fdr, t_less_fdr], bins=bins, label=['Control', 'EMS Treated'],
                    stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
    axes[1, 1].axvline(0.05, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('dN/dS p-values (less) - FDR corrected')
    axes[1, 1].set_xlabel('FDR')
    axes[1, 1].set_ylabel('Count')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=2)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.suptitle('Per-gene dN/dS empirical p-value distributions', fontsize=16)
    plt.savefig(os.path.join(dist_plot_dir, 'dn_ds_pvalue_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    def save_panel(data_pair, title, filename, xlabel='p-value'):
        plt.figure(figsize=(12, 6))
        plt.hist(data_pair, bins=bins, label=['Control', 'EMS Treated'],
                 stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
        plt.axvline(0.05, color='r', linestyle='--', alpha=0.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(dist_plot_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    save_panel([c_more, t_more], 'dN/dS p-values (more)', 'dn_ds_pvalues_more.png')
    save_panel([c_less, t_less], 'dN/dS p-values (less)', 'dn_ds_pvalues_less.png')
    save_panel([c_more_fdr, t_more_fdr], 'dN/dS FDR (more)', 'dn_ds_pvalues_more_fdr.png', xlabel='FDR')
    save_panel([c_less_fdr, t_less_fdr], 'dN/dS FDR (less)', 'dn_ds_pvalues_less_fdr.png', xlabel='FDR')

    print(f"dN/dS p-value distribution plots saved to: {dist_plot_dir}")

def plot_effect_relationships_publication(df: pd.DataFrame, gene_lengths: Dict[str, int], output_dir: str) -> None:
    """Publication-style concise plot of effect score vs gene length for treated.

    - Clips axes to reduce the impact of outliers (1st–99th percentiles)
    - Uses "Positive effect" instead of "Protective"
    - Title: "Effect score vs gene length - treated"
    - Log scale for gene length
    """
    # Significance using FDR corrected p-values
    deleterious = (df['pvalue_more_fdr'] < 0.05)
    positive = (df['pvalue_less_fdr'] < 0.05)
    nonsig = ~(deleterious | positive)

    # Gene lengths
    gene_lens = df['gene_id'].map(lambda x: gene_lengths.get(x, np.nan))

    # Fixed bounds and single-axis with outlier markers for publication
    x_min = 75
    x_max = 8000.0
    y_main_low, y_main_high = -200.0, 10.0

    # Masks
    within = (gene_lens >= x_min) & (gene_lens <= x_max) & \
             (df['effect_score'] >= y_main_low) & (df['effect_score'] <= y_main_high)
    below = (gene_lens >= x_min) & (gene_lens <= x_max) & (df['effect_score'] < y_main_low)
    above = (gene_lens >= x_min) & (gene_lens <= x_max) & (df['effect_score'] > y_main_high)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Inliers
    ax.scatter(gene_lens[nonsig & within], df[nonsig & within]['effect_score'], s=10,
               color='lightgrey', alpha=0.7, label='Non-significant')
    ax.scatter(gene_lens[deleterious & within], df[deleterious & within]['effect_score'], s=12,
               color='darkorange', alpha=0.9, label='Deleterious (FDR<0.05)')
    ax.scatter(gene_lens[positive & within], df[positive & within]['effect_score'], s=12,
               color='royalblue', alpha=0.9, label='Positive effect (FDR<0.05)')

    # Outliers as aggregated edge markers
    pad = 0.03 * (y_main_high - y_main_low)
    bins = np.logspace(np.log10(x_min), np.log10(x_max), 25)
    centers = np.sqrt(bins[:-1] * bins[1:])
    if below.any():
        counts_below, _ = np.histogram(gene_lens[below], bins=bins)
        mask = counts_below > 0
        sizes = 18 + 16 * np.sqrt(counts_below[mask])
        ax.scatter(centers[mask], np.full(mask.sum(), y_main_low + pad),
                   marker='v', s=sizes, facecolors='none', edgecolors='black', linewidths=0.6,
                   alpha=0.9, label='Below range', zorder=5)
    if above.any():
        counts_above, _ = np.histogram(gene_lens[above], bins=bins)
        mask = counts_above > 0
        sizes = 18 + 16 * np.sqrt(counts_above[mask])
        ax.scatter(centers[mask], np.full(mask.sum(), y_main_high - pad),
                   marker='^', s=sizes, facecolors='none', edgecolors='black', linewidths=0.6,
                   alpha=0.9, label='Above range', zorder=5)

    # Annotate counts
    n_below = int(below.sum())
    n_above = int(above.sum())
    if n_below:
        ax.text(0.99, 0.02, f"{n_below} below", transform=ax.transAxes, fontsize=10,
                va='bottom', ha='right', color='black')
    if n_above:
        ax.text(0.01, 0.98, f"{n_above} above", transform=ax.transAxes, fontsize=10,
                va='top', ha='left', color='black')

    ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_main_low, y_main_high)
    ax.set_xlabel('Gene length (bp)', fontsize=14)
    ax.set_ylabel('Effect score', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(frameon=True, loc='lower left', fontsize=12, fancybox=True, framealpha=0.85, borderpad=0.7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'effect_vs_gene_length_treated_publication.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'permuted_provean_EMS_effect_relationships.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def compute_per_gene_metric_pvalues(results_dir: str, output_dir: str, metric: str, fdr_threshold: float = 0.05,
                                   min_obs_syn: int = 3, min_obs_total: int = 5, min_perms: int = 10) -> None:
	"""Compute per-gene empirical p-values for a metric and write significant gene lists and volcano plots.

	Supported metrics:
	  - 'dn_ds': uses observed non_syn/syn and per-permutation non_syn/syn
	  - 'nonsyn_fraction': uses observed non_syn/total and per-permutation non_syn/total

	Writes per-group CSVs with p-values and FDR, significant-only CSVs for more/less tails,
	and volcano plots per group.
	"""
	results_path = Path(results_dir)
	out_dir = Path(output_dir) / 'per_gene_significance'
	out_dir.mkdir(parents=True, exist_ok=True)

	def pick_group_file(name: str) -> dict:
		exact = results_path / f'permuted_provean_results_{name}.json'
		if exact.exists():
			with open(exact) as f:
				print(f"Using pooled file: {exact}")
				return json.load(f)
		tags = ['treated', 'ems'] if name == 'treated' else ['control', 'controls', 'nt']
		files = list(results_path.glob('*.json'))
		candidates = [p for p in files if any(t in p.name.lower() for t in tags)]
		if candidates:
			preferred = [p for p in candidates if p.name.lower().startswith('permuted_provean')]
			fp = preferred[0] if preferred else candidates[0]
			with open(fp) as f:
				print(f"Using pooled file (tag match): {fp}")
				return json.load(f)
		print(f"Warning: pooled results not found for {name} in {results_path}")
		return {}

	def per_gene_rows(group_dict: dict) -> pd.DataFrame:
		eps = 1e-9
		rows = []
		for gene_id, vals in group_dict.items():
			ms = vals.get('mutation_stats', {})
			obs_non = float(ms.get('non_syn_muts', 0) or 0)
			obs_syn = float(ms.get('syn_muts', 0) or 0)
			obs_tot = float(ms.get('total_mutations', 0) or (obs_non + obs_syn))
			# Filters on observed counts
			if metric == 'dn_ds':
				if obs_syn < min_obs_syn or obs_tot < min_obs_total:
					continue
				obs_val = obs_non / max(obs_syn, eps)
			elif metric == 'nonsyn_fraction':
				if obs_tot < min_obs_total:
					continue
				obs_val = obs_non / max(obs_tot, eps)
			else:
				continue
			perm = vals.get('permutation', {})
			perm_non = np.array(perm.get('non_syn_counts', []), dtype=float)
			perm_tot = np.array(perm.get('total_mutations', []), dtype=float)
			if perm_non.size == 0 or perm_tot.size == 0:
				continue
			if metric == 'dn_ds':
				perm_syn = perm_tot - perm_non
				with np.errstate(divide='ignore', invalid='ignore'):
					perm_vals = np.divide(perm_non, np.maximum(perm_syn, eps))
			else:
				with np.errstate(divide='ignore', invalid='ignore'):
					perm_vals = np.divide(perm_non, np.maximum(perm_tot, eps))
			perm_vals = perm_vals[np.isfinite(perm_vals)]
			if perm_vals.size < min_perms:
				continue
			n_more = int(np.sum(perm_vals > obs_val))
			n_less = int(np.sum(perm_vals < obs_val))
			p_more = n_more / perm_vals.size
			p_less = n_less / perm_vals.size
			rows.append({
				'gene_id': gene_id,
				'obs_non_syn': obs_non,
				'obs_syn': obs_syn,
				'obs_total': obs_tot,
				'observed_metric': obs_val,
				'p_more': p_more,
				'p_less': p_less,
				'n_perms': int(perm_vals.size)
			})
		return pd.DataFrame(rows)

	for group in ['controls', 'treated']:
		data = pick_group_file(group)
		if not data:
			continue
		df = per_gene_rows(data)
		if df.empty:
			continue
		# FDR
		df['p_more_fdr'] = np.nan
		df['p_less_fdr'] = np.nan
		if df['p_more'].notna().any():
			df.loc[df['p_more'].notna(), 'p_more_fdr'] = multipletests(df.loc[df['p_more'].notna(), 'p_more'], method='fdr_bh')[1]
		if df['p_less'].notna().any():
			df.loc[df['p_less'].notna(), 'p_less_fdr'] = multipletests(df.loc[df['p_less'].notna(), 'p_less'], method='fdr_bh')[1]
		# Save full table
		metric_tag = 'dn_ds' if metric == 'dn_ds' else 'nonsyn_fraction'
		full_path = out_dir / f'per_gene_{metric_tag}_pvalues_{group}.csv'
		df.to_csv(full_path, index=False)
		print(f"Wrote: {full_path}")
		# Significant lists
		more_sig = df[df['p_more_fdr'] < fdr_threshold].copy()
		less_sig = df[df['p_less_fdr'] < fdr_threshold].copy()
		more_path = out_dir / f'per_gene_{metric_tag}_significant_more_{group}.csv'
		less_path = out_dir / f'per_gene_{metric_tag}_significant_less_{group}.csv'
		more_sig.to_csv(more_path, index=False)
		less_sig.to_csv(less_path, index=False)
		print(f"Wrote: {more_path} ({len(more_sig)})")
		print(f"Wrote: {less_path} ({len(less_sig)})")
		# Volcano plot
		if not df.empty and (df['p_more_fdr'].notna().any() or df['p_less_fdr'].notna().any()):
			plt.figure(figsize=(6,5))
			# Two-tail display: color by tail with smaller FDR
			tail = np.where(df['p_more_fdr'].fillna(1.0) <= df['p_less_fdr'].fillna(1.0), 'more', 'less')
			x = df['observed_metric']
			y = -np.log10(np.minimum(df['p_more_fdr'].fillna(1.0), df['p_less_fdr'].fillna(1.0)))
			colors = np.where(tail == 'more', 'firebrick', 'royalblue')
			plt.scatter(x, y, c=colors, alpha=0.6, s=12)
			plt.axhline(-np.log10(fdr_threshold), color='k', linestyle='--', alpha=0.4)
			xlab = 'Observed dN/dS' if metric == 'dn_ds' else 'Observed non-syn fraction'
			title = f"Volcano: {xlab} ({group})"
			plt.xlabel(xlab)
			plt.ylabel('-log10(FDR)')
			plt.title(title)
			plt.tight_layout()
			vp = out_dir / f'volcano_{metric_tag}_{group}.png'
			plt.savefig(vp, dpi=300, bbox_inches='tight')
			plt.close()


def create_nonsyn_fraction_pvalue_distribution_plots(results_dir: str, output_dir: str, min_perms: int = 10, min_obs_total: int = 5) -> None:
	"""Create distribution plots of empirical p-values for per-gene non-synonymous fraction.
	Uses per-gene permutation arrays (non_syn_counts, total_mutations).
	"""
	dirp = Path(results_dir)

	def pick_group_file(name: str) -> dict:
		exact = dirp / f'permuted_provean_results_{name}.json'
		if exact.exists():
			with open(exact) as f:
				print(f"Using pooled file: {exact}")
				return json.load(f)
		tags = ['treated', 'ems'] if name == 'treated' else ['control', 'controls', 'nt']
		files = list(dirp.glob('*.json'))
		candidates = [p for p in files if any(t in p.name.lower() for t in tags)]
		if candidates:
			preferred = [p for p in candidates if p.name.lower().startswith('permuted_provean')]
			fp = preferred[0] if preferred else candidates[0]
			with open(fp) as f:
				print(f"Using pooled file (tag match): {fp}")
				return json.load(f)
		print(f"Warning: pooled results not found for {name} in {dirp}")
		return {}

	def per_gene_pvals(group_data: dict) -> tuple[list, list]:
		p_more_list: list[float] = []
		p_less_list: list[float] = []
		eps = 1e-9
		for _, vals in group_data.items():
			ms = vals.get('mutation_stats', {})
			obs_non = ms.get('non_syn_muts', None)
			obs_tot = ms.get('total_mutations', None)
			if obs_tot is None and (ms.get('syn_muts', None) is not None) and (obs_non is not None):
				obs_tot = (ms.get('syn_muts') or 0) + (obs_non or 0)
			if obs_non is None or obs_tot is None:
				continue
			if obs_tot < min_obs_total:
				continue
			obs_frac = obs_non / max(obs_tot, eps)
			perm = vals.get('permutation', {})
			perm_non = np.array(perm.get('non_syn_counts', []), dtype=float)
			perm_tot = np.array(perm.get('total_mutations', []), dtype=float)
			if perm_non.size == 0 or perm_tot.size == 0:
				continue
			with np.errstate(divide='ignore', invalid='ignore'):
				frac_vec = np.divide(perm_non, np.maximum(perm_tot, eps))
			frac_vec = frac_vec[np.isfinite(frac_vec)]
			if frac_vec.size < min_perms:
				continue
			n_more = int(np.sum(frac_vec > obs_frac))
			n_less = int(np.sum(frac_vec < obs_frac))
			p_more_list.append(n_more / frac_vec.size)
			p_less_list.append(n_less / frac_vec.size)
		return p_more_list, p_less_list

	controls = pick_group_file('controls')
	treated = pick_group_file('treated')

	c_more, c_less = per_gene_pvals(controls) if controls else ([], [])
	t_more, t_less = per_gene_pvals(treated) if treated else ([], [])

	c_more_fdr = multipletests(c_more, method='fdr_bh')[1].tolist() if len(c_more) else []
	c_less_fdr = multipletests(c_less, method='fdr_bh')[1].tolist() if len(c_less) else []
	t_more_fdr = multipletests(t_more, method='fdr_bh')[1].tolist() if len(t_more) else []
	t_less_fdr = multipletests(t_less, method='fdr_bh')[1].tolist() if len(t_less) else []

	dist_plot_dir = os.path.join(output_dir, "distribution_plots")
	os.makedirs(dist_plot_dir, exist_ok=True)

	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	bins = np.linspace(0, 1, 51)

	axes[0, 0].hist([c_more, t_more], bins=bins, label=['Control', 'EMS Treated'],
					stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
	axes[0, 0].axvline(0.05, color='r', linestyle='--', alpha=0.5)
	axes[0, 0].set_title('Non-syn fraction p-values (more)')
	axes[0, 0].set_xlabel('p-value')
	axes[0, 0].set_ylabel('Count')

	axes[0, 1].hist([c_less, t_less], bins=bins, label=['Control', 'EMS Treated'],
					stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
	axes[0, 1].axvline(0.05, color='r', linestyle='--', alpha=0.5)
	axes[0, 1].set_title('Non-syn fraction p-values (less)')
	axes[0, 1].set_xlabel('p-value')
	axes[0, 1].set_ylabel('Count')

	axes[1, 0].hist([c_more_fdr, t_more_fdr], bins=bins, label=['Control', 'EMS Treated'],
					stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
	axes[1, 0].axvline(0.05, color='r', linestyle='--', alpha=0.5)
	axes[1, 0].set_title('Non-syn fraction p-values (more) - FDR corrected')
	axes[1, 0].set_xlabel('FDR')
	axes[1, 0].set_ylabel('Count')

	axes[1, 1].hist([c_less_fdr, t_less_fdr], bins=bins, label=['Control', 'EMS Treated'],
					stacked=True, alpha=0.7, color=['lightgrey', 'darkorange'])
	axes[1, 1].axvline(0.05, color='r', linestyle='--', alpha=0.5)
	axes[1, 1].set_title('Non-syn fraction p-values (less) - FDR corrected')
	axes[1, 1].set_xlabel('FDR')
	axes[1, 1].set_ylabel('Count')

	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=2)
	plt.tight_layout(rect=[0, 0.06, 1, 0.96])
	plt.suptitle('Per-gene non-syn fraction empirical p-value distributions', fontsize=16)
	plt.savefig(os.path.join(dist_plot_dir, 'nonsyn_fraction_pvalue_distributions.png'), dpi=300, bbox_inches='tight')
	plt.close()

# --- New: Synonymous sites analysis from provean_effectscore outputs ---

def analyze_synonymous_sites(results_dir: str, output_dir: str, gene_lengths: Dict[str, int]) -> None:
    """Analyze observed vs potential synonymous sites produced by provean_effectscore.
    - Reads control/treated CSVs: *_observed_vs_potential_syn_sites.csv
    - Adds gene length and per-kb normalization
    - Reads expected vs observed non-syn fraction to derive expected syn fraction
    - Writes combined tables and plots distributions (observed fraction, expected syn fraction)
    """
    os.makedirs(os.path.join(output_dir, 'synonymous_analysis'), exist_ok=True)
    syn_dir = os.path.join(output_dir, 'synonymous_analysis')

    def load_csv(path: str) -> pd.DataFrame:
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    # Observed vs potential synonymous sites (treated only)
    treat_syn_path = os.path.join(results_dir, 'treated_observed_vs_potential_syn_sites.csv')
    df_treat = load_csv(treat_syn_path)

    if df_treat.empty:
        print("Treated synonymous site CSV not found; skipping synonymous analysis")
        return

    # Add gene length and per-kb normalizations
    def add_len_norm(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        # Robust mapping: coerce IDs to string; fallback to numeric extraction
        df['gene_length'] = df['gene_id'].astype(str).map(gene_lengths)
        missing = df['gene_length'].isna()
        if missing.any():
            gid_num = df.loc[missing, 'gene_id'].astype(str).str.extract(r'(\d+)')[0]
            df.loc[missing, 'gene_length'] = gid_num.map(gene_lengths)
        # Compute observed fraction if missing
        if 'observed_fraction_of_potential_syn' not in df.columns or df['observed_fraction_of_potential_syn'].isna().all():
            with np.errstate(divide='ignore', invalid='ignore'):
                df['observed_fraction_of_potential_syn'] = np.where(
                    df['potential_syn_sites'] > 0,
                    df['observed_syn_sites'] / df['potential_syn_sites'],
                    np.nan
                )
        kb = df['gene_length'] / 1000.0
        with np.errstate(divide='ignore', invalid='ignore'):
            df['potential_syn_sites_per_kb'] = np.where(kb > 0, df['potential_syn_sites'] / kb, np.nan)
            df['observed_syn_sites_per_kb'] = np.where(kb > 0, df['observed_syn_sites'] / kb, np.nan)
        return df

    df_treat = add_len_norm(df_treat)

    if not df_treat.empty:
        df_treat.to_csv(os.path.join(syn_dir, 'treated_syn_sites_len_normalized.csv'), index=False)

    # Treated-only plotting of observed fraction distributions
    if not df_treat.empty:
        df_all = df_treat.copy()
        df_all.to_csv(os.path.join(syn_dir, 'syn_sites_treated_only.csv'), index=False)

        # Histogram of observed fraction of potential synonymous mutations (treated only)
        plt.figure(figsize=(8, 5))
        bins = np.linspace(0, 1, 51)
        data_treat = df_all['observed_fraction_of_potential_syn'].dropna().values
        plt.hist([data_treat], bins=bins, label=['EMS Treated'],
                 stacked=True, alpha=0.8, color=['darkorange'])
        plt.xlabel('Observed fraction of potential synonymous mutations')
        plt.ylabel('Count')
        plt.title('Synonymous: observed fraction of potential sites (treated)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(syn_dir, 'observed_fraction_of_potential_syn_hist_treated.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Scatter: gene length vs observed fraction (treated only)
        plt.figure(figsize=(8, 6))
        any_positive = False
        # Filter to positive gene lengths and finite fractions
        s = df_all[['gene_length', 'observed_fraction_of_potential_syn']].dropna()
        s = s[s['gene_length'] > 0]
        if not s.empty:
            any_positive = (s['gene_length'] > 0).any()
            plt.scatter(s['gene_length'], s['observed_fraction_of_potential_syn'],
                        alpha=0.6, s=16, label='treated', color='darkorange')
        if any_positive:
            plt.xscale('log')
        plt.xlabel('Gene length (bp)')
        plt.ylabel('Observed fraction of potential synonymous')
        plt.title('Synonymous fraction vs gene length (treated)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(syn_dir, 'syn_fraction_vs_gene_length_treated.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Expected vs observed synonymous fractions derived from expected/observed non-syn CSVs (treated only)
    treat_en_path = os.path.join(results_dir, 'treated_expected_vs_observed_nonsyn_fraction.csv')
    df_treat_en = load_csv(treat_en_path)

    def derive_syn(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out['expected_syn_fraction'] = 1.0 - out['expected_nonsyn_fraction']
        out['observed_syn_fraction'] = 1.0 - out['observed_nonsyn_fraction']
        out['gene_length'] = out['gene_id'].map(gene_lengths)
        return out

    df_treat_syn = derive_syn(df_treat_en) if not df_treat_en.empty else pd.DataFrame()

    if not df_treat_syn.empty:
        df_treat_syn.to_csv(os.path.join(syn_dir, 'expected_vs_observed_syn_fraction_treated.csv'), index=False)

        # Scatter + stats: expected vs observed syn fraction (treated only)
        plt.figure(figsize=(7, 6))
        sub = df_treat_syn.dropna(subset=['expected_syn_fraction', 'observed_syn_fraction'])
        if not sub.empty:
            plt.scatter(sub['expected_syn_fraction'], sub['observed_syn_fraction'],
                        alpha=0.7, s=16, color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        plt.xlabel('Expected synonymous fraction')
        plt.ylabel('Observed synonymous fraction')
        plt.title('Expected vs observed synonymous fraction per gene (treated)')
        plt.tight_layout()
        plt.savefig(os.path.join(syn_dir, 'expected_vs_observed_syn_fraction_scatter_treated.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Statistical comparison: Wilcoxon signed-rank (paired per gene)
        try:
            from scipy.stats import wilcoxon
            if not sub.empty and len(sub) >= 5:
                stat, pval = wilcoxon(sub['observed_syn_fraction'], sub['expected_syn_fraction'], zero_method='wilcox', alternative='two-sided', correction=False, mode='auto')
                pd.DataFrame({
                    'test': ['wilcoxon_signed_rank'],
                    'n_genes': [len(sub)],
                    'statistic': [stat],
                    'p_value': [pval],
                    'observed_mean': [float(np.mean(sub['observed_syn_fraction']))],
                    'expected_mean': [float(np.mean(sub['expected_syn_fraction']))],
                    'observed_median': [float(np.median(sub['observed_syn_fraction']))],
                    'expected_median': [float(np.median(sub['expected_syn_fraction']))]
                }).to_csv(os.path.join(syn_dir, 'expected_vs_observed_syn_stats_treated.csv'), index=False)
        except Exception as e:
            print(f"Warning: failed to compute Wilcoxon stats for treated syn fractions: {e}")

        # Hist: expected syn fraction distribution (treated only)
        plt.figure(figsize=(8, 5))
        bins = np.linspace(0, 1, 51)
        et = df_treat_syn['expected_syn_fraction'].dropna().values
        plt.hist([et], bins=bins, label=['EMS Treated'],
                 stacked=True, alpha=0.8, color=['darkorange'])
        plt.xlabel('Expected synonymous fraction')
        plt.ylabel('Count')
        plt.title('Expected distribution of synonymous fraction (treated)')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(syn_dir, 'expected_syn_fraction_distribution_treated.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_gene_metrics_vs_length(df: pd.DataFrame, gene_lengths: Dict[str, int], sample: str, output_dir: str) -> None:
    """Create plots similar to effect_relationships for non-syn fraction (non_syn/total) and dN/dS (non_syn/syn)."""
    # Prepare metrics
    gene_lens = df['gene_id'].map(lambda x: gene_lengths.get(x, np.nan))
    # non-syn fraction
    with np.errstate(divide='ignore', invalid='ignore'):
        ns_frac = np.where(df['total_mutations'] > 0, df['non_syn_muts'] / df['total_mutations'], np.nan)
    dn_ds = df.get('dn_ds', np.nan)

    x_min = 75
    x_max = 8000.0

    def scatter_metric(y, y_label, filename, more_fdr_col=None, less_fdr_col=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        y_valid = pd.Series(y).astype(float)
        within = (gene_lens >= x_min) & (gene_lens <= x_max) & np.isfinite(y_valid)
        # Significance labeling (match style of effect plot)
        colors = np.array(['grey'] * len(df))
        alphas = np.array([0.3] * len(df), dtype=float)
        labels = np.array(['Non-significant'] * len(df), dtype=object)
        if more_fdr_col in df.columns:
            more_sig = df[more_fdr_col] < 0.05
            colors[more_sig] = 'darkorange'
            alphas[more_sig] = 0.7
            labels[more_sig] = 'More (significant)'
        if less_fdr_col in df.columns:
            less_sig = df[less_fdr_col] < 0.05
            colors[less_sig] = 'royalblue'
            alphas[less_sig] = 0.7
            labels[less_sig] = 'Less (significant)'
        # Plot points by mask to create legend entries
        mask = within & (labels == 'Non-significant')
        ax.scatter(gene_lens[mask], y_valid[mask], alpha=0.3, s=16, color='grey', label='Non-significant')
        mask = within & (labels == 'More (significant)')
        ax.scatter(gene_lens[mask], y_valid[mask], alpha=0.7, s=16, color='darkorange', label='More (significant)')
        mask = within & (labels == 'Less (significant)')
        ax.scatter(gene_lens[mask], y_valid[mask], alpha=0.7, s=16, color='royalblue', label='Less (significant)')
        # Axes formatting
        ax.set_xlabel('Gene Length (bp)', fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        ax.tick_params(axis='both', which='major', labelsize=14)
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='x', style='plain')
        # Legend bottom-left
        ax.legend(loc='lower left', frameon=True, fontsize=13, fancybox=True, framealpha=0.85, borderpad=0.7)
        ax.set_title(f'{y_label} vs Gene Length - Treated', fontsize=18)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{sample}_{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    scatter_metric(ns_frac, 'Non-synonymous fraction (non_syn / total)', 'nonsyn_fraction_relationships', more_fdr_col='pvalue_nonsyn_more_fdr', less_fdr_col='pvalue_nonsyn_less_fdr')
    scatter_metric(dn_ds, 'dN/dS (non_syn / syn)', 'dn_ds_relationships', more_fdr_col='pvalue_dnds_more_fdr', less_fdr_col='pvalue_dnds_less_fdr')

def write_gene_perm_counts_csv(results_dir: str, output_dir: str, group: str = 'treated', n_perms: int = 100) -> None:
    """Write a CSV with one row per gene containing observed non-syn/syn counts and
    the first n_perms permutation non-syn and syn counts.

    Columns:
      gid, observed_nonsyn, observed_syn, perm1_non, perm1_syn, ..., perm{n}_non, perm{n}_syn
    """
    results_path = Path(results_dir)
    # Pick pooled file like other helpers
    def pick_group_file(dir_path: Path, grp: str) -> dict:
        exact = dir_path / f'permuted_provean_results_{grp}.json'
        if exact.exists():
            with open(exact) as f:
                print(f"Using pooled file: {exact}")
                return json.load(f)
        tags = ['treated', 'ems'] if grp == 'treated' else ['control', 'controls', 'nt']
        files = list(dir_path.glob('*.json'))
        candidates = [p for p in files if any(t in p.name.lower() for t in tags)]
        if candidates:
            preferred = [p for p in candidates if p.name.lower().startswith('permuted_provean')]
            fp = preferred[0] if preferred else candidates[0]
            with open(fp) as f:
                print(f"Using pooled file (tag match): {fp}")
                return json.load(f)
        print(f"Warning: pooled results not found for {grp} in {dir_path}")
        return {}

    data = pick_group_file(results_path, group)
    if not data:
        return

    rows = []
    for gene_id, vals in data.items():
        ms = vals.get('mutation_stats', {})
        obs_non = ms.get('non_syn_muts', None)
        obs_syn = ms.get('syn_muts', None)
        if obs_syn is None:
            tot = ms.get('total_mutations', None)
            if tot is not None and obs_non is not None:
                obs_syn = max(tot - obs_non, 0)
        perm = vals.get('permutation', {})
        non_arr = np.array(perm.get('non_syn_counts', []), dtype=float)
        if 'syn_counts' in perm:
            syn_arr = np.array(perm.get('syn_counts', []), dtype=float)
        else:
            tot_arr = np.array(perm.get('total_mutations', []), dtype=float)
            syn_arr = tot_arr - non_arr if tot_arr.size else np.array([])
        # Ensure same length
        L = min(len(non_arr), len(syn_arr), n_perms)
        # Build row dict
        row = {
            'gid': gene_id,
            'observed_nonsyn': obs_non if obs_non is not None else np.nan,
            'observed_syn': obs_syn if obs_syn is not None else np.nan
        }
        for i in range(L):
            row[f'perm{i+1}_non'] = float(non_arr[i])
            row[f'perm{i+1}_syn'] = float(syn_arr[i])
        # Pad missing perms up to n_perms with NaN columns to keep header consistent
        for i in range(L+1, n_perms+1):
            row[f'perm{i}_non'] = np.nan
            row[f'perm{i}_syn'] = np.nan
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        # Order columns
        cols = ['gid', 'observed_nonsyn', 'observed_syn'] + [c for i in range(1, n_perms+1) for c in (f'perm{i}_non', f'perm{i}_syn')]
        df = df.reindex(columns=cols)
        outp = Path(output_dir) / f'gene_perm_counts_{group}_first{n_perms}.csv'
        df.to_csv(outp, index=False)
        print(f"Wrote: {outp}")

def main():
	parser = argparse.ArgumentParser(description='Analyze PROVEAN score results')
	parser.add_argument('results_dir', help='Directory containing sample results JSON files')
	parser.add_argument('output_dir', help='Directory to save analysis results')
	parser.add_argument('-c', '--config', required=True, help='Path to config file with reference paths')
	args = parser.parse_args()
	
	# Load config
	with open(args.config) as f:
		config = yaml.safe_load(f)
	
	# Load gene lengths and strands from GFF
	gene_lengths, gene_strands = load_gene_lengths_and_strands(config['references']['annotation'])
	
	# Get cache path from config
	cache_file = config['references']['gene_info_cache']
	
	# Create output directory
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Load module assignments
	module_df = load_module_assignments(config['references']['module_assignments'])
	
	# Load expression data
	expression_df = load_expression_data(config['references']['expression_data'])
	
	# Build WD to GID mapping
	wd_to_gid_map = build_wd_to_ncbi_map(config['references']['annotation'])
	
	# Load all sample results with gene lengths
	sample_dfs = load_sample_results(args.results_dir, gene_lengths, args.output_dir)
	
	# First analyze each sample to get gene descriptions
	samples_with_info = {}
	for sample, df in sample_dfs.items():
		print(f"Analyzing sample: {sample}")
		sample_dir = os.path.join(args.output_dir, sample)
		os.makedirs(sample_dir, exist_ok=True)
		analyze_sample(df, sample, sample_dir, cache_file, gene_lengths, gene_strands)
		df.to_csv(os.path.join(sample_dir, "results_with_percentiles.csv"), index=False)
		samples_with_info[sample] = df
		
		# Create effect coverage plots
		plot_effect_coverage_by_module(df, module_df, sample_dir, wd_to_gid_map, gene_lengths)
		
		# Create effect by expression plots
		plot_effect_by_expression(df, expression_df, sample_dir, wd_to_gid_map)
		
		# Create expression vs mutations plots
		plot_expression_vs_mutations(df, expression_df, sample_dir, wd_to_gid_map, gene_lengths)
	
	# Now do shared gene analysis
	if samples_with_info:
		print(f"\nAnalyzing shared genes between {len(samples_with_info)} samples...")
		analyze_shared_genes(samples_with_info, args.output_dir, gene_strands)
	else:
		print("\nNo samples found - skipping shared gene analysis")
	
	# Create significance counts
	create_significance_counts(samples_with_info, args.output_dir, cache_file=cache_file)
	
	# Compare scoring methods
	# for sample, df in samples_with_info.items():
	#     compare_scoring_methods(df, gene_lengths, sample, args.output_dir)  # COMMENTED OUT FOR NOW
	
	# List significant genes
	for sample, df in samples_with_info.items():
		list_significant_genes(df, sample, args.output_dir, cache_file, gene_strands)
	
	# Generate comprehensive summary tables
	generate_summary_tables(samples_with_info, args.output_dir, cache_file, gene_strands)
	
	# Create p-value distribution plots
	create_pvalue_distribution_plots(samples_with_info, args.output_dir)
	create_dn_ds_pvalue_distribution_plots(args.results_dir, args.output_dir)
	# New: diagnostics requested
	compute_global_nonsyn_fractions(args.results_dir, args.output_dir)
	compute_global_dn_ds(args.results_dir, args.output_dir)
	perform_dn_ds_analysis(args.results_dir, args.output_dir)
	perform_simple_counts_analysis(args.results_dir, args.output_dir)
	plot_per_gene_dn_ds_histograms(args.results_dir, args.output_dir)
	plot_per_gene_dn_ds_vs_deleterious_ratio(args.results_dir, args.output_dir)
	plot_codon_position_distributions(args.results_dir, config, args.output_dir)
	verify_site_level_consistency(args.results_dir, args.output_dir)
	# New: per-gene significance tables and volcano plots for dn/ds and non-syn fraction
	compute_per_gene_metric_pvalues(args.results_dir, args.output_dir, 'dn_ds')
	compute_per_gene_metric_pvalues(args.results_dir, args.output_dir, 'nonsyn_fraction')
	create_nonsyn_fraction_pvalue_distribution_plots(args.results_dir, args.output_dir)

	# New: synonymous-sites analysis and length-normalized summaries
	analyze_synonymous_sites(args.results_dir, args.output_dir, gene_lengths)
	
	# New: export per-gene observed and first-100 permutation syn/non counts (treated)
	write_gene_perm_counts_csv(args.results_dir, args.output_dir, group='treated', n_perms=100)
	
	# Generate concise publication plot for treated if available
	if 'EMS_treated' in samples_with_info:
		plot_effect_relationships_publication(samples_with_info['EMS_treated'], gene_lengths, args.output_dir)
	
	print("Analysis complete!")

if __name__ == '__main__':
	main()