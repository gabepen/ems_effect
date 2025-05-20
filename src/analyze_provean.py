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
    
    print(f"Looking up gene {ncbi_id} in cache")
    
    # Check cache first
    if str(ncbi_id) in cache:
        print(f"Found gene {ncbi_id} in cache")
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

def load_sample_results(results_dir: str, gene_lengths: Dict[str, int]) -> Dict[str, pd.DataFrame]:
    """Load all sample results and convert to DataFrames."""
    sample_dfs = {}
    
    for file in Path(results_dir).glob('*_results.json'):
        sample = file.stem.replace('_results', '')
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

            # Calculate p-value (two-tailed) for deleterious ratio
            if values.get('mutation_stats', {}).get('total_mutations', 0) > 0 and len(perm_scores) > 10:
                deleterious_ratio = values.get('mutation_stats', {}).get('deleterious_count', 0) / values.get('mutation_stats', {}).get('total_mutations', 0)
                perm_ratios = [r / values['mutation_stats']['total_mutations'] for r in values['permutation']['deleterious_counts']]
                pvalue_ratio_more, pvalue_ratio_less = calculate_deleterious_ratio_pvalues(deleterious_ratio, perm_ratios)
            else:
                deleterious_ratio = 0
                pvalue_ratio_more = None
                pvalue_ratio_less = None
            
            row = {
                'gene_id': gene_id,
                'effect_score': effect_score,
                'pvalue_more': pvalue_more,
                'pvalue_less': pvalue_less,
                'pvalue_ratio_more': pvalue_ratio_more,
                'pvalue_ratio_less': pvalue_ratio_less,
                'permutation_mean': values['permutation'].get('mean', 0),
                'permutation_std': values['permutation'].get('std', 0),
                'n_permutations': len(perm_scores),
                'deleterious_mutations': values.get('mutation_stats', {}).get('deleterious_count', 0),
                'total_mutations': values.get('mutation_stats', {}).get('total_mutations', 0),
                'deleterious_ratio': deleterious_ratio
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Calculate normalized scores using gene lengths
        df['effect_norm_len'] = df.apply(lambda x: x['effect_score'] / gene_lengths.get(x['gene_id'], 1), axis=1)
        
        # Apply FDR corrections
        valid_pvals_more = df['pvalue_more'].notna()
        if valid_pvals_more.any():
            df.loc[valid_pvals_more, 'pvalue_more_fdr'] = multipletests(df.loc[valid_pvals_more, 'pvalue_more'], method='fdr_bh')[1]
        
        valid_pvals_less = df['pvalue_less'].notna()
        if valid_pvals_less.any():
            df.loc[valid_pvals_less, 'pvalue_less_fdr'] = multipletests(df.loc[valid_pvals_less, 'pvalue_less'], method='fdr_bh')[1]
        
        valid_ratio_more = df['pvalue_ratio_more'].notna()
        if valid_ratio_more.any():
            df.loc[valid_ratio_more, 'pvalue_ratio_more_fdr'] = multipletests(df.loc[valid_ratio_more, 'pvalue_ratio_more'], method='fdr_bh')[1]
        
        valid_ratio_less = df['pvalue_ratio_less'].notna()
        if valid_ratio_less.any():
            df.loc[valid_ratio_less, 'pvalue_ratio_less_fdr'] = multipletests(df.loc[valid_ratio_less, 'pvalue_ratio_less'], method='fdr_bh')[1]
        
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

def calculate_deleterious_ratio_pvalues(observed_ratio: float, perm_ratios: List[float] ) -> pd.Series:
    """Calculate p-values based on deleterious ratio distribution.
    
    For deleterious ratios:
    - Higher ratios indicate more deleterious mutations
    - P-value is fraction of background ratios that are more extreme
    - Low p-values indicate unusually high or low ratios
    """
    if len(perm_ratios) <= 10:
        return None
    
    n_more_deleterious = sum(x > observed_ratio for x in perm_ratios)
    n_less_deleterious = sum(x < observed_ratio for x in perm_ratios)
    
    pvalue_more = n_more_deleterious / len(perm_ratios)
    pvalue_less = n_less_deleterious / len(perm_ratios)
    return pvalue_more, pvalue_less

def create_significant_genes_summary(df: pd.DataFrame, sample: str, output_dir: str, cache_file: str, pvalue_threshold: float = 0.05) -> None:
    """Create a summary CSV file of significant genes."""
    # Filter out None/NaN p-values
    df_valid = df.dropna(subset=['pvalue_more_fdr', 'pvalue_less_fdr'])
    
    # Filter for significant genes (both deleterious and protective)
    deleterious = (df_valid['pvalue_more_fdr'] < pvalue_threshold) & (df_valid['effect_score'] < 0)
    protective = (df_valid['pvalue_less_fdr'] < pvalue_threshold) & (df_valid['effect_score'] > 0)
    sig_genes = df_valid[deleterious | protective].copy()
    
    # Sort by absolute effect score (strongest effects first)
    sig_genes['abs_effect'] = abs(sig_genes['effect_score'])
    sig_genes = sig_genes.sort_values('abs_effect', ascending=False)
    
    # Select and rename columns for the summary
    summary_df = sig_genes[[
        'gene_id',
        'effect_score',
        'effect_norm_len',
        'pvalue_more',
        'pvalue_less',
        'pvalue_more_fdr',
        'pvalue_less_fdr',
        'permutation_mean',
        'permutation_std',
        'n_permutations',
        'deleterious_mutations',
        'total_mutations',
        'deleterious_ratio'
    ]].copy()
    
    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        'effect_score': 'raw_effect_score',
        'effect_norm_len': 'length_normalized_score',
        'permutation_mean': 'permutation_mean_score',
        'permutation_std': 'permutation_std_dev',
        'n_permutations': 'number_of_permutations',
        'deleterious_ratio': 'deleterious_mutation_ratio'
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
    gene_summary_df = pd.DataFrame({
        'gene_id': sig_genes['gene_id'],
        'gene_name': sig_genes.get('gene_name', ''),
        'gene_description': sig_genes.get('gene_description', ''),
        'effect_score': sig_genes['effect_score'],
        'effect_norm_len': sig_genes['effect_norm_len'],
        'deleterious_ratio': sig_genes['deleterious_ratio'],
        'pvalue_more': sig_genes['pvalue_more'],
        'pvalue_less': sig_genes['pvalue_less'],
        'pvalue_more_fdr': sig_genes['pvalue_more_fdr'],
        'pvalue_less_fdr': sig_genes['pvalue_less_fdr']
    })

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

def analyze_sample(df: pd.DataFrame, sample: str, output_dir: str, cache_file: str, gene_lengths: Dict[str, int]) -> None:
    """Analyze a single sample's results."""
    sample_dir = os.path.join(output_dir, sample)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create plots
    plot_effect_relationships(df, gene_lengths, sample, sample_dir)
    plot_effect_relationships_ratio(df, gene_lengths, sample, sample_dir)
    
    # Add method comparison
    compare_scoring_methods(df, gene_lengths, sample, sample_dir)
    
    # Create significant genes summary
    create_significant_genes_summary(df, sample, sample_dir, cache_file)
    
    # List significant genes
    list_significant_genes(df, sample, sample_dir, cache_file)

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

def analyze_shared_genes(sample_dfs: Dict[str, pd.DataFrame], output_dir: str, pvalue_threshold: float = 0.05) -> None:
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
        
        merged_data.append({
            'gene_id': gene,
            'total_mutations': total_mutations,
            'deleterious_mutations': deleterious_mutations,
            'deleterious_ratio': deleterious_mutations / total_mutations if total_mutations > 0 else 0,
            'mean_effect_score': np.mean(effect_scores),
            'mean_pvalue': np.mean(pvalues),
            'control_samples_count': control_count
        })
    
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
    plt.figure(figsize=(10, 8))
    
    # Define significance using FDR corrected p-values
    deleterious = (df['pvalue_more_fdr'] < 0.05) & (df['effect_score'] < 0)
    protective = (df['pvalue_less_fdr'] < 0.05) & (df['effect_score'] > 0)
    nonsig = ~(deleterious | protective)
    
    # Get gene lengths for plotting
    gene_lens = df['gene_id'].map(lambda x: gene_lengths.get(x, 1))
    
    # Plot: Gene Length vs Effect Score
    plt.scatter(gene_lens[nonsig], 
                df[nonsig]['effect_score'],
                alpha=0.3, color='grey', label='Non-significant')
    plt.scatter(gene_lens[deleterious],
                df[deleterious]['effect_score'],
                alpha=0.7, color='red', label='Deleterious')
    plt.scatter(gene_lens[protective],
                df[protective]['effect_score'],
                alpha=0.7, color='blue', label='Protective')
    
    plt.xlabel('Gene Length (bp)')
    plt.ylabel('Effect Score')
    plt.legend()
    
    plt.title(f'Effect Score vs Gene Length - {sample}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample}_effect_relationships.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Get genes significant by either measure (using FDR corrected p-values)
    all_genes_low_p = set()
    all_genes_high_p = set()
    all_genes_high_ratio = set()
    all_genes_low_ratio = set()
    
    for df in sample_dfs.values():
        # Effect score based (using FDR corrected p-values)
        all_genes_low_p.update(set(df[df['pvalue_more_fdr'] < 0.05]['gene_id']))
        all_genes_high_p.update(set(df[df['pvalue_less_fdr'] > (1 - 0.05)]['gene_id']))
        # Ratio based (already FDR corrected)
        all_genes_high_ratio.update(set(df[df['pvalue_ratio_more_fdr'] < 0.05]['gene_id']))
        all_genes_low_ratio.update(set(df[df['pvalue_ratio_less_fdr'] > 0.99]['gene_id']))

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
    plt.ylabel('Deleterious Mutation Ratio')
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

def plot_effect_relationships_ratio(df: pd.DataFrame, gene_lengths: Dict[str, int], sample: str, output_dir: str) -> None:
    """Create plots showing relationships between ratio and gene length."""
    plt.figure(figsize=(10, 8))
    
    # Get gene lengths for plotting
    gene_lens = df['gene_id'].map(lambda x: gene_lengths.get(x, 1))
    
    # Plot: Gene Length vs Deleterious Ratio
    plt.scatter(gene_lens[df['pvalue_ratio_more_fdr'] < 0.05], 
                df[df['pvalue_ratio_more_fdr'] < 0.05]['deleterious_ratio'],
                alpha=0.7, color='red', label='High ratio (p < 0.05)')
    plt.scatter(gene_lens[df['pvalue_ratio_less_fdr'] < 0.05],
                df[df['pvalue_ratio_less_fdr'] < 0.05]['deleterious_ratio'],
                alpha=0.7, color='blue', label='Low ratio (p < 0.05)')
    
    plt.xlabel('Gene Length (bp)')
    plt.ylabel('Deleterious Ratio')
    plt.xscale('log')
    plt.legend()
    
    plt.title(f'Deleterious Ratio vs Gene Length - {sample}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample}_ratio_relationships.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def compare_scoring_methods(df: pd.DataFrame, gene_lengths: Dict[str, int], sample: str, output_dir: str) -> None:
    """Compare different scoring methods and create visualization."""
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Filter out None/NaN p-values
    df_valid = df.dropna(subset=['pvalue_more_fdr', 'pvalue_less_fdr'])
    
    # Define significance for effect scores
    deleterious = (df_valid['pvalue_more_fdr'] < 0.05) & (df_valid['effect_score'] < 0)
    protective = (df_valid['pvalue_less_fdr'] < 0.05) & (df_valid['effect_score'] > 0)
    
    # Define significance for ratios
    high_ratio = df_valid['pvalue_ratio_more_fdr'] < 0.05
    low_ratio = df_valid['pvalue_ratio_less_fdr'] < 0.05
    
    # Create masks for different categories
    both_del = deleterious & high_ratio
    both_prot = protective & low_ratio
    effect_only_del = deleterious & ~high_ratio
    effect_only_prot = protective & ~low_ratio
    ratio_only_del = high_ratio & ~deleterious
    ratio_only_prot = low_ratio & ~protective
    nonsig = ~(deleterious | protective | high_ratio | low_ratio)
    
    # Plot: Effect Score vs Deleterious Ratio
    ax1 = plt.subplot(gs[0])
    
    # Plot non-significant points first
    ax1.scatter(df_valid[nonsig]['effect_score'], df_valid[nonsig]['deleterious_ratio'],
                alpha=0.3, color='grey', label='Non-significant', s=30)
    
    # Plot significant points
    ax1.scatter(df_valid[both_del]['effect_score'], df_valid[both_del]['deleterious_ratio'],
                alpha=0.7, color='red', label='Deleterious in both', s=50)
    ax1.scatter(df_valid[effect_only_del]['effect_score'], df_valid[effect_only_del]['deleterious_ratio'],
                alpha=0.7, color='orange', label='Effect score only', s=50)
    ax1.scatter(df_valid[ratio_only_del]['effect_score'], df_valid[ratio_only_del]['deleterious_ratio'],
                alpha=0.7, color='purple', label='Ratio only', s=50)
    
    # Plot protective points
    ax1.scatter(df_valid[both_prot]['effect_score'], df_valid[both_prot]['deleterious_ratio'],
                alpha=0.7, color='blue', label='Protective in both', s=50)
    ax1.scatter(df_valid[effect_only_prot]['effect_score'], df_valid[effect_only_prot]['deleterious_ratio'],
                alpha=0.7, color='cyan', label='Effect score protective only', s=50)
    ax1.scatter(df_valid[ratio_only_prot]['effect_score'], df_valid[ratio_only_prot]['deleterious_ratio'],
                alpha=0.7, color='magenta', label='Ratio protective only', s=50)
    
    ax1.set_xlabel('Effect Score')
    ax1.set_ylabel('Deleterious Mutation Ratio')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Calculate correlations
    score_ratio_corr = df_valid['effect_score'].corr(df_valid['deleterious_ratio'])
    norm_score_ratio_corr = df_valid['effect_norm_len'].corr(df_valid['deleterious_ratio'])
    pval_corr = df_valid['pvalue_more_fdr'].corr(df_valid['pvalue_ratio_more_fdr'])
    
    # Add correlation text
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    correlation_text = (
        f'Correlations:\n'
        f'Effect Score vs Ratio: {score_ratio_corr:.3f}\n'
        f'Normalized Score vs Ratio: {norm_score_ratio_corr:.3f}\n'
        f'Effect p-value vs Ratio p-value: {pval_corr:.3f}'
    )
    ax2.text(0.5, 0.5, correlation_text, ha='center', va='center')
    
    plt.suptitle(f'Scoring Method Comparison - {sample}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample}_method_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_volcano_ratio(df: pd.DataFrame, sample: str, output_dir: str) -> None:
    """Create volcano plot using deleterious ratio p-values."""
    plt.figure(figsize=(10, 8))
    
    # Filter out None/NaN p-values
    df_valid = df.dropna(subset=['pvalue_ratio_more_fdr'])
    
    # Define significance using FDR corrected p-values
    median_ratio = np.median(df_valid['deleterious_ratio'])
    high_ratio = (df_valid['pvalue_ratio_more_fdr'] < 0.05) & (df_valid['deleterious_ratio'] > median_ratio)
    low_ratio = (df_valid['pvalue_ratio_more_fdr'] < 0.05) & (df_valid['deleterious_ratio'] < median_ratio)
    nonsig = ~(high_ratio | low_ratio)
    
    # Plot points
    plt.scatter(df_valid[nonsig]['gene_len'], 
               df_valid[nonsig]['deleterious_ratio'],
               alpha=0.5, color='grey', label='Non-significant')
    
    plt.scatter(df_valid[high_ratio]['gene_len'],
                df_valid[high_ratio]['deleterious_ratio'],
                alpha=0.7, color='red', label='High ratio (p < 0.05)')
    
    plt.scatter(df_valid[low_ratio]['gene_len'],
                df_valid[low_ratio]['deleterious_ratio'],
                alpha=0.7, color='blue', label='Low ratio (p > 0.99)')
    
    plt.xlabel('Gene Length (bp)')
    plt.ylabel('Deleterious Mutation Ratio')
    plt.title(f'Deleterious Ratio Relationships - {sample}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample}_ratio_relationships.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def list_significant_genes(df: pd.DataFrame, sample: str, output_dir: str, cache_file: str = None) -> None:
    """List genes with significant protective and deleterious FDR-corrected p-values."""
    # Filter out None/NaN p-values
    df_valid = df.dropna(subset=['pvalue_more_fdr', 'pvalue_less_fdr'])
    
    # Filter for significant deleterious and protective genes
    significant_deleterious = df_valid[(df_valid['pvalue_more_fdr'] < 0.05) & (df_valid['effect_score'] < 0)]
    significant_protective = df_valid[(df_valid['pvalue_less_fdr'] < 0.05) & (df_valid['effect_score'] > 0)]
    
    # Also get ratio-based significant genes
    ratio_high = df_valid[df_valid['pvalue_ratio_more_fdr'] < 0.05]
    ratio_low = df_valid[df_valid['pvalue_ratio_less_fdr'] < 0.05]
    
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
        
        # Select and reorder columns
        columns = [
            'gene_id', 
            'gene_description', 
            'effect_score',
            'pvalue_more_fdr', 
            'pvalue_less_fdr',
            'deleterious_mutations',
            'total_mutations',
            'deleterious_ratio'
        ]
        
        return result_df[columns]
    
    # Process each dataframe
    del_df = add_gene_info(significant_deleterious)
    prot_df = add_gene_info(significant_protective)
    high_ratio_df = add_gene_info(ratio_high)
    low_ratio_df = add_gene_info(ratio_low)
    
    # Save to CSV
    deleterious_file = os.path.join(output_dir, f"{sample}_significant_deleterious_genes.csv")
    protective_file = os.path.join(output_dir, f"{sample}_significant_protective_genes.csv")
    ratio_high_file = os.path.join(output_dir, f"{sample}_significant_high_ratio_genes.csv")
    ratio_low_file = os.path.join(output_dir, f"{sample}_significant_low_ratio_genes.csv")
    
    del_df.to_csv(deleterious_file, index=False)
    prot_df.to_csv(protective_file, index=False)
    high_ratio_df.to_csv(ratio_high_file, index=False)
    low_ratio_df.to_csv(ratio_low_file, index=False)
    
    print(f"\nSignificant genes for {sample}:")
    print(f"Deleterious genes: {len(significant_deleterious)}")
    print(f"Protective genes: {len(significant_protective)}")
    print(f"High ratio genes: {len(ratio_high)}")
    print(f"Low ratio genes: {len(ratio_low)}")

def generate_summary_tables(sample_dfs: Dict[str, pd.DataFrame], output_dir: str, cache_file: str = None) -> None:
    """Generate comprehensive summary tables of significant genes."""
    os.makedirs(os.path.join(output_dir, "summary_tables"), exist_ok=True)
    summary_dir = os.path.join(output_dir, "summary_tables")
    
    # Load gene info cache
    gene_info_cache = load_gene_info_cache(cache_file) if cache_file else {}
    
    # Only consider treated samples (exclude controls, e.g., those with 'NT' in the name)
    treated_sample_dfs = {sample: df for sample, df in sample_dfs.items() if 'NT' not in sample}

    multi_sample_deleterious = {}
    multi_sample_neutral = {}
    multi_sample_ratio_high = {}
    multi_sample_ratio_low = {}

    for sample, df in treated_sample_dfs.items():
        for gene_id in df[(df['pvalue_more_fdr'] < 0.05)]['gene_id']:
            multi_sample_deleterious.setdefault(gene_id, []).append(sample)
        for gene_id in df[(df['pvalue_less_fdr'] < 0.05)]['gene_id']:
            multi_sample_neutral.setdefault(gene_id, []).append(sample)
        for gene_id in df[df['pvalue_ratio_more_fdr'] < 0.05]['gene_id']:
            multi_sample_ratio_high.setdefault(gene_id, []).append(sample)
        for gene_id in df[df['pvalue_ratio_less_fdr'] < 0.05]['gene_id']:
            multi_sample_ratio_low.setdefault(gene_id, []).append(sample)

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
    write_multi_sample_table(
        multi_sample_ratio_high, "multi_sample_high_ratio.csv", gene_info_cache, treated_sample_dfs,
        value_fields=['deleterious_ratio', 'pvalue_ratio_more_fdr']
    )
    write_multi_sample_table(
        multi_sample_ratio_low, "multi_sample_low_ratio.csv", gene_info_cache, treated_sample_dfs,
        value_fields=['deleterious_ratio', 'pvalue_ratio_less_fdr']
    )

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
        
        # Filter out NaN values
        df_valid = df.dropna(subset=['pvalue_more', 'pvalue_less', 'pvalue_ratio_more', 'pvalue_ratio_less'])
        
        # Add p-values to the appropriate lists
        target_dict['effect_more'].extend(df_valid['pvalue_more'].values)
        target_dict['effect_less'].extend(df_valid['pvalue_less'].values)
        target_dict['ratio_more'].extend(df_valid['pvalue_ratio_more'].values)
        target_dict['ratio_less'].extend(df_valid['pvalue_ratio_less'].values)
        
        # Add FDR corrected p-values
        if 'pvalue_more_fdr' in df_valid.columns:
            target_dict['effect_more_fdr'].extend(df_valid['pvalue_more_fdr'].values)
        if 'pvalue_less_fdr' in df_valid.columns:
            target_dict['effect_less_fdr'].extend(df_valid['pvalue_less_fdr'].values)
        if 'pvalue_ratio_more_fdr' in df_valid.columns:
            target_dict['ratio_more_fdr'].extend(df_valid['pvalue_ratio_more_fdr'].values)
        if 'pvalue_ratio_less_fdr' in df_valid.columns:
            target_dict['ratio_less_fdr'].extend(df_valid['pvalue_ratio_less_fdr'].values)
    
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

def load_gene_lengths(gff_path: str) -> Dict[str, int]:
    """Load gene lengths from GFF file into a lookup dictionary."""
    gene_lengths = {}
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
                
                # Extract gene ID from attributes
                attrs = dict(attr.split('=') for attr in parts[8].split(';') if '=' in attr)
                dbxref = attrs.get('Dbxref', '')
                if 'GeneID:' in dbxref:
                    gene_id = dbxref.split('GeneID:')[1]
                    gene_lengths[gene_id] = length
                    
        return gene_lengths
    except Exception as e:
        print(f"Error loading GFF file: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Analyze PROVEAN score results')
    parser.add_argument('results_dir', help='Directory containing sample results JSON files')
    parser.add_argument('output_dir', help='Directory to save analysis results')
    parser.add_argument('-c', '--config', required=True, help='Path to config file with reference paths')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load gene lengths from GFF
    gene_lengths = load_gene_lengths(config['references']['annotation'])
    
    # Get cache path from config
    cache_file = config['references']['gene_info_cache']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all sample results with gene lengths
    sample_dfs = load_sample_results(args.results_dir, gene_lengths)
    
    # First analyze each sample to get gene descriptions
    samples_with_info = {}
    for sample, df in sample_dfs.items():
        print(f"Analyzing sample: {sample}")
        sample_dir = os.path.join(args.output_dir, sample)
        os.makedirs(sample_dir, exist_ok=True)
        analyze_sample(df, sample, sample_dir, cache_file, gene_lengths)
        samples_with_info[sample] = df
    
    # Now do shared gene analysis
    if samples_with_info:
        print(f"\nAnalyzing shared genes between {len(samples_with_info)} samples...")
        analyze_shared_genes(samples_with_info, args.output_dir)
    else:
        print("\nNo samples found - skipping shared gene analysis")
    
    # Create significance counts
    create_significance_counts(samples_with_info, args.output_dir, cache_file=cache_file)
    
    # Compare scoring methods
    for sample, df in samples_with_info.items():
        compare_scoring_methods(df, gene_lengths, sample, args.output_dir)
    
    # List significant genes
    for sample, df in samples_with_info.items():
        list_significant_genes(df, sample, args.output_dir, cache_file)
    
    # Generate comprehensive summary tables
    generate_summary_tables(samples_with_info, args.output_dir, cache_file)
    
    # Create p-value distribution plots
    create_pvalue_distribution_plots(samples_with_info, args.output_dir)
    
    print("Analysis complete!")

if __name__ == '__main__':
    main()