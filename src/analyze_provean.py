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
    
    
def load_sample_results(results_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all sample results and convert to DataFrames."""
    sample_dfs = {}
    
    for file in Path(results_dir).glob('*_results.json'):
        sample = file.stem.replace('_results', '')
        with open(file) as f:
            data = json.load(f)
            
        # Convert to DataFrame
        rows = []
        for gene_id, values in data.items():
            # Calculate p-value from permutations
            perm_scores = values['permutation']['scores']
            effect_score = values['effect']
            
            # Handle case where there are no permutation scores
            if not perm_scores:
                pvalue = 1.0  # Set to 1.0 when no permutations available
            else:
                # Calculate empirical p-value (two-tailed)
                more_extreme = sum(abs(score) >= abs(effect_score) for score in perm_scores)
                pvalue = more_extreme / len(perm_scores)
            
            row = {
                'gene_id': gene_id,
                'effect_score': effect_score,
                'gene_len': values['gene_len'],
                'avg_cov': values['avg_cov'],
                'pvalue': pvalue,
                'permutation_mean': values['permutation'].get('mean', 0),
                'permutation_std': values['permutation'].get('std', 0),
                'permutation_percentile': values['permutation'].get('percentile', 0),
                'n_permutations': len(perm_scores),
                'deleterious_mutations': values.get('mutation_stats', {}).get('deleterious_count', 0),
                'total_mutations': values.get('mutation_stats', {}).get('total_mutations', 0)
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Calculate normalized scores
        df['effect_norm_len'] = df['effect_score'] / df['gene_len']
        df['effect_norm_full'] = df['effect_norm_len'] / df['avg_cov'].replace(0, 1)
        
        # Calculate -log10(pvalue)
        df['-log10_pvalue'] = -np.log10(df['pvalue'].clip(1e-10, 1))
        
        # Add deleterious ratio calculation
        df['deleterious_ratio'] = df['deleterious_mutations'] / df['total_mutations']
        df['deleterious_ratio'] = df['deleterious_ratio'].fillna(0)  # Handle div by zero
        
        # Calculate normalized deleterious ratios
        df['deleterious_ratio_norm_len'] = df['deleterious_ratio'] / df['gene_len']
        df['deleterious_ratio_norm_full'] = df['deleterious_ratio_norm_len'] / df['avg_cov'].replace(0, 1)
        
        sample_dfs[sample] = df
        
        # Log summary of permutation counts
        print(f"\nSample {sample}:")
        print(f"Total genes: {len(df)}")
        print(f"Genes with no permutations: {sum(df['n_permutations'] == 0)}")
        print(f"Average permutations per gene: {df['n_permutations'].mean():.1f}")
        
    return sample_dfs

def plot_volcano(df: pd.DataFrame, title: str, score_col: str = 'effect_score', output_dir: str = None, sample: str = None) -> None:
    """Create volcano plot of effect scores vs -log10(pvalue)."""
    plt.figure(figsize=(10, 8))
    
    # Calculate significance threshold (e.g., p < 0.05)
    sig_threshold = -np.log10(0.05)
    
    # Create scatter plot
    plt.scatter(df[score_col], df['-log10_pvalue'], 
               alpha=0.6, s=50)
    
    # Add threshold line
    plt.axhline(y=sig_threshold, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Effect Score')
    plt.ylabel('-log10(p-value)')
    plt.title(title)
    plt.tight_layout()
    
    # Export plot data if output directory is provided
    if output_dir and sample:
        # Create DataFrame with plot data
        columns = ['gene_id', score_col, 'pvalue', '-log10_pvalue', 'gene_len', 'avg_cov']
        
        # Add optional columns if they exist
        if 'deleterious_mutations' in df.columns:
            columns.append('deleterious_mutations')
        if 'total_mutations' in df.columns:
            columns.append('total_mutations')
            
        plot_data = df[columns].copy()
        
        # Sort by significance (p-value)
        plot_data = plot_data.sort_values('pvalue')
        
        # Save to CSV
        plot_file = os.path.join(output_dir, f"{sample}_{score_col}_volcano_data.csv")
        plot_data.to_csv(plot_file, index=False)

def plot_pvalue_density(df: pd.DataFrame, title: str) -> None:
    """Create density plot of p-values."""
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(data=df['pvalue'], fill=True)
    plt.axvline(x=0.05, color='r', linestyle='--', alpha=0.5,
                label='p=0.05 threshold')
    
    plt.xlabel('p-value')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def create_significant_genes_summary(df: pd.DataFrame, sample: str, output_dir: str, cache_file: str, pvalue_threshold: float = 0.05) -> None:
    """Create a summary CSV file of significant genes."""
    # Filter for significant genes
    sig_genes = df[df['pvalue'] < pvalue_threshold].copy()
    
    # Sort by absolute effect score (strongest effects first)
    sig_genes['abs_effect'] = abs(sig_genes['effect_score'])
    sig_genes = sig_genes.sort_values('abs_effect', ascending=False)
    
    # Select and rename columns for the summary
    summary_df = sig_genes[[
        'gene_id',
        'effect_score',
        'effect_norm_len',
        'effect_norm_full',
        'pvalue',
        'gene_len',
        'avg_cov',
        'permutation_mean',
        'permutation_std',
        'n_permutations',
        'deleterious_mutations',
        'total_mutations',
        'deleterious_ratio',
        'deleterious_ratio_norm_len',
        'deleterious_ratio_norm_full'
    ]].copy()
    
    # Add fold change from permutation mean
    summary_df['fold_change'] = summary_df['effect_score'] / summary_df['permutation_mean'].replace(0, np.nan)
    
    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        'effect_score': 'raw_effect_score',
        'effect_norm_len': 'length_normalized_score',
        'effect_norm_full': 'fully_normalized_score',
        'gene_len': 'gene_length',
        'avg_cov': 'average_coverage',
        'permutation_mean': 'permutation_mean_score',
        'permutation_std': 'permutation_std_dev',
        'n_permutations': 'number_of_permutations',
        'deleterious_ratio': 'deleterious_mutation_ratio',
        'deleterious_ratio_norm_len': 'length_normalized_deleterious_ratio',
        'deleterious_ratio_norm_full': 'fully_normalized_deleterious_ratio'
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
        'gene_length': sig_genes['gene_len'],
        'average_coverage': sig_genes['avg_cov'],
        'total_mutations': sig_genes['total_mutations'],
        'effect_score': sig_genes['effect_score'],
        'effect_score_norm_len': sig_genes['effect_norm_len'],
        'deleterious_ratio': sig_genes['deleterious_ratio'],
        'pvalue': sig_genes['pvalue']
    })

    # Sort by effect score (most negative first)
    gene_summary_df = gene_summary_df.sort_values('effect_score_norm_len')

    # Save gene info summary
    gene_info_file = os.path.join(output_dir, f"{sample}_gene_info_summary.csv")
    gene_summary_df.to_csv(gene_info_file, index=False)
    
    # Print summary
    print(f"\nSignificant genes summary for {sample}:")
    print(f"Total significant genes: {len(sig_genes)}")
    print(f"Strongest effect: {sig_genes['effect_score'].min():.2f}")
    print(f"Summary saved to: {output_file}")

def calculate_deleterious_ratio_pvalues(df: pd.DataFrame) -> pd.Series:
    """Calculate p-values based on deleterious ratio distribution."""
    # Get background distribution from all genes
    background_ratios = df['deleterious_ratio'].values
    
    # Calculate p-values for each gene
    pvalues = []
    for ratio in df['deleterious_ratio']:
        # Calculate what fraction of background ratios are more extreme
        # For deleterious ratio, higher values are more deleterious
        pvalue = np.mean(background_ratios <= ratio)
        pvalues.append(pvalue)
    
    return pd.Series(pvalues, index=df.index)

def analyze_sample(df: pd.DataFrame, sample: str, output_dir: str, cache_file: str) -> None:
    """Analyze a single sample's results."""
    sample_dir = os.path.join(output_dir, sample)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Add deleterious ratio based p-values
    df['ratio_pvalue'] = calculate_deleterious_ratio_pvalues(df)
    
    # Create both versions of plots
    plot_volcano(df, f'{sample} - Effect Score vs Significance', 
                score_col='effect_score',
                output_dir=sample_dir,
                sample=sample)
    plot_volcano_ratio(df, sample, sample_dir)  # New ratio-based volcano
    plot_effect_relationships(df, sample, sample_dir)
    plot_effect_relationships_ratio(df, sample, sample_dir)  # New ratio-based relationships
    
    # Create significant genes summary with cache file
    create_significant_genes_summary(df, sample, sample_dir, cache_file)
    
    # Update summary statistics with explicit type conversion
    stats_dict = {
        'total_genes': int(len(df)),
        'significant_genes': int(sum(df['pvalue'] < 0.05)),
        'mean_effect': float(df['effect_score'].mean()),
        'median_effect': float(df['effect_score'].median()),
        'mean_pvalue': float(df['pvalue'].mean()),
        'median_pvalue': float(df['pvalue'].median()),
        'mean_deleterious_ratio': float(df['deleterious_ratio'].mean()),
        'median_deleterious_ratio': float(df['deleterious_ratio'].median()),
        'genes_with_deleterious_mutations': int(sum(df['deleterious_mutations'] > 0)),
        'total_deleterious_mutations': int(df['deleterious_mutations'].sum()),
        'total_mutations': int(df['total_mutations'].sum())
    }
    
    with open(os.path.join(sample_dir, 'summary_stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)

def plot_combined_pvalue_density(sample_dfs: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Create combined histogram plots of p-values for all samples."""
    
    # First plot - Control vs Treated
    plt.figure(figsize=(12, 8))
    
    # Collect p-values for controls and treated samples
    control_pvals = []
    treated_pvals = []
    
    for sample, df in sample_dfs.items():
        if 'NT' in sample:
            control_pvals.extend(df['pvalue'].values)
        else:
            treated_pvals.extend(df['pvalue'].values)
    
    # Create histogram
    bins = np.linspace(0, 1, 51)  # 50 bins from 0 to 1
    plt.hist([control_pvals, treated_pvals], 
             bins=bins,
             label=['Control', 'EMS Treated'],
             stacked=True,
             alpha=0.7,
             color=['lightgrey', 'darkorange'])
    
    # Add significance threshold line
    plt.axvline(x=0.05, color='r', linestyle='--', alpha=0.5,
                label='p=0.05 threshold')
    
    plt.xlabel('p-value')
    plt.ylabel('Count')
    plt.title('P-value Distribution Across Samples')
    plt.legend()
    plt.tight_layout()
    
    # Save first plot
    plt.savefig(os.path.join(output_dir, 'combined_pvalue_histogram.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Second plot - Controls with unique colors
    plt.figure(figsize=(12, 8))
    
    # Define color palette for controls
    control_colors = sns.color_palette("Set2", n_colors=10)  # Adjust n_colors as needed
    treated_color = 'darkorange'
    
    # Collect p-values for each sample separately
    sample_pvals = {}
    for sample, df in sample_dfs.items():
        sample_pvals[sample] = df['pvalue'].values
    
    # Separate control and treated samples
    control_samples = [s for s in sample_pvals.keys() if 'NT' in s]
    treated_samples = [s for s in sample_pvals.keys() if 'NT' not in s]
    
    # Create lists for stacked histogram
    all_pvals = []
    labels = []
    colors = []
    
    # Add control samples first
    for i, sample in enumerate(control_samples):
        all_pvals.append(sample_pvals[sample])
        labels.append(f'Control - {sample}')
        colors.append(control_colors[i % len(control_colors)])
    
    # Combine all treated samples
    treated_combined = np.concatenate([sample_pvals[s] for s in treated_samples])
    all_pvals.append(treated_combined)
    labels.append('EMS Treated')
    colors.append(treated_color)
    
    # Create stacked histogram
    plt.hist(all_pvals,
             bins=bins,
             label=labels,
             stacked=True,
             alpha=0.7,
             color=colors)
    
    # Add significance threshold line
    plt.axvline(x=0.05, color='r', linestyle='--', alpha=0.5,
                label='p=0.05 threshold')
    
    plt.xlabel('p-value')
    plt.ylabel('Count')
    plt.title('P-value Distribution Across Samples (Unique Control Colors)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save second plot
    plt.savefig(os.path.join(output_dir, 'combined_pvalue_histogram_unique_controls.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

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
        sig_genes = set(df[df['pvalue'] < pvalue_threshold]['gene_id'])
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
            pvalues.append(gene_row['pvalue'])
        
        # Count control samples where this gene is significant
        control_count = sum(1 for control in control_samples 
                          if gene in set(sample_dfs[control][sample_dfs[control]['pvalue'] < pvalue_threshold]['gene_id']))
        
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

def plot_effect_relationships(df: pd.DataFrame, sample: str, output_dir: str) -> None:
    """Create multi-panel scatter plot of effect score relationships."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define significance masks
    deleterious = df['pvalue'] < 0.05
    protective = df['pvalue'] > 0.95
    nonsig = ~(deleterious | protective)
    
    # Plot 1: Effect score vs gene length
    # Plot non-significant points first
    ax1.scatter(df[nonsig]['gene_len'], 
               df[nonsig]['effect_score'],
               alpha=0.5, color='grey', label='Non-significant')
    
    # Plot deleterious points
    ax1.scatter(df[deleterious]['gene_len'],
                df[deleterious]['effect_score'],
                alpha=0.7, color='red', label='Deleterious (p < 0.05)')
    
    # Plot protective points
    ax1.scatter(df[protective]['gene_len'],
                df[protective]['effect_score'],
                alpha=0.7, color='blue', label='Protective (p > 0.95)')
    
    ax1.set_xlabel('Gene Length (bp)')
    ax1.set_ylabel('Effect Score')
    ax1.set_xscale('log')
    ax1.legend()
    
    # Plot 2: Effect score vs average coverage
    # Plot non-significant points first
    ax2.scatter(df[nonsig]['avg_cov'],
                df[nonsig]['effect_score'],
                alpha=0.5, color='grey', label='Non-significant')
    
    # Plot deleterious points
    ax2.scatter(df[deleterious]['avg_cov'],
                df[deleterious]['effect_score'],
                alpha=0.7, color='red', label='Deleterious (p < 0.05)')
    
    # Plot protective points
    ax2.scatter(df[protective]['avg_cov'],
                df[protective]['effect_score'],
                alpha=0.7, color='blue', label='Protective (p > 0.95)')
    
    ax2.set_xlabel('Average Coverage')
    ax2.set_ylabel('Effect Score')
    ax2.set_xscale('log')
    ax2.legend()
    
    plt.suptitle(f'Effect Score Relationships - {sample}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample}_effect_relationships.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_significance_counts(sample_dfs: Dict[str, pd.DataFrame], output_dir: str, pvalue_threshold: float = 0.05, cache_file: str = None) -> None:
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
    
    # Add ratio-based p-values to each sample
    for sample, df in sample_dfs.items():
        df['ratio_pvalue'] = calculate_deleterious_ratio_pvalues(df)
    
    # Get genes significant by either measure
    all_genes_low_p = set()
    all_genes_high_p = set()
    all_genes_high_ratio = set()
    all_genes_low_ratio = set()
    
    for df in sample_dfs.values():
        # Effect score based
        all_genes_low_p.update(set(df[df['pvalue'] < pvalue_threshold]['gene_id']))
        all_genes_high_p.update(set(df[df['pvalue'] > (1 - pvalue_threshold)]['gene_id']))
        # Ratio based
        all_genes_high_ratio.update(set(df[df['ratio_pvalue'] < pvalue_threshold]['gene_id']))
        all_genes_low_ratio.update(set(df[df['ratio_pvalue'] > (1 - pvalue_threshold)]['gene_id']))
    
    # Process genes for both measures
    effect_deleterious = process_genes(all_genes_low_p, high_p=False, use_ratio=False, pvalue_threshold=pvalue_threshold)
    effect_protective = process_genes(all_genes_high_p, high_p=True, use_ratio=False, pvalue_threshold=pvalue_threshold)
    ratio_high = process_genes(all_genes_high_ratio, high_p=False, use_ratio=True, pvalue_threshold=pvalue_threshold)
    ratio_low = process_genes(all_genes_low_ratio, high_p=True, use_ratio=True, pvalue_threshold=pvalue_threshold)
    
    # Save all results
    for counts, suffix in [
        (effect_deleterious, 'effect_deleterious'),
        (effect_protective, 'effect_protective'),
        (ratio_high, 'ratio_high'),
        (ratio_low, 'ratio_low')
    ]:
        if counts:
            df = pd.DataFrame(counts)
            
            # Sort differently for deleterious vs protective
            if suffix == 'effect_deleterious' or suffix == 'effect_protective':
                df = df.sort_values(['treated_samples_significant', 'mean_pvalue'], 
                                  ascending=[False, True])  # Lower p-values first for deleterious
            else:
                df = df.sort_values(['treated_samples_significant', 'mean_pvalue'], 
                                  ascending=[False, False])  # Higher p-values first for protective
            
            # For protective genes, look up any missing gene info
            if suffix == 'effect_protective':
                print("\nLooking up missing gene info for protective genes...")
                updated_genes = 0
                for idx, row in df.iterrows():
                    gene_id = str(row['gene_id'])
                    if gene_id not in gene_info or not gene_info[gene_id].get('description'):
                        try:
                            gene_data = lookup_gene_info(gene_id)
                            if gene_data and gene_data.get('description'):
                                gene_info[gene_id] = gene_data
                                df.at[idx, 'gene_description'] = gene_data['description']
                                updated_genes += 1
                                print(f"Updated gene {gene_id}: {gene_data['description'][:100]}...")
                        except Exception as e:
                            print(f"Error looking up gene {gene_id}: {e}")
                
                if updated_genes > 0:
                    print(f"Updated information for {updated_genes} protective genes")
                    # Save updated cache
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(gene_info, f, indent=2)
                        print("Updated gene info cache saved")
                    except Exception as e:
                        print(f"Error saving updated cache: {e}")
            
            # Save to CSV
            output_file = os.path.join(output_dir, f'gene_significance_counts_{suffix}.csv')
            df.to_csv(output_file, index=False)
            print(f"\nSaved {suffix} gene counts to: {output_file}")
            print(f"Total {suffix} genes found: {len(df)}")
            
            # Create merged mutation plot
            plot_merged_significance(df, output_dir, suffix)
            
            # Print summary of distribution
            print(f"\nDistribution of {suffix} genes by number of treated samples:")
            treated_dist = df['treated_samples_significant'].value_counts().sort_index()
            for n_samples, count in treated_dist.items():
                print(f"Significant in {n_samples} treated samples: {count} genes")

def plot_merged_significance(df: pd.DataFrame, output_dir: str, analysis_type: str = 'deleterious') -> None:
    """Create scatter plot of merged mutation data colored by significance count."""
    
    plt.figure(figsize=(12, 8))
    
    # Create colormap for number of treated samples
    max_treated = df['treated_samples_significant'].max()
    norm = plt.Normalize(1, max_treated)
    cmap = plt.cm.viridis
    
    # Create scatter plot
    scatter = plt.scatter(df['total_mutations'], 
                         df['deleterious_ratio'],
                         c=df['treated_samples_significant'],
                         cmap=cmap,
                         norm=norm,
                         alpha=0.6,
                         s=50)
    
    plt.xlabel('Total Mutations (All Treated Samples)')
    plt.ylabel('Deleterious Mutation Ratio')
    title_prefix = 'Protective' if analysis_type == 'protective' else 'Deleterious'
    plt.title(f'{title_prefix} Mutation Analysis Across Treated Samples')
    plt.xscale('log')  # Log scale for total mutations
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Treated Samples Significant In')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'merged_mutation_significance_{analysis_type}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_volcano_ratio(df: pd.DataFrame, sample: str, output_dir: str) -> None:
    """Create volcano plot using deleterious ratio p-values."""
    plt.figure(figsize=(10, 8))
    
    # Define significance masks
    deleterious = df['ratio_pvalue'] < 0.05
    protective = df['ratio_pvalue'] > 0.95
    nonsig = ~(deleterious | protective)
    
    # Plot points
    plt.scatter(df[nonsig]['deleterious_ratio'], 
               -np.log10(df[nonsig]['ratio_pvalue']),
               alpha=0.5, color='grey', label='Non-significant')
    
    plt.scatter(df[deleterious]['deleterious_ratio'],
                -np.log10(df[deleterious]['ratio_pvalue']),
                alpha=0.7, color='red', label='High ratio (p < 0.05)')
    
    plt.scatter(df[protective]['deleterious_ratio'],
                -np.log10(df[protective]['ratio_pvalue']),
                alpha=0.7, color='blue', label='Low ratio (p > 0.95)')
    
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Deleterious Mutation Ratio')
    plt.ylabel('-log10(p-value)')
    plt.title(f'Volcano Plot (Ratio-based) - {sample}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volcano_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_effect_relationships_ratio(df: pd.DataFrame, sample: str, output_dir: str) -> None:
    """Create relationship plots using deleterious ratio p-values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define significance masks
    deleterious = df['ratio_pvalue'] < 0.05
    protective = df['ratio_pvalue'] > 0.95
    nonsig = ~(deleterious | protective)
    
    # Plot 1: Deleterious ratio vs gene length
    ax1.scatter(df[nonsig]['gene_len'], 
               df[nonsig]['deleterious_ratio'],
               alpha=0.5, color='grey', label='Non-significant')
    
    ax1.scatter(df[deleterious]['gene_len'],
                df[deleterious]['deleterious_ratio'],
                alpha=0.7, color='red', label='High ratio (p < 0.05)')
    
    ax1.scatter(df[protective]['gene_len'],
                df[protective]['deleterious_ratio'],
                alpha=0.7, color='blue', label='Low ratio (p > 0.95)')
    
    ax1.set_xlabel('Gene Length (bp)')
    ax1.set_ylabel('Deleterious Mutation Ratio')
    ax1.set_xscale('log')
    ax1.legend()
    
    # Plot 2: Deleterious ratio vs average coverage
    ax2.scatter(df[nonsig]['avg_cov'],
                df[nonsig]['deleterious_ratio'],
                alpha=0.5, color='grey', label='Non-significant')
    
    ax2.scatter(df[deleterious]['avg_cov'],
                df[deleterious]['deleterious_ratio'],
                alpha=0.7, color='red', label='High ratio (p < 0.05)')
    
    ax2.scatter(df[protective]['avg_cov'],
                df[protective]['deleterious_ratio'],
                alpha=0.7, color='blue', label='Low ratio (p > 0.95)')
    
    ax2.set_xlabel('Average Coverage')
    ax2.set_ylabel('Deleterious Mutation Ratio')
    ax2.set_xscale('log')
    ax2.legend()
    
    plt.suptitle(f'Deleterious Ratio Relationships - {sample}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample}_ratio_relationships.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def process_genes(gene_set: set, high_p: bool = False, use_ratio: bool = False, pvalue_threshold: float = 0.05) -> List[Dict]:
    """Process a set of genes to get their statistics across samples."""
    gene_counts = []
    comparison = operator.gt if high_p else operator.lt
    p_threshold = 1 - pvalue_threshold if high_p else pvalue_threshold
    
    for gene in gene_set:
        treated_count = 0
        control_count = 0
        total_mutations = 0
        deleterious_mutations = 0
        effect_scores = []
        pvalues = []
        ratio_pvalues = []
        
        # Check treated samples
        for sample in treated_samples:
            df = sample_dfs[sample]
            gene_rows = df[df['gene_id'] == gene]
            if len(gene_rows) > 0:
                gene_row = gene_rows.iloc[0]
                # Use either ratio p-value or effect score p-value
                p_val = gene_row['ratio_pvalue'] if use_ratio else gene_row['pvalue']
                if comparison(p_val, p_threshold):
                    treated_count += 1
                total_mutations += gene_row['total_mutations']
                deleterious_mutations += gene_row['deleterious_mutations']
                effect_scores.append(gene_row['effect_score'])
                pvalues.append(gene_row['pvalue'])
                ratio_pvalues.append(gene_row['ratio_pvalue'])
        
        # Check control samples
        for sample in control_samples:
            df = sample_dfs[sample]
            gene_rows = df[df['gene_id'] == gene]
            if len(gene_rows) > 0:
                gene_row = gene_rows.iloc[0]
                p_val = gene_row['ratio_pvalue'] if use_ratio else gene_row['pvalue']
                if comparison(p_val, p_threshold):
                    control_count += 1
        
        # Calculate averages
        mean_effect = np.mean(effect_scores) if effect_scores else 0
        mean_pvalue = np.mean(ratio_pvalues if use_ratio else pvalues) if pvalues else 1.0
        deleterious_ratio = deleterious_mutations / total_mutations if total_mutations > 0 else 0
        
        # Get gene description
        gene_description = gene_info.get(str(gene), {}).get('description', '')
        
        gene_counts.append({
            'gene_id': gene,
            'gene_description': gene_description,
            'treated_samples_significant': treated_count,
            'control_samples_significant': control_count,
            'total_samples_significant': treated_count + control_count,
            'total_mutations': total_mutations,
            'deleterious_mutations': deleterious_mutations,
            'deleterious_ratio': deleterious_ratio,
            'mean_effect_score': mean_effect,
            'mean_pvalue': mean_pvalue
        })
    return gene_counts

def main():
    parser = argparse.ArgumentParser(description='Analyze PROVEAN score results')
    parser.add_argument('results_dir', help='Directory containing sample results JSON files')
    parser.add_argument('output_dir', help='Directory to save analysis results')
    parser.add_argument('-c', '--config', required=True, help='Path to config file with reference paths')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Get cache path from config
    cache_file = config['references']['gene_info_cache']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all sample results
    sample_dfs = load_sample_results(args.results_dir)
    
    # Create combined p-value distribution plot
    plot_combined_pvalue_density(sample_dfs, args.output_dir)
    
    # First analyze each sample to get gene descriptions
    samples_with_info = {}
    for sample, df in sample_dfs.items():
        print(f"Analyzing sample: {sample}")
        sample_dir = os.path.join(args.output_dir, sample)
        os.makedirs(sample_dir, exist_ok=True)
        analyze_sample(df, sample, sample_dir, cache_file)
        samples_with_info[sample] = df  # Store the original DataFrame
    
    # Now do shared gene analysis
    if samples_with_info:
        print(f"\nAnalyzing shared genes between {len(samples_with_info)} samples...")
        analyze_shared_genes(samples_with_info, args.output_dir)
    else:
        print("\nNo samples found - skipping shared gene analysis")
    
    # Create significance counts
    create_significance_counts(samples_with_info, args.output_dir, cache_file=cache_file)
    
    print("Analysis complete!")

if __name__ == '__main__':
    main()