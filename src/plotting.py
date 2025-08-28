import json
import argparse
import matplotlib.pyplot as plt
from glob import glob
import os
import csv
import seaborn as sns
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import fisher_exact
from collections import defaultdict
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def count_mutations_unique_sites(json_files: list) -> dict:
    '''Count unique mutation sites across all samples.
    
    Args:
        json_files (list): List of paths to mutation JSON files
    
    Returns:
        dict: Dictionary of mutation frequencies per sample (counting unique sites)
    '''
    total_freqs = {}
    
    # iterate over each sample json file in the directory
    for mutation_json in json_files:
        # get sample name from the file path
        mutation_json = Path(mutation_json)  # Convert to Path if not already
        sample = mutation_json.stem  # Use Path.stem instead of split
        
        # initialize the total frequencies for the sample
        total_freqs[sample] = {}
        
        # load the mutation json file
        mutation_counts = load_json(mutation_json)
        
        # iterate over each gene in the mutation json file
        for gene in mutation_counts:
            # for each site along the gene that has a mutation count
            for site in mutation_counts[gene]['mutations']:
                # get the mutation type ('C>T', 'A>G', etc.)
                mutation_type = site[-3:]
                new = mutation_type[-1]
                
                # skip if the new base is not a valid nucleotide
                if new not in ['A','C','G','T']:
                    continue
                
                # Count unique sites with mutations, not alternate allele counts
                if mutation_type not in total_freqs[sample]:
                    total_freqs[sample][mutation_type] = 1  # Count the site, not the allele count
                else:
                    total_freqs[sample][mutation_type] += 1  # Count the site, not the allele count
               
    return total_freqs

def count_mutations_alt_alleles(json_files: list) -> dict:
    '''Count alternate allele frequencies across all samples.
    
    Args:
        json_files (list): List of paths to mutation JSON files
    
    Returns:
        dict: Dictionary of mutation frequencies per sample (counting alternate alleles)
    '''
    total_freqs = {}
    
    # iterate over each sample json file in the directory
    for mutation_json in json_files:
        # get sample name from the file path
        mutation_json = Path(mutation_json)  # Convert to Path if not already
        sample = mutation_json.stem  # Use Path.stem instead of split
        
        # initialize the total frequencies for the sample
        total_freqs[sample] = {}
        
        # load the mutation json file
        mutation_counts = load_json(mutation_json)
        
        # iterate over each gene in the mutation json file
        for gene in mutation_counts:
            # for each site along the gene that has a mutation count
            for site in mutation_counts[gene]['mutations']:
                # get the mutation type ('C>T', 'A>G', etc.)
                mutation_type = site[-3:]
                new = mutation_type[-1]
                
                # skip if the new base is not a valid nucleotide
                if new not in ['A','C','G','T']:
                    continue
                
                # Count alternate allele counts
                if mutation_type not in total_freqs[sample]:
                    total_freqs[sample][mutation_type] = mutation_counts[gene]['mutations'][site]
                else:
                    total_freqs[sample][mutation_type] += mutation_counts[gene]['mutations'][site]
               
    return total_freqs

def count_mutations(json_files: list) -> dict:
    '''Count mutation frequencies across all samples (default: unique sites).
    
    Args:
        json_files (list): List of paths to mutation JSON files
    
    Returns:
        dict: Dictionary of mutation frequencies per sample
    '''
    return count_mutations_unique_sites(json_files)

def plot_kmer_context(context_counts: dict, genome_kmer_counts: dict, output_dir: str) -> None:
    '''Plot the 7-mer context barplot (7-mers only).'''
    output_path = os.path.join(output_dir, 'kmer_context.png')
    observed_kmers = {}
    ems_samples = [sample for sample in context_counts if 'EMS' in sample and '7d' not in sample and '3d' not in sample]

    # Use only 7-mers and the '7' key in genome_kmer_counts
    genome_7mer_counts = genome_kmer_counts['7'] if '7' in genome_kmer_counts else {}

    for sample in ems_samples:
        sample_data = context_counts[sample]
        for kmer, count in sample_data.items():
            if len(kmer) == 7 and kmer in genome_7mer_counts and genome_7mer_counts[kmer] > 0:
                freq = count / genome_7mer_counts[kmer]
                if kmer not in observed_kmers:
                    observed_kmers[kmer] = []
                observed_kmers[kmer].append(freq)

    # Average across samples
    averaged_kmers = {kmer: np.mean(freqs) for kmer, freqs in observed_kmers.items() if freqs}

    # Sort and get top 30
    top_kmers = dict(sorted(averaged_kmers.items(), key=lambda x: x[1], reverse=True)[:30])

    plt.figure(figsize=(12, 6))
    plt.bar(top_kmers.keys(), top_kmers.values(), color='darkorange')
    plt.xlabel('7-mer Context')
    plt.ylabel('Normalized Frequency (Observed/Genome)')
    plt.title('Top 30 Most Frequent 7-mer Contexts (EMS Samples)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def extract_sample_id(sample):
    '''Extract abbreviated sample ID from full sample name.
    
    Args:
        sample (str): Full sample name
        
    Returns:
        str: Abbreviated sample identifier
    '''
    # First get the basic ID
    if 'EMS-' in sample:
        sample_id = sample.split('EMS-')[1]
        prefix = 'EMS'
    elif 'EMS' in sample:
        sample_id = sample.split('EMS')[1]
        prefix = 'EMS'
    elif 'NT-' in sample:
        sample_id = sample.split('NT-')[1]
        prefix = 'NT'
    elif 'NT' in sample:
        sample_id = sample.split('NT')[1]
        prefix = 'NT'
    else:
        return sample
        
    # Clean up the ID to just keep number and treatment time if present
    if '_' in sample_id:
        # Handle cases like "1_3d" or "6_7d"
        parts = sample_id.split('_')
        if len(parts) >= 2 and 'd' in parts[1]:
            return f"{parts[0]}_{parts[1].split('_')[0]}"  # Keep just the number and days
        return f"{parts[0]}"  # Just keep the number if no valid treatment time
    
    # Remove any trailing text after numbers
    number = re.match(r'\d+', sample_id)
    return f"{number.group()}" if number else sample_id

def plot_ems_mutation_frequencies_per_sample(data_path: str, output_dir: str) -> None:
    '''Create bar plot of mutation frequencies per sample using matplotlib.'''
    output_path = os.path.join(output_dir, 'ems_mutation_frequencies_per_sample.png')
    
    df = pd.read_csv(data_path)
    mutations = sorted(df['mutation'].unique())
    control_samples = df[df['ems'] == '-']['sample'].unique()
    
    # Sort treated samples by total mutation frequency
    treated_totals = df[df['ems'] == '+'].groupby('sample')['norm_count'].sum()
    treated_samples = treated_totals.sort_values(ascending=True).index
    
    # Group treated samples into three bins by mutation rate
    n_samples = len(treated_samples)
    samples_per_group = n_samples // 3
    remainder = n_samples % 3
    
    # Create groups with even distribution of remainder
    group_sizes = [samples_per_group + (1 if i < remainder else 0) for i in range(3)]
    sample_groups = []
    start_idx = 0
    for size in group_sizes:
        sample_groups.append(treated_samples[start_idx:start_idx + size])
        start_idx += size
    
    group_labels = []
    for i, group in enumerate(['Low', 'Medium', 'High']):
        sample_ids = [extract_sample_id(s) for s in sample_groups[i]]
        # Split sample IDs into roughly equal groups
        n = len(sample_ids)
        split_point = (n + 1) // 2
        if n > 4:  # Only split if more than 4 samples
            group_labels.append(f"EMS {', '.join(sample_ids[:split_point])}")
            group_labels.append(f"EMS {', '.join(sample_ids[split_point:])}")
        else:
            group_labels.append(f"EMS {', '.join(sample_ids)}")
    
    # Plot setup and control samples (unchanged)
    plt.figure(figsize=(12, 6))
    bar_width = 0.8 / (len(control_samples) + len(treated_samples))
    group_spacing = 0.5
    x = np.arange(len(mutations)) * (1 + group_spacing)
    
    # Plot control samples
    for i, sample in enumerate(control_samples):
        sample_data = df[df['sample'] == sample].set_index('mutation')['norm_count']
        sample_values = [sample_data.get(mut, 0) for mut in mutations]
        pos = x - (0.4) + (i * bar_width)
        plt.bar(pos, sample_values, bar_width, color='lightgrey')
    
    # Plot treated samples with grouped colors
    colors = [plt.cm.Oranges(0.3), plt.cm.Oranges(0.6), plt.cm.Oranges(0.9)]
    current_idx = 0  # Keep track of overall position
    for group_idx, group_samples in enumerate(sample_groups):
        for i, sample in enumerate(group_samples):
            sample_data = df[df['sample'] == sample].set_index('mutation')['norm_count']
            sample_values = [sample_data.get(mut, 0) for mut in mutations]
            pos = x + (current_idx * bar_width)  # Use running index instead of group calculation
            plt.bar(pos, sample_values, bar_width, color=colors[group_idx])
            current_idx += 1  # Increment position counter
    
    # Create legend with split groups
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, color='lightgrey', label='Controls')
    ]
    
    # Add legend entries for each subgroup
    current_color_idx = 0
    for i in range(len(group_labels)):
        legend_elements.append(
            plt.Rectangle((0,0), 1, 1, 
                         color=colors[current_color_idx], 
                         label=group_labels[i])
        )
        if i % 2 == 1:  # Increment color index after each pair
            current_color_idx += 1
    
    plt.xlabel('Mutation Type')
    plt.ylabel('Normalized Mutation Rate')
    plt.title('EMS Mutation Type Frequencies by Sample')
    plt.xticks(x, mutations, rotation=45)
    # Change legend position
    plt.legend(
        handles=legend_elements, 
        loc='upper right',  # Position in top right
        bbox_to_anchor=(0.98, 0.98),  # Fine-tune position
        framealpha=0.9  # Make legend background slightly transparent
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_mutation_frequencies(data_path: str, output_dir: str) -> None:
    '''Create bar plot of mutation frequencies using matplotlib.'''
    output_path = os.path.join(output_dir, 'ems_mutation_frequencies.png')
    
    df = pd.read_csv(data_path)
    
    # Split data into control and treated groups and sort mutations
    mutations = sorted(df['mutation'].unique())
    control = df[df['ems'] == '-'].groupby('mutation')['norm_count'].mean().reindex(mutations)
    treated = df[df['ems'] == '+'].groupby('mutation')['norm_count'].mean().reindex(mutations)
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Create x-axis positions
    x = np.arange(len(mutations))
    
    # Create bars
    plt.bar(x - 0.2, control, 0.4, label='Control', color='lightgray')
    plt.bar(x + 0.2, treated, 0.4, label='EMS Treated', color='darkorange')
    
    # Customize plot
    plt.xlabel('Mutation Type')
    plt.ylabel('Normalized Mutation Rate')
    plt.title('EMS Mutation Type Frequencies')
    plt.xticks(x, mutations, rotation=45)
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_ems_mutation_frequencies_seaborn(data_path: str, output_dir: str) -> None:
    
    # Set output path for the plot
    output_path = os.path.join(output_dir, 'ems_mutation_frequencies.png')
    
    # Set the theme for the seaborn plot
    sns.set_theme(style="ticks", palette="colorblind")

    # Read the mutation frequencies from the CSV file
    mutation_frequencies = pd.read_csv(data_path)

    # Create a bar plot for the mutation frequencies
    g = sns.catplot(kind="bar", x="mutation", y="norm_count",
                    hue='ems', data=mutation_frequencies,
                    height=6, aspect=1.5)
                
    # Remove the top and right spines from the plot
    g.despine(left=True)
    # Set the axis labels
    g.set_axis_labels("", "Normalized Mutation Rate")
    # Rotate the x-axis labels for better readability
    g.set_xticklabels(rotation=40)
  
    # Add a legend to the plot
    plt.legend(loc='upper right', title='EMS', fancybox=True)
    # Format the y-axis labels to scientific notation
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    # Save the plot to a file
    plt.savefig(output_path, dpi=300, pad_inches=0.1)

def normalize_mutation_counts(total_freqs: dict, base_counts: dict, output_dir: str, suffix: str = '') -> str:
    '''Normalize mutation counts by base counts and save to CSV.
    
    Args:
        total_freqs (dict): Dictionary of mutation frequencies per sample
        base_counts (dict): Dictionary of genome-wide base counts {'A': count, 'T': count, ...}
        output_dir (str): Directory to save output CSV
        suffix (str): Optional suffix for the CSV filename to distinguish datasets
        
    Returns:
        str: Path to the saved CSV file
    '''
    
    output_path = os.path.join(output_dir, f'mutation_frequencies{suffix}.csv')
    
    # Prepare data for CSV
    csv_data = []
    
    for sample in total_freqs:
        # Determine if sample is EMS treated
        ems_status = '+' if 'EMS' in sample else '-'
        
        for mut_type in total_freqs[sample]:
            if '>' in mut_type:
                ref = mut_type.split('>')[0]
                if ref in base_counts:
                    # Normalize by genome-wide base count
                    norm_freq = total_freqs[sample][mut_type] / base_counts[ref]
                    
                    csv_data.append({
                        'sample': sample,
                        'mutation': mut_type,
                        'count': total_freqs[sample][mut_type],
                        'norm_count': norm_freq,
                        'ems': ems_status
                    })
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    
    return output_path

def plot_dnds_ratio(amino_acid_mutations, output_dir, sample_name):
    '''Plot dN/dS ratio relationships.
    
    Args:
        amino_acid_mutations (dict): Dictionary of amino acid mutations per gene
        output_dir (str): Directory to save output plots
        sample_name (str): Name of the sample for plot labeling
    '''
    # Extract data from amino_acid_mutations
    genes = []
    dnds_ratios = []
    gene_lengths = []
    coverages = []
    
    for gene in amino_acid_mutations:
        if 'dnds_norm' in amino_acid_mutations[gene] and 'gene_len' in amino_acid_mutations[gene]:
            genes.append(gene)
            dnds_ratios.append(amino_acid_mutations[gene]['dnds_norm'])
            gene_lengths.append(amino_acid_mutations[gene]['gene_len'])
            coverages.append(amino_acid_mutations[gene].get('avg_cov', 0))
    
    # Calculate average dN/dS
    avg_dnds = sum(dnds_ratios) / len(dnds_ratios) if dnds_ratios else 0
    print(f"Average dN/dS ratio for {sample_name}: {avg_dnds:.3f}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot dN/dS vs gene length
    ax1.scatter(gene_lengths, dnds_ratios, alpha=0.5, color='darkorange')
    ax1.set_xlabel('Gene Length (bp)')
    ax1.set_ylabel('dN/dS Ratio')
    ax1.set_title(f'dN/dS Ratio vs Gene Length - {sample_name}')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(plt.ScalarFormatter())
    
    # Plot dN/dS vs coverage
    ax2.scatter(coverages, dnds_ratios, alpha=0.5, color='darkorange') 
    ax2.set_xlabel('Average Coverage')
    ax2.set_ylabel('dN/dS Ratio')
    ax2.set_title(f'dN/dS Ratio vs Coverage - {sample_name}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(plt.ScalarFormatter())
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'dnds_relationships_{sample_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def mutation_type_barplot(json_files: list, base_counts: dict, output_dir: str) -> None:
    '''Generate mutation type frequency barplot.
    
    Args:
        json_files (list): List of paths to mutation JSON files
        base_counts (dict): Dictionary of genome-wide base counts {'A': count, 'T': count, ...}
        output_dir (str): Directory to save output plot
    '''
    
    # Count mutations across all samples
    total_freqs = count_mutations(json_files)
    
    # Normalize using provided base counts and generate CSV
    mutation_frequency_csv = normalize_mutation_counts(total_freqs, base_counts, output_dir, '')
    
    # Generate plots using the CSV
    plot_ems_mutation_frequencies(mutation_frequency_csv, output_dir)
    plot_ems_mutation_frequencies_per_sample(mutation_frequency_csv, output_dir)

def plot_combined_dnds_ratios(aa_mutations_files: list, output_dir: str) -> None:
    """Plot combined dN/dS ratio analysis."""
    all_data = {
        'names': [],
        'coverages': [],
        'syn_ratios': [],
        'is_control': []
    }
    
    for file in aa_mutations_files:
        with open(file) as f:
            data = json.load(f)
            
        # The data structure has changed - each file contains a list of results
        for result in data:
            # Get values from the 'original' section of results
            original = result['original']
            
            # Calculate dN/dS using unique sites
            unique_syn = original['unique_syn_sites']
            unique_non_syn = original['unique_non_syn_sites']
            dnds = original['dnds_raw']  # This is already calculated correctly
            
            all_data['names'].append(result['sample'])
            # Note: We don't have coverage info in this data structure
            all_data['coverages'].append(0)  # Placeholder
            all_data['syn_ratios'].append(dnds)
            all_data['is_control'].append('NT' in result['sample'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot dN/dS vs gene length
    for is_control, color, label in [(True, 'lightgrey', 'Control'), (False, 'darkorange', 'EMS Treated')]:
        mask = [x == is_control for x in all_data['is_control']]
        ax1.scatter(
            [c for c, m in zip(all_data['coverages'], mask) if m],
            [r for r, m in zip(all_data['syn_ratios'], mask) if m],
            alpha=0.5,
            color=color,
            label=label
        )
    
    ax1.set_xlabel('Average Coverage')
    ax1.set_ylabel('dN/dS Ratio')
    ax1.set_title('dN/dS Ratio vs Coverage - All Samples')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax1.legend()
    
    # Calculate and print average dN/dS for control and treated
    control_ratios = [r for r, c in zip(all_data['syn_ratios'], all_data['is_control']) if c]
    treated_ratios = [r for r, c in zip(all_data['syn_ratios'], all_data['is_control']) if not c]
    
    avg_control = sum(control_ratios) / len(control_ratios) if control_ratios else 0
    avg_treated = sum(treated_ratios) / len(treated_ratios) if treated_ratios else 0
    
    print(f"\nAverage dN/dS ratios:")
    print(f"Control samples: {avg_control:.3f}")
    print(f"Treated samples: {avg_treated:.3f}")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dnds_relationships_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_dnds_vs_coverage(aa_mutations_files: list, output_dir: str) -> None:
    '''Create plot comparing average dN/dS ratio vs average coverage across samples.
    
    Args:
        aa_mutations_files (list): List of paths to amino acid mutation JSON files
        output_dir (str): Directory to save output plots
    '''
    # Data structures for sample averages
    sample_data = {
        'names': [],
        'avg_dnds': [],
        'avg_coverage': [],
        'is_control': []
    }
    
    # Calculate averages for each sample
    for aa_file in aa_mutations_files:
        with open(aa_file) as f:
            aa_mutations = json.load(f)
            
        # Get sample info
        sample_name = aa_file.stem
        is_control = any(x in sample_name for x in ['NT', 'Minus', 'Pre'])
        
        # Calculate averages across genes
        dnds_values = []
        coverage_values = []
        for gene in aa_mutations:
            if 'dnds_norm' in aa_mutations[gene] and 'avg_cov' in aa_mutations[gene]:
                dnds_values.append(aa_mutations[gene]['dnds_norm'])
                coverage_values.append(aa_mutations[gene]['avg_cov'])
        
        if dnds_values and coverage_values:  # Only add if we have data
            sample_data['names'].append(sample_name)
            sample_data['avg_dnds'].append(sum(dnds_values) / len(dnds_values))
            sample_data['avg_coverage'].append(sum(coverage_values) / len(coverage_values))
            sample_data['is_control'].append(is_control)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot points with different colors for control/treated
    for is_control, color, label in [(True, 'lightgrey', 'Control'), (False, 'darkorange', 'EMS Treated')]:
        mask = [x == is_control for x in sample_data['is_control']]
        plt.scatter(
            [c for c, m in zip(sample_data['avg_coverage'], mask) if m],
            [d for d, m in zip(sample_data['avg_dnds'], mask) if m],
            alpha=0.7,
            color=color,
            label=label,
            s=100  # Larger point size
        )
    
    # Add sample labels with abbreviated IDs
    for i, txt in enumerate(sample_data['names']):
        abbreviated_name = extract_sample_id(txt)
        plt.annotate(abbreviated_name, 
                    (sample_data['avg_coverage'][i], sample_data['avg_dnds'][i]),
                    xytext=(10, 10),  # Increased offset from (5, 5)
                    textcoords='offset points',
                    fontsize=8,
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))  # Add connecting lines
    
    plt.xlabel('Average Coverage')
    plt.ylabel('Average dN/dS Ratio')
    plt.title('Sample Average dN/dS Ratio vs Coverage')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_dnds_vs_coverage.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_dnds_vs_intergenic(results_dir: str, output_dir: str) -> None:
    """Plot dN/dS vs intergenic mutation rates."""
    plot_data = {
        'sample': [],
        'dnds': [],
        'intergenic_rate': [],
        'syn_sites_mutated_rate': [],
        'non_syn_sites_mutated_rate': [],
        'is_control': []
    }
    
    csv_data = []
    
    for file in glob.glob(os.path.join(results_dir, '*_results.json')):
        with open(file) as f:
            data = json.load(f)
            
        # Use the correct dN/dS calculation from the original results
        dnds = data['original']['dnds_raw']  # This should already be calculated correctly
        
        # Rest of the function remains the same

def plot_dnds_analysis(results_csv: str, output_dir: str, title: str = "dN/dS Analysis"):
    """Generate plots for dN/dS analysis."""
    # Load results
    df = pd.read_csv(results_csv)
    
    # Create a shorter sample ID for plotting
    df['Sample_ID'] = df['Sample'].str.replace('_wmelASM802v1_variants', '')
    
    # Convert 'Significant' column to boolean if it's not already
    if df['Significant'].dtype == 'object':
        df['Significant'] = df['Significant'].map({'True': True, 'Yes': True, 'False': False, 'No': False})
    
    # Sort by ratio
    df = df.sort_values('Ratio', ascending=False)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: dN/dS values with confidence intervals
    x = np.arange(len(df))
    width = 0.35
    
    # Original dN/dS with error bars
    original = ax1.bar(x - width/2, df['Original_dNdS'], width, label='Original dN/dS',
                      yerr=[df['Original_dNdS'] - df['Original_dNdS_Lower_CI'], 
                            df['Original_dNdS_Upper_CI'] - df['Original_dNdS']], 
                      capsize=5, color='darkblue', alpha=0.7)
    
    # Random dN/dS with error bars
    random = ax1.bar(x + width/2, df['Random_dNdS'], width, label='Random dN/dS',
                    yerr=[df['Random_dNdS'] - df['Random_dNdS_Lower_CI'], 
                          df['Random_dNdS_Upper_CI'] - df['Random_dNdS']], 
                    capsize=5, color='darkgreen', alpha=0.7)
    
    # Add horizontal line at dN/dS = 1
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # Customize plot
    ax1.set_ylabel('dN/dS Ratio')
    ax1.set_title(f'{title} - dN/dS Values')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Sample_ID'])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.legend()
    
    # Plot 2: Ratio of Original/Random dN/dS
    # Create a color map based on significance
    colors = df['Significant'].map({True: 'darkred', False: 'teal'}).tolist()
    
    # Create the bar plot with explicit colors
    bars = ax2.bar(x, df['Ratio'], color=colors)
    
    # Add horizontal line at ratio = 1
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7)
    
    # Customize plot
    ax2.set_ylabel('Original/Random Ratio')
    ax2.set_title('Ratio of Original to Random dN/dS')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Sample_ID'])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add a legend for significance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', label='Significant'),
        Patch(facecolor='teal', label='Not Significant')
    ]
    ax2.legend(handles=legend_elements)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dnds_analysis.png", dpi=300)
    plt.savefig(f"{output_dir}/dnds_analysis.pdf")
    plt.close()

    # Also create a version with samples sorted by type (NT vs EMS)
    df['Sample_Type'] = df['Sample_ID'].str.extract(r'(NT|EMS)')
    df = df.sort_values(['Sample_Type', 'Ratio'], ascending=[True, False])
    
    # Repeat the plotting with the new sorting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    x = np.arange(len(df))
    
    # Original dN/dS with error bars
    original = ax1.bar(x - width/2, df['Original_dNdS'], width, label='Original dN/dS',
                      yerr=[df['Original_dNdS'] - df['Original_dNdS_Lower_CI'], 
                            df['Original_dNdS_Upper_CI'] - df['Original_dNdS']], 
                      capsize=5, color='darkblue', alpha=0.7)
    
    # Random dN/dS with error bars
    random = ax1.bar(x + width/2, df['Random_dNdS'], width, label='Random dN/dS',
                    yerr=[df['Random_dNdS'] - df['Random_dNdS_Lower_CI'], 
                          df['Random_dNdS_Upper_CI'] - df['Random_dNdS']], 
                    capsize=5, color='darkgreen', alpha=0.7)
    
    # Add horizontal line at dN/dS = 1
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # Customize plot
    ax1.set_ylabel('dN/dS Ratio')
    ax1.set_title(f'{title} - dN/dS Values (Sorted by Sample Type)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Sample_ID'])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.legend()
    
    # Plot 2: Ratio of Original/Random dN/dS
    # Create a color map based on significance
    colors = df['Significant'].map({True: 'darkred', False: 'teal'}).tolist()
    
    # Create the bar plot with explicit colors
    bars = ax2.bar(x, df['Ratio'], color=colors)
    
    # Add horizontal line at ratio = 1
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7)
    
    # Customize plot
    ax2.set_ylabel('Original/Random Ratio')
    ax2.set_title('Ratio of Original to Random dN/dS')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Sample_ID'])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add a legend for significance
    ax2.legend(handles=legend_elements)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dnds_analysis_by_type.png", dpi=300)
    plt.savefig(f"{output_dir}/dnds_analysis_by_type.pdf")
    plt.close()

def plot_per_gene_dnds(results_dir: str, output_dir: str) -> None:
    """Plot per-gene dN/dS ratios for each sample."""
    
    # Find all sample result files
    for sample_dir in Path(results_dir).glob('*'):
        if not sample_dir.is_dir():
            continue
            
        results_file = sample_dir / 'dnds_results.json'
        if not results_file.exists():
            continue
            
        with open(results_file) as f:
            data = json.load(f)
            
        # Extract sample name
        sample_name = sample_dir.name
        
        # Get per-gene data if available
        if 'per_gene' in data:
            gene_data = data['per_gene']
            
            # Create lists for plotting
            genes = []
            dnds_values = []
            errors = []
            
            for gene, stats in gene_data.items():
                if stats['dnds'] > 0:  # Only include genes with non-zero dN/dS
                    genes.append(gene)
                    dnds_values.append(stats['dnds'])
                    errors.append([
                        stats['dnds'] - stats['dnds_lower_ci'],
                        stats['dnds_upper_ci'] - stats['dnds']
                    ])
            
            if genes:  # Only create plot if we have data
                plt.figure(figsize=(12, 6))
                
                # Create bar plot with error bars
                y_pos = np.arange(len(genes))
                plt.bar(y_pos, dnds_values, align='center', alpha=0.5)
                
                if errors:
                    errors = np.array(errors).T
                    plt.errorbar(y_pos, dnds_values, yerr=errors, fmt='none', color='black', capsize=5)
                
                # Add horizontal line at dN/dS = 1
                plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)
                
                plt.ylabel('dN/dS Ratio')
                plt.title(f'Per-gene dN/dS Ratios - {sample_name}')
                plt.xticks(y_pos, genes, rotation=45, ha='right')
                
                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(f"{output_dir}/per_gene_dnds_{sample_name}.png", dpi=300, bbox_inches='tight')
                plt.close()

def plot_normalized_kmer_context_per_sample(context_counts: dict, genome_kmer_counts: dict, output_dir: str) -> None:
    '''Plot normalized 7-mer context for each sample.'''
    output_path = os.path.join(output_dir, 'normalized_kmer_context_per_sample.png')
    sample_data = {}
    all_kmers = set()
    genome_7mer_counts = genome_kmer_counts['7'] if '7' in genome_kmer_counts else {}

    for sample, data in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
        sample_data[sample] = {}
        for kmer, count in data.items():
            if len(kmer) == 7 and kmer in genome_7mer_counts and genome_7mer_counts[kmer] > 0:
                norm_freq = count / genome_7mer_counts[kmer]
                sample_data[sample][kmer] = norm_freq
                all_kmers.add(kmer)

    if not sample_data:
        print("No normalized 7-mer kmer data found")
        return

    # Get top 20 most variable kmers across all samples
    kmer_variances = {}
    for kmer in all_kmers:
        values = [sample_data[sample].get(kmer, 0) for sample in sample_data]
        if len(values) > 1:
            kmer_variances[kmer] = np.var(values)

    top_kmers = sorted(kmer_variances.items(), key=lambda x: x[1], reverse=True)[:20]
    top_kmer_names = [kmer for kmer, _ in top_kmers]

    ems_samples = [s for s in sample_data.keys() if 'EMS' in s and '7d' not in s and '3d' not in s]
    control_samples = [s for s in sample_data.keys() if 'EMS' not in s and '3d' not in s]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    plot_data = []
    sample_labels = []

    for sample in control_samples + ems_samples:
        row = [sample_data[sample].get(kmer, 0) for kmer in top_kmer_names]
        plot_data.append(row)
        sample_labels.append(sample)

    im = ax1.imshow(plot_data, cmap='RdYlBu_r', aspect='auto')
    ax1.set_xticks(range(len(top_kmer_names)))
    ax1.set_xticklabels(top_kmer_names, rotation=45, ha='right')
    ax1.set_yticks(range(len(sample_labels)))
    ax1.set_yticklabels(sample_labels)
    ax1.set_title('Normalized 7-mer Context Frequencies (Top 20 Most Variable)')
    ax1.set_xlabel('7-mer Context')
    ax1.set_ylabel('Sample')
    plt.colorbar(im, ax=ax1, label='Normalized Frequency')

    if control_samples:
        ax1.axhline(y=len(control_samples)-0.5, color='black', linewidth=2)

    if control_samples and ems_samples:
        control_means = {kmer: np.mean([sample_data[s].get(kmer, 0) for s in control_samples]) for kmer in top_kmer_names}
        ems_means = {kmer: np.mean([sample_data[s].get(kmer, 0) for s in ems_samples]) for kmer in top_kmer_names}
        x = np.arange(len(top_kmer_names))
        width = 0.35
        ax2.bar(x - width/2, [control_means[k] for k in top_kmer_names], width, label='Control', color='lightgray', alpha=0.7)
        ax2.bar(x + width/2, [ems_means[k] for k in top_kmer_names], width, label='EMS Treated', color='darkorange', alpha=0.7)
        ax2.set_xlabel('7-mer Context')
        ax2.set_ylabel('Mean Normalized Frequency')
        ax2.set_title('Average Normalized 7-mer Frequencies: Control vs EMS')
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_kmer_names, rotation=45, ha='right')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_ems_mutation_kmer_bias(context_counts: dict, genome_kmer_counts: dict, output_dir: str) -> None:
    '''Analyze 7-mer kmer bias for each EMS sample individually.'''
    if not genome_kmer_counts or '7' not in genome_kmer_counts:
        print("No genome 7-mer kmer counts found for bias analysis")
        return

    genome_7mer_counts = genome_kmer_counts['7']
    # Only use C/G-centered 7-mers for expected
    cg_genome_kmers = {kmer: count for kmer, count in genome_7mer_counts.items() if len(kmer) == 7 and kmer[3] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())

    for sample, sample_kmers in context_counts.items():
        if 'EMS' not in sample or '7d' in sample or '3d' in sample:
            continue

        output_path = os.path.join(output_dir, f'ems_mutation_kmer_bias_{sample}.png')
        # Only use C/G-centered 7-mers for observed
        ems_mutation_kmers = {kmer: count for kmer, count in sample_kmers.items() if len(kmer) == 7 and kmer[3] in ['C', 'G']}

        total_mutations = sum(ems_mutation_kmers.values())

        bias_analysis = []
        for kmer, observed_count in ems_mutation_kmers.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan

            # Fisher's exact test
            table = [
                [observed_count, total_mutations - observed_count],
                [genome_count, total_genome_kmers - genome_count]
            ]
            _, p_value = fisher_exact(table, alternative='greater')

            bias_analysis.append({
                'kmer': kmer,
                'observed_count': observed_count,
                'expected_count': expected_count,
                'enrichment_ratio': enrichment_ratio,
                'p_value': p_value,
                'significant': bool(p_value < 0.05 and enrichment_ratio > 1)
            })

        bias_analysis.sort(key=lambda x: x['enrichment_ratio'], reverse=True)
        significant_kmers = [x for x in bias_analysis if x['significant']]

        # Save detailed results
        results_path = os.path.join(output_dir, f'ems_kmer_bias_analysis_{sample}.json')
        with open(results_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_mutations': total_mutations,
                    'total_kmers_analyzed': len(bias_analysis),
                    'significantly_enriched': len(significant_kmers),
                    'median_enrichment': float(np.median([x['enrichment_ratio'] for x in bias_analysis if not np.isnan(x['enrichment_ratio'])])),
                    'mean_enrichment': float(np.mean([x['enrichment_ratio'] for x in bias_analysis if not np.isnan(x['enrichment_ratio'])]))
                },
                'top_enriched_kmers': bias_analysis[:20],
                'significant_kmers': significant_kmers
            }, f, indent=2)

        # Plot top 20 enriched kmers for this sample
        top_kmers = bias_analysis[:20]
        if top_kmers:
            kmers = [x['kmer'] for x in top_kmers]
            ratios = [x['enrichment_ratio'] for x in top_kmers]
            colors = ['red' if x['significant'] else 'orange' for x in top_kmers]

            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(kmers))
            plt.barh(y_pos, ratios, color=colors, alpha=0.7)
            plt.yticks(y_pos, kmers)
            plt.xlabel('Enrichment Ratio (Observed/Expected)')
            plt.title(f'Top 20 Enriched 7-mers in {sample}')
            plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        # Print summary
        print(f"\nEMS Mutation 7-mer Kmer Bias Analysis for {sample}:")
        print(f"Total mutations analyzed: {total_mutations:,}")
        print(f"Significantly enriched kmers: {len(significant_kmers)}")
        if significant_kmers:
            print(f"\nTop 5 significantly enriched kmers:")
            for i, kmer_data in enumerate(significant_kmers[:5]):
                print(f"  {i+1}. {kmer_data['kmer']}: {kmer_data['enrichment_ratio']:.2f}x enriched (p={kmer_data['p_value']:.2e})")
        print(f"Detailed results saved to {results_path}")

def analyze_ems_cg_kmer_bias(context_counts: dict, genome_kmer_counts: dict, output_dir: str) -> None:
    '''Analyze EMS mutation bias specifically for 7-mers with C or G as center base.'''
    output_path = os.path.join(output_dir, 'ems_cg_kmer_bias.png')
    if not genome_kmer_counts or '7' not in genome_kmer_counts:
        print("No genome 7-mer kmer counts found for C/G kmer bias analysis")
        return
    genome_7mer_counts = genome_kmer_counts['7']
    ems_mutation_kmers = {}
    cg_genome_kmers = genome_7mer_counts
    # Collect mutation kmers from EMS samples (only C/G center, 7-mers only)
    for sample, data in context_counts.items():
        if 'EMS' in sample and sample != 'genome_kmer_counts' and '7d' not in sample and '3d' not in sample:
            for kmer, count in data.items():
                if isinstance(count, int) and len(kmer) == 7 and kmer[3] in ['C', 'G']:
                    ems_mutation_kmers[kmer] = ems_mutation_kmers.get(kmer, 0) + count
    if not ems_mutation_kmers:
        print("No EMS mutation C/G 7-mer kmer data found")
        return
    total_mutations = sum(ems_mutation_kmers.values())
    total_cg_genome_kmers = sum(cg_genome_kmers.values())
    bias_analysis = []
    for kmer in cg_genome_kmers:
        observed_count = ems_mutation_kmers.get(kmer, 0)
        expected_freq = cg_genome_kmers[kmer] / total_cg_genome_kmers
        expected_count = expected_freq * total_mutations
        enrichment_ratio = observed_count / expected_count if expected_count > 0 else 0
        from scipy.stats import binomtest
        if total_mutations > 0 and expected_freq > 0:
            result = binomtest(observed_count, total_mutations, expected_freq)
            p_value = result.pvalue
        else:
            p_value = 1.0
        bias_analysis.append({
            'kmer': kmer,
            'center_base': kmer[3],
            'observed': observed_count,
            'expected': expected_count,
            'genome_count': cg_genome_kmers[kmer],
            'enrichment_ratio': enrichment_ratio,
            'p_value': p_value,
            'significant': bool(p_value < 0.05 and enrichment_ratio > 1.5)
        })
    bias_analysis.sort(key=lambda x: x['enrichment_ratio'], reverse=True)
    c_kmers = [x for x in bias_analysis if x['center_base'] == 'C']
    g_kmers = [x for x in bias_analysis if x['center_base'] == 'G']
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    top_c_enriched = [x for x in c_kmers if x['enrichment_ratio'] > 1.0][:15]
    if top_c_enriched:
        kmers = [x['kmer'] for x in top_c_enriched]
        ratios = [x['enrichment_ratio'] for x in top_c_enriched]
        colors = ['red' if x['significant'] else 'orange' for x in top_c_enriched]
        y_pos = np.arange(len(kmers))
        ax1.barh(y_pos, ratios, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(kmers)
        ax1.set_xlabel('Enrichment Ratio (Observed/Expected)')
        ax1.set_title('Top Enriched C-Center 7-mers in EMS Mutations')
        ax1.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    top_g_enriched = [x for x in g_kmers if x['enrichment_ratio'] > 1.0][:15]
    if top_g_enriched:
        kmers = [x['kmer'] for x in top_g_enriched]
        ratios = [x['enrichment_ratio'] for x in top_g_enriched]
        colors = ['red' if x['significant'] else 'orange' for x in top_g_enriched]
        y_pos = np.arange(len(kmers))
        ax2.barh(y_pos, ratios, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(kmers)
        ax2.set_xlabel('Enrichment Ratio (Observed/Expected)')
        ax2.set_title('Top Enriched G-Center 7-mers in EMS Mutations')
        ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    if top_c_enriched:
        kmers = [x['kmer'] for x in top_c_enriched[:10]]
        observed = [x['observed'] for x in top_c_enriched[:10]]
        expected = [x['expected'] for x in top_c_enriched[:10]]
        x = np.arange(len(kmers))
        width = 0.35
        ax3.bar(x - width/2, observed, width, label='Observed', color='red', alpha=0.7)
        ax3.bar(x + width/2, expected, width, label='Expected', color='blue', alpha=0.7)
        ax3.set_xlabel('C-Center 7-mer')
        ax3.set_ylabel('Mutation Count')
        ax3.set_title('Observed vs Expected Counts (Top C-Center 7-mers)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(kmers, rotation=45, ha='right')
        ax3.legend()
    if top_g_enriched:
        kmers = [x['kmer'] for x in top_g_enriched[:10]]
        observed = [x['observed'] for x in top_g_enriched[:10]]
        expected = [x['expected'] for x in top_g_enriched[:10]]
        x = np.arange(len(kmers))
        width = 0.35
        ax4.bar(x - width/2, observed, width, label='Observed', color='red', alpha=0.7)
        ax4.bar(x + width/2, expected, width, label='Expected', color='blue', alpha=0.7)
        ax4.set_xlabel('G-Center 7-mer')
        ax4.set_ylabel('Mutation Count')
        ax4.set_title('Observed vs Expected Counts (Top G-Center 7-mers)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(kmers, rotation=45, ha='right')
        ax4.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    c_significant = [x for x in c_kmers if x['significant']]
    g_significant = [x for x in g_kmers if x['significant']]
    results_path = os.path.join(output_dir, 'ems_cg_kmer_bias_analysis.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'total_mutations': total_mutations,
                'total_cg_kmers_analyzed': len(bias_analysis),
                'c_center_kmers': len(c_kmers),
                'g_center_kmers': len(g_kmers),
                'c_significant': len(c_significant),
                'g_significant': len(g_significant),
                'median_c_enrichment': np.median([x['enrichment_ratio'] for x in c_kmers]) if c_kmers else 0,
                'median_g_enrichment': np.median([x['enrichment_ratio'] for x in g_kmers]) if g_kmers else 0,
                'mean_c_enrichment': np.mean([x['enrichment_ratio'] for x in c_kmers]) if c_kmers else 0,
                'mean_g_enrichment': np.mean([x['enrichment_ratio'] for x in g_kmers]) if g_kmers else 0
            },
            'top_c_enriched': c_kmers[:10],
            'top_g_enriched': g_kmers[:10],
            'significant_c_kmers': c_significant,
            'significant_g_kmers': g_significant
        }, f, indent=2)
    print(f"\nEMS C/G 7-mer Bias Analysis:")
    print(f"Total C/G mutations analyzed: {total_mutations:,}")
    print(f"C-center 7-mers analyzed: {len(c_kmers)}")
    print(f"G-center 7-mers analyzed: {len(g_kmers)}")
    print(f"Significantly enriched C-center 7-mers: {len(c_significant)}")
    print(f"Significantly enriched G-center 7-mers: {len(g_significant)}")
    c_ratios = [x['enrichment_ratio'] for x in c_kmers]
    g_ratios = [x['enrichment_ratio'] for x in g_kmers]
    print(f"Median C-center enrichment: {np.median(c_ratios):.2f}" if c_ratios else "No C-center data")
    print(f"Median G-center enrichment: {np.median(g_ratios):.2f}" if g_ratios else "No G-center data")
    if c_significant:
        print(f"\nTop 3 significantly enriched C-center 7-mers:")
        for i, kmer_data in enumerate(c_significant[:3]):
            print(f"  {i+1}. {kmer_data['kmer']}: {kmer_data['enrichment_ratio']:.2f}x enriched (p={kmer_data['p_value']:.2e})")
    if g_significant:
        print(f"\nTop 3 significantly enriched G-center 7-mers:")
        for i, kmer_data in enumerate(g_significant[:3]):
            print(f"  {i+1}. {kmer_data['kmer']}: {kmer_data['enrichment_ratio']:.2f}x enriched (p={kmer_data['p_value']:.2e})")
    print(f"Detailed results saved to {results_path}")
    print(f"Visualization saved to {output_path}")

def plot_aggregate_mutation_frequencies(normalized_rates: dict, output_dir: str) -> None:
    '''Plot aggregate mutation frequencies across treatment groups.'''
    
    output_path = os.path.join(output_dir, 'mutation_frequency_aggregate.png')
    
    # Create DataFrame for plotting
    plot_data = []
    for sample in normalized_rates:
        sample_id = extract_sample_id(sample)
        for mutation_type in normalized_rates[sample]:
            plot_data.append({
                'sample': sample_id,
                'mutation_type': mutation_type,
                'rate': normalized_rates[sample][mutation_type],
                'treatment': 'EMS' if 'EMS' in sample else 'Control'
            })
    
    if not plot_data:
        print("No mutation data found for plotting")
        return
        
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Filter for main mutation types
    main_mutations = ['C>T', 'G>A', 'A>G', 'T>C', 'A>T', 'T>A', 'C>G', 'G>C']
    df_filtered = df[df['mutation_type'].isin(main_mutations)]
    
    if df_filtered.empty:
        print("No main mutation types found for plotting")
        return
    
    # Create grouped bar plot
    sns.barplot(data=df_filtered, x='mutation_type', y='rate', hue='treatment', 
                palette=['orange', 'blue'], alpha=0.8)
    
    plt.title('Mutation Rates by Type and Treatment (Aggregate)')
    plt.xlabel('Mutation Type')
    plt.ylabel('Normalized Mutation Rate')
    plt.xticks(rotation=45)
    plt.legend(title='Treatment')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Aggregate mutation frequency plot saved to {output_path}")

def plot_per_sample_mutation_frequencies(normalized_rates: dict, output_dir: str) -> None:
    '''Plot mutation frequencies for each individual sample.'''
    
    output_path = os.path.join(output_dir, 'mutation_frequency_per_sample.png')
    
    # Create DataFrame for plotting
    plot_data = []
    for sample in normalized_rates:
        sample_id = extract_sample_id(sample)
        for mutation_type in normalized_rates[sample]:
            plot_data.append({
                'sample': sample_id,
                'mutation_type': mutation_type,
                'rate': normalized_rates[sample][mutation_type],
                'treatment': 'EMS' if 'EMS' in sample else 'Control'
            })
    
    if not plot_data:
        print("No mutation data found for plotting")
        return
        
    df = pd.DataFrame(plot_data)
    
    # Filter for main mutation types
    main_mutations = ['C>T', 'G>A', 'A>G', 'T>C', 'A>T', 'T>A', 'C>G', 'G>C']
    df_filtered = df[df['mutation_type'].isin(main_mutations)]
    
    if df_filtered.empty:
        print("No main mutation types found for plotting")
        return
    
    # Create a single plot with samples on x-axis and mutation types as different colors
    plt.figure(figsize=(16, 8))
    
    # Create grouped bar plot with samples on x-axis and mutation types as hue
    sns.barplot(data=df_filtered, x='sample', y='rate', hue='mutation_type', 
                palette='Set2', alpha=0.8)
    
    plt.title('Mutation Rates by Sample and Type')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Mutation Rate')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Mutation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add treatment group annotations
    ax = plt.gca()
    sample_names = df_filtered['sample'].unique()
    for i, sample in enumerate(sample_names):
        treatment = 'EMS' if any('EMS' in s for s in normalized_rates.keys() if extract_sample_id(s) == sample) else 'Control'
        color = 'orange' if treatment == 'EMS' else 'blue'
        ax.text(i, -0.05 * ax.get_ylim()[1], treatment, ha='center', va='top', 
                color=color, fontweight='bold', transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-sample mutation frequency plot saved to {output_path}")

def plot_ems_mutation_frequencies_heatmap(normalized_rates: dict, output_dir: str) -> None:
    '''Create a heatmap of mutation frequencies across samples.'''
    
    output_path = os.path.join(output_dir, 'mutation_frequency_heatmap.png')
    
    # Prepare data for heatmap
    samples = list(normalized_rates.keys())
    mutation_types = ['C>T', 'G>A', 'A>G', 'T>C', 'A>T', 'T>A', 'C>G', 'G>C']
    
    # Create matrix
    matrix_data = []
    sample_labels = []
    
    for sample in samples:
        sample_id = extract_sample_id(sample)
        sample_labels.append(sample_id)
        row = []
        for mut_type in mutation_types:
            rate = normalized_rates[sample].get(mut_type, 0)
            row.append(rate)
        matrix_data.append(row)
    
    # Convert to DataFrame
    df_heatmap = pd.DataFrame(matrix_data, index=sample_labels, columns=mutation_types)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_heatmap, annot=True, fmt='.2e', cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized Mutation Rate'})
    
    plt.title('Mutation Rate Heatmap Across Samples')
    plt.xlabel('Mutation Type')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mutation frequency heatmap saved to {output_path}")

def clean_ems_sample_name(sample):
    """Clean EMS sample name to just show EMS-1, EMS-2, etc."""
    if 'EMS' not in sample or '7d' in sample or '3d' in sample:
        return sample
    
    # Extract the number after EMS
    if 'EMS-' in sample:
        sample_id = sample.split('EMS-')[1]
    elif 'EMS' in sample:
        sample_id = sample.split('EMS')[1]
    else:
        return sample
    
    # Extract just the number
    number = re.match(r'\d+', sample_id)
    if number:
        return f"EMS-{number.group()}"
    return sample

def plot_consistently_enriched_kmers_heatmap(context_counts: dict, genome_kmer_counts: dict, output_dir: str, min_enrichment: float = 1.5, top_n: int = 30):
    """
    Plot a heatmap of raw enrichment ratios for 7-mers that are consistently enriched across all EMS samples.
    """
    ems_samples = [sample for sample in context_counts if 'EMS' in sample and '7d' not in sample and '3d' not in sample]
    if not ems_samples:
        print("No EMS samples found for consistently enriched 7-mer heatmap.")
        return
    genome_7mer_counts = genome_kmer_counts['7'] if '7' in genome_kmer_counts else {}
    cg_genome_kmers = {kmer: count for kmer, count in genome_7mer_counts.items() if len(kmer) == 7 and kmer[3] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())
    enrichment_data = {}
    for sample in ems_samples:
        sample_kmers = context_counts[sample]
        ems_mutation_kmers = {kmer: count for kmer, count in sample_kmers.items() if len(kmer) == 7 and kmer[3] in ['C', 'G']}
        total_mutations = sum(ems_mutation_kmers.values())
        enrichment_data[sample] = {}
        for kmer, observed_count in ems_mutation_kmers.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            enrichment_data[sample][kmer] = enrichment_ratio
    df = pd.DataFrame.from_dict(enrichment_data, orient='index')
    # Clean sample names
    df.index = [clean_ems_sample_name(sample) for sample in df.index]
    consistently_enriched_kmers = df.columns[(df.min(axis=0) > min_enrichment)]
    if len(consistently_enriched_kmers) == 0:
        print(f"No 7-mers found with minimum enrichment > {min_enrichment} across all samples.")
        return
    top_kmers = df[consistently_enriched_kmers].median(axis=0).sort_values(ascending=False).head(top_n).index
    df_top = df[top_kmers]
    plt.figure(figsize=(1.2*len(top_kmers), 0.5*len(df_top)+4))
    sns.heatmap(df_top, cmap='YlOrRd', annot=True, fmt=".2f", cbar_kws={'label': 'Enrichment Ratio (Obs/Exp)'})
    plt.title(f'Consistently Enriched 7-mers (min enrichment > {min_enrichment})\nTop {top_n} by median enrichment')
    plt.xlabel('7-mer')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'consistently_enriched_7mer_heatmap_top{top_n}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_consistently_depleted_kmers_heatmap(
    context_counts: dict,
    genome_kmer_counts: dict,
    output_dir: str,
    max_enrichment: float = 0.67,
    top_n: int = 30
):
    """
    Plot a heatmap of raw enrichment ratios for 7-mers that are consistently depleted across all EMS samples.
    """
    ems_samples = [sample for sample in context_counts if 'EMS' in sample and '7d' not in sample and '3d' not in sample]
    if not ems_samples:
        print("No EMS samples found for consistently depleted 7-mer heatmap.")
        return
    genome_7mer_counts = genome_kmer_counts['7'] if '7' in genome_kmer_counts else {}
    cg_genome_kmers = {kmer: count for kmer, count in genome_7mer_counts.items() if len(kmer) == 7 and kmer[3] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())
    enrichment_data = {}
    for sample in ems_samples:
        sample_kmers = context_counts[sample]
        ems_mutation_kmers = {kmer: count for kmer, count in sample_kmers.items() if len(kmer) == 7 and kmer[3] in ['C', 'G']}
        total_mutations = sum(ems_mutation_kmers.values())
        enrichment_data[sample] = {}
        for kmer, observed_count in ems_mutation_kmers.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            enrichment_data[sample][kmer] = enrichment_ratio
    df = pd.DataFrame.from_dict(enrichment_data, orient='index')
    # Clean sample names
    df.index = [clean_ems_sample_name(sample) for sample in df.index]
    consistently_depleted_kmers = df.columns[(df.max(axis=0) < max_enrichment)]
    if len(consistently_depleted_kmers) == 0:
        print(f"No 7-mers found with maximum enrichment < {max_enrichment} across all samples.")
        return
    top_kmers = df[consistently_depleted_kmers].median(axis=0).sort_values(ascending=True).head(top_n).index
    df_top = df[top_kmers]
    plt.figure(figsize=(1.2*len(top_kmers), 0.5*len(df_top)+4))
    sns.heatmap(df_top, cmap='Blues', annot=True, fmt=".2f", cbar_kws={'label': 'Enrichment Ratio (Obs/Exp)'})
    plt.title(f'Consistently Depleted 7-mers (max enrichment < {max_enrichment})\nTop {top_n} by lowest median enrichment')
    plt.xlabel('7-mer')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'consistently_depleted_7mer_heatmap_top{top_n}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_gene_mutation_statistics(gene_stats_path: str, output_dir: str, top_n: int = 30):
    """
    Plots:
    1. Scatter: total mutations per gene vs average coverage (across all samples, skipping 7d)
    2. Scatter: total mutations per gene vs gene length (across all samples, skipping 7d)
    3. Heatmap: top mutated genes (normalized by gene length) across samples (skipping 7d)
    """

    # Load gene statistics
    with open(gene_stats_path) as f:
        gene_stats = json.load(f)

    # Build DataFrame: rows = (sample, gene), columns = mutations, coverage, gene_length
    records = []
    for sample, genes in gene_stats.items():
        if "7d" in sample:
            continue
        for gene_id, stats in genes.items():
            records.append({
                "sample": sample,
                "gene": gene_id,
                "total_mutations": stats.get("total_mutations", 0),
                "average_coverage": stats.get("average_coverage", None),
                "gene_length": stats.get("gene_length", None)
            })
    df = pd.DataFrame(records)

    # 1 & 2. Scatter plots (aggregate across samples)
    agg = df.groupby("gene").agg(
        total_mutations_sum=("total_mutations", "sum"),
        average_coverage_mean=("average_coverage", "mean"),
        gene_length=("gene_length", "first")
    ).reset_index()

    # Identify outliers: genes with total_mutations_sum > 99th percentile
    outlier_thresh = agg["total_mutations_sum"].quantile(0.99)
    outliers = agg[agg["total_mutations_sum"] > outlier_thresh]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Scatter: total mutations vs average coverage
    axes[0].scatter(agg["average_coverage_mean"], agg["total_mutations_sum"], alpha=0.7)
    axes[0].set_xlabel("Average Coverage (mean across samples)")
    axes[0].set_ylabel("Total Mutations (sum across samples)")
    axes[0].set_title("Total Mutations per Gene vs Average Coverage")

    # Label outliers
    for _, row in outliers.iterrows():
        axes[0].annotate(row["gene"], (row["average_coverage_mean"], row["total_mutations_sum"]),
                         textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, color='red')

    # Scatter: total mutations vs gene length
    axes[1].scatter(agg["gene_length"], agg["total_mutations_sum"], alpha=0.7)
    axes[1].set_xlabel("Gene Length")
    axes[1].set_ylabel("Total Mutations (sum across samples)")
    axes[1].set_title("Total Mutations per Gene vs Gene Length")

    # Label outliers
    for _, row in outliers.iterrows():
        axes[1].annotate(row["gene"], (row["gene_length"], row["total_mutations_sum"]),
                         textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gene_mutation_scatterplots.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap: top mutated genes (normalized by gene length) across samples
    # Compute normalized mutations
    df["mut_per_kb"] = df["total_mutations"] / df["gene_length"] * 1000

    # Filter out 3d and 7d samples
    df_filtered = df[~df['sample'].str.contains('3d|7d')]

    # Find top N genes by median mut_per_kb across samples
    top_genes = (
        df_filtered.groupby("gene")["mut_per_kb"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    heatmap_df = (
        df_filtered[df_filtered["gene"].isin(top_genes)]
        .pivot(index="sample", columns="gene", values="mut_per_kb")
        .fillna(0)
    )

    # Order samples (rows) so EMS samples are grouped
    heatmap_df = heatmap_df.loc[sorted(heatmap_df.index, key=lambda x: ("EMS" not in x, x))]

    plt.figure(figsize=(1.2*top_n, 0.5*len(heatmap_df)+4))
    sns.heatmap(heatmap_df, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={"label": "Mutations per kb"})
    plt.title(f"Top {top_n} Most Mutated Genes (Normalized by Length)\nAcross EMS Samples (Excluding 3d/7d)")
    plt.xlabel("Gene")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"top_mutated_genes_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

def _extract_3mer_counts_from_5mers(context_counts, genome_kmer_counts):
    """Aggregate 5-mer counts into 3-mer counts (centered on C/G)."""
    # Count all 3-mers from the genome sequence
    genome_3mer_counts = {}
    for kmer5, gcount in genome_kmer_counts.items():
        if len(kmer5) == 5:
            # Extract the 3-mer centered on the mutation position (position 2)
            kmer3 = kmer5[1:4]  # positions 1,2,3 (centered on position 2)
            if kmer3[1] in "CG":  # Only count 3-mers with C/G in the center
                genome_3mer_counts[kmer3] = genome_3mer_counts.get(kmer3, 0) + gcount

    # Aggregate sample 3-mer counts from 5-mers
    sample_3mer_counts = {}
    for sample, sample_counts in context_counts.items():
        sample_3mer_counts[sample] = {}
        for kmer5, scount in sample_counts.items():
            if len(kmer5) == 5:
                # Extract the 3-mer centered on the mutation position (position 2)
                kmer3 = kmer5[1:4]  # positions 1,2,3 (centered on position 2)
                if kmer3[1] in "CG":  # Only count 3-mers with C/G in the center
                    sample_3mer_counts[sample][kmer3] = sample_3mer_counts[sample].get(kmer3, 0) + scount

    return sample_3mer_counts, genome_3mer_counts

def plot_consistently_enriched_3mer_kmers_heatmap(context_counts, genome_kmer_counts, output_dir, min_enrichment=1.5, top_n=30):
   

    sample_3mer_counts, genome_3mer_counts = _extract_3mer_counts_from_5mers(context_counts, genome_kmer_counts)
    ems_samples = [s for s in sample_3mer_counts if 'EMS' in s and '7d' not in s and '3d' not in s]
    if not ems_samples:
        print("No EMS samples found for consistently enriched 3-mer kmer heatmap.")
        return

    total_genome_3mers = sum(genome_3mer_counts.values())
    enrichment_data = {}
    for sample in ems_samples:
        sample_kmers = sample_3mer_counts[sample]
        total_mutations = sum(sample_kmers.values())
        enrichment_data[sample] = {}
        for kmer, observed_count in sample_kmers.items():
            genome_count = genome_3mer_counts.get(kmer, 0)
            if genome_count == 0 or total_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_3mers
            expected_count = expected_freq * total_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else float('nan')
            enrichment_data[sample][kmer] = enrichment_ratio

    df = pd.DataFrame.from_dict(enrichment_data, orient='index')
    # Clean sample names
    df.index = [clean_ems_sample_name(sample) for sample in df.index]
    consistently_enriched_kmers = df.columns[(df.min(axis=0) > min_enrichment)]
    if len(consistently_enriched_kmers) == 0:
        print(f"No 3-mers found with minimum enrichment > {min_enrichment} across all samples.")
        return

    top_kmers = df[consistently_enriched_kmers].median(axis=0).sort_values(ascending=False).head(top_n).index
    df_top = df[top_kmers]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(1.2*len(top_kmers), 0.5*len(df_top)+4))
    sns.heatmap(df_top, cmap='YlOrRd', annot=True, fmt=".2f", cbar_kws={'label': 'Enrichment Ratio (Obs/Exp)'})
    plt.title(f'Consistently Enriched 3-mers (min enrichment > {min_enrichment})\nTop {top_n} by median enrichment')
    plt.xlabel('3-mer')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'consistently_enriched_3mer_heatmap_top{top_n}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_consistently_depleted_3mer_kmers_heatmap(context_counts, genome_kmer_counts, output_dir, max_enrichment=0.67, top_n=30):
   

    sample_3mer_counts, genome_3mer_counts = _extract_3mer_counts_from_5mers(context_counts, genome_kmer_counts)
    ems_samples = [s for s in sample_3mer_counts if 'EMS' in s and '7d' not in s and '3d' not in s]
    if not ems_samples:
        print("No EMS samples found for consistently depleted 3-mer kmer heatmap.")
        return

    total_genome_3mers = sum(genome_3mer_counts.values())
    enrichment_data = {}
    for sample in ems_samples:
        sample_kmers = sample_3mer_counts[sample]
        total_mutations = sum(sample_kmers.values())
        enrichment_data[sample] = {}
        for kmer, observed_count in sample_kmers.items():
            genome_count = genome_3mer_counts.get(kmer, 0)
            if genome_count == 0 or total_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_3mers
            expected_count = expected_freq * total_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else float('nan')
            enrichment_data[sample][kmer] = enrichment_ratio

    df = pd.DataFrame.from_dict(enrichment_data, orient='index')
    # Clean sample names
    df.index = [clean_ems_sample_name(sample) for sample in df.index]
    consistently_depleted_kmers = df.columns[(df.max(axis=0) < max_enrichment)]
    if len(consistently_depleted_kmers) == 0:
        print(f"No 3-mers found with maximum enrichment < {max_enrichment} across all samples.")
        return

    top_kmers = df[consistently_depleted_kmers].median(axis=0).sort_values(ascending=True).head(top_n).index
    df_top = df[top_kmers]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(1.2*len(top_kmers), 0.5*len(df_top)+4))
    sns.heatmap(df_top, cmap='Blues', annot=True, fmt=".2f", cbar_kws={'label': 'Enrichment Ratio (Obs/Exp)'})
    plt.title(f'Consistently Depleted 3-mers (max enrichment < {max_enrichment})\nTop {top_n} by lowest median enrichment')
    plt.xlabel('3-mer')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'consistently_depleted_3mer_heatmap_top{top_n}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_5mer_signature_per_sample(context_counts: dict, output_dir: str):
    """
    Plot normalized 5-mer context mutation signature for EMS canonical mutations (C>T and G>A) for each sample.
    context_counts: dict mapping sample -> dict of 5-mer -> count
    output_dir: directory to save plots
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from collections import defaultdict

    for sample, kmer_counts in context_counts.items():
        # Skip 7d/3d samples if desired (but per user, do all samples here)
        c2t_counts = {}
        g2a_counts = {}
        total_c2t = 0
        total_g2a = 0
        for kmer, count in kmer_counts.items():
            if len(kmer) != 5:
                continue
            center = kmer[2]
            if center == 'C':
                c2t_counts[kmer] = count
                total_c2t += count
            elif center == 'G':
                g2a_counts[kmer] = count
                total_g2a += count
        # Normalize
        c2t_norm = {k: v / total_c2t for k, v in c2t_counts.items()} if total_c2t > 0 else {}
        g2a_norm = {k: v / total_g2a for k, v in g2a_counts.items()} if total_g2a > 0 else {}
        # Sort kmers alphabetically for consistent x-axis
        c2t_kmers = sorted(c2t_norm.keys())
        g2a_kmers = sorted(g2a_norm.keys())
        # Plot
        plt.figure(figsize=(max(12, 0.15*max(len(c2t_kmers), len(g2a_kmers))), 6))
        x_c2t = np.arange(len(c2t_kmers))
        x_g2a = np.arange(len(g2a_kmers)) + len(c2t_kmers)  # Offset G>A after C>T
        # Bar plots
        plt.bar(x_c2t, [c2t_norm[k] for k in c2t_kmers], color='orange', label='C>T', width=0.8)
        plt.bar(x_g2a, [g2a_norm[k] for k in g2a_kmers], color='blue', label='G>A', width=0.8)
        # X-ticks
        xticks = list(c2t_kmers) + list(g2a_kmers)
        plt.xticks(np.concatenate([x_c2t, x_g2a]), xticks, rotation=90, fontsize=7)
        plt.xlabel('5-mer Context')
        plt.ylabel('Normalized Frequency')
        plt.title(f'EMS 5-mer Mutation Signature ({sample})')
        plt.legend()
        plt.tight_layout()
        outpath = os.path.join(output_dir, f'ems_5mer_signature_{sample}.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()

def plot_ems_5mer_signature_all_samples(context_counts: dict, output_dir: str):
    """
    Plot normalized 5-mer context mutation signature for EMS canonical mutations (C>T and G>A), aggregated across all samples (excluding 7d/3d).
    context_counts: dict mapping sample -> dict of 5-mer -> count
    output_dir: directory to save plot
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from collections import defaultdict

    c2t_counts = defaultdict(int)
    g2a_counts = defaultdict(int)
    total_c2t = 0
    total_g2a = 0
    for sample, kmer_counts in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
        for kmer, count in kmer_counts.items():
            if len(kmer) != 5:
                continue
            center = kmer[2]
            if center == 'C':
                c2t_counts[kmer] += count
                total_c2t += count
            elif center == 'G':
                g2a_counts[kmer] += count
                total_g2a += count
    # Normalize
    c2t_norm = {k: v / total_c2t for k, v in c2t_counts.items()} if total_c2t > 0 else {}
    g2a_norm = {k: v / total_g2a for k, v in g2a_counts.items()} if total_g2a > 0 else {}
    # Sort kmers alphabetically for consistent x-axis
    c2t_kmers = sorted(c2t_norm.keys())
    g2a_kmers = sorted(g2a_norm.keys())
    # Plot
    plt.figure(figsize=(max(12, 0.15*max(len(c2t_kmers), len(g2a_kmers))), 6))
    x_c2t = np.arange(len(c2t_kmers))
    x_g2a = np.arange(len(g2a_kmers)) + len(c2t_kmers)  # Offset G>A after C>T
    # Bar plots
    plt.bar(x_c2t, [c2t_norm[k] for k in c2t_kmers], color='orange', label='C>T', width=0.8)
    plt.bar(x_g2a, [g2a_norm[k] for k in g2a_kmers], color='blue', label='G>A', width=0.8)
    # X-ticks
    xticks = list(c2t_kmers) + list(g2a_kmers)
    plt.xticks(np.concatenate([x_c2t, x_g2a]), xticks, rotation=90, fontsize=7)
    plt.xlabel('5-mer Context')
    plt.ylabel('Normalized Frequency')
    plt.title('EMS 5-mer Mutation Signature (All Samples, Excl. 7d/3d)')
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'ems_5mer_signature_all_samples.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_3mer_signature_per_sample(context_counts: dict, genome_kmer_counts: dict, output_dir: str):
    """
    Plot normalized 3-mer context mutation signature for EMS canonical mutations (C>T and G>A) for each sample.
    context_counts: dict mapping sample -> dict of 5-mer -> count
    genome_kmer_counts: dict mapping 5-mer -> count
    output_dir: directory to save plots
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from collections import defaultdict

    sample_3mer_counts, _ = _extract_3mer_counts_from_5mers(context_counts, genome_kmer_counts)

    for sample, kmer_counts in sample_3mer_counts.items():
        c2t_counts = {}
        g2a_counts = {}
        total_c2t = 0
        total_g2a = 0
        for kmer, count in kmer_counts.items():
            if len(kmer) != 3:
                continue
            center = kmer[1]
            if center == 'C':
                c2t_counts[kmer] = count
                total_c2t += count
            elif center == 'G':
                g2a_counts[kmer] = count
                total_g2a += count
        # Normalize
        c2t_norm = {k: v / total_c2t for k, v in c2t_counts.items()} if total_c2t > 0 else {}
        g2a_norm = {k: v / total_g2a for k, v in g2a_counts.items()} if total_g2a > 0 else {}
        # Sort kmers alphabetically for consistent x-axis
        c2t_kmers = sorted(c2t_norm.keys())
        g2a_kmers = sorted(g2a_norm.keys())
        # Plot
        plt.figure(figsize=(max(12, 0.15*max(len(c2t_kmers), len(g2a_kmers))), 6))
        x_c2t = np.arange(len(c2t_kmers))
        x_g2a = np.arange(len(g2a_kmers)) + len(c2t_kmers)  # Offset G>A after C>T
        # Bar plots
        plt.bar(x_c2t, [c2t_norm[k] for k in c2t_kmers], color='orange', label='C>T', width=0.8)
        plt.bar(x_g2a, [g2a_norm[k] for k in g2a_kmers], color='blue', label='G>A', width=0.8)
        # X-ticks
        xticks = list(c2t_kmers) + list(g2a_kmers)
        plt.xticks(np.concatenate([x_c2t, x_g2a]), xticks, rotation=90, fontsize=7)
        plt.xlabel('3-mer Context')
        plt.ylabel('Normalized Frequency')
        plt.title(f'EMS 3-mer Mutation Signature ({sample})')
        plt.legend()
        plt.tight_layout()
        outpath = os.path.join(output_dir, f'ems_3mer_signature_{sample}.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()

def plot_ems_3mer_signature_all_samples(context_counts: dict, genome_kmer_counts: dict, output_dir: str):
    """
    Plot normalized 3-mer context mutation signature for EMS canonical mutations (C>T and G>A), aggregated across all samples (excluding 7d/3d).
    context_counts: dict mapping sample -> dict of 5-mer -> count
    genome_kmer_counts: dict mapping 5-mer -> count
    output_dir: directory to save plot
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from collections import defaultdict

    sample_3mer_counts, _ = _extract_3mer_counts_from_5mers(context_counts, genome_kmer_counts)

    c2t_counts = defaultdict(int)
    g2a_counts = defaultdict(int)
    total_c2t = 0
    total_g2a = 0
    for sample, kmer_counts in sample_3mer_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
        for kmer, count in kmer_counts.items():
            if len(kmer) != 3:
                continue
            center = kmer[1]
            if center == 'C':
                c2t_counts[kmer] += count
                total_c2t += count
            elif center == 'G':
                g2a_counts[kmer] += count
                total_g2a += count
    # Normalize
    c2t_norm = {k: v / total_c2t for k, v in c2t_counts.items()} if total_c2t > 0 else {}
    g2a_norm = {k: v / total_g2a for k, v in g2a_counts.items()} if total_g2a > 0 else {}
    # Sort kmers alphabetically for consistent x-axis
    c2t_kmers = sorted(c2t_norm.keys())
    g2a_kmers = sorted(g2a_norm.keys())
    # Plot
    plt.figure(figsize=(max(12, 0.15*max(len(c2t_kmers), len(g2a_kmers))), 6))
    x_c2t = np.arange(len(c2t_kmers))
    x_g2a = np.arange(len(g2a_kmers)) + len(c2t_kmers)  # Offset G>A after C>T
    # Bar plots
    plt.bar(x_c2t, [c2t_norm[k] for k in c2t_kmers], color='orange', label='C>T', width=0.8)
    plt.bar(x_g2a, [g2a_norm[k] for k in g2a_kmers], color='blue', label='G>A', width=0.8)
    # X-ticks
    xticks = list(c2t_kmers) + list(g2a_kmers)
    plt.xticks(np.concatenate([x_c2t, x_g2a]), xticks, rotation=90, fontsize=7)
    plt.xlabel('3-mer Context')
    plt.ylabel('Normalized Frequency')
    plt.title('EMS 3-mer Mutation Signature (All Samples, Excl. 7d/3d)')
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'ems_3mer_signature_all_samples.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_3mer_signature_multiplot(context_counts: dict, genome_kmer_counts: dict, output_dir: str):
    """
    Create a multi-panel plot of normalized 3-mer signatures for all EMS samples (excluding 3d/7d).
    Each subplot shows the normalized 3-mer signature for one sample, with C>T and G>A colored differently.
    X-axis is consistent across all subplots.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from collections import defaultdict

    sample_3mer_counts, _ = _extract_3mer_counts_from_5mers(context_counts, genome_kmer_counts)
    # Select EMS samples without 3d/7d
    ems_samples = [s for s in sample_3mer_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
    if not ems_samples:
        print("No EMS samples found for 3-mer multiplot.")
        return

    # Collect all 3-mers observed in any sample (for consistent x-axis)
    all_c2t_kmers = set()
    all_g2a_kmers = set()
    for sample in ems_samples:
        for kmer in sample_3mer_counts[sample]:
            if len(kmer) == 3:
                center = kmer[1]
                if center == 'C':
                    all_c2t_kmers.add(kmer)
                elif center == 'G':
                    all_g2a_kmers.add(kmer)
    all_c2t_kmers = sorted(all_c2t_kmers)
    all_g2a_kmers = sorted(all_g2a_kmers)
    all_xticks = all_c2t_kmers + all_g2a_kmers
    x_c2t = np.arange(len(all_c2t_kmers))
    x_g2a = np.arange(len(all_g2a_kmers)) + len(all_c2t_kmers)
    x_all = np.concatenate([x_c2t, x_g2a])

    n_samples = len(ems_samples)
    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(12, 4*n_cols), 3*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, sample in enumerate(ems_samples):
        ax = axes[idx]
        kmer_counts = sample_3mer_counts[sample]
        c2t_counts = {k: kmer_counts.get(k, 0) for k in all_c2t_kmers}
        g2a_counts = {k: kmer_counts.get(k, 0) for k in all_g2a_kmers}
        total_c2t = sum(c2t_counts.values())
        total_g2a = sum(g2a_counts.values())
        c2t_norm = [c2t_counts[k] / total_c2t if total_c2t > 0 else 0 for k in all_c2t_kmers]
        g2a_norm = [g2a_counts[k] / total_g2a if total_g2a > 0 else 0 for k in all_g2a_kmers]
        ax.bar(x_c2t, c2t_norm, color='orange', label='C>T', width=0.8)
        ax.bar(x_g2a, g2a_norm, color='blue', label='G>A', width=0.8)
        ax.set_title(sample)
        if idx % n_cols == 0:
            ax.set_ylabel('Normalized Frequency')
        if idx >= (n_rows-1)*n_cols:
            ax.set_xticks(x_all)
            ax.set_xticklabels(all_xticks, rotation=90, fontsize=7)
        else:
            ax.set_xticks([])
        if idx == 0:
            ax.legend()
    # Hide unused axes
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'ems_3mer_signature_multiplot.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_5mer_signature_multiplot(context_counts: dict, output_dir: str):
    """
    Create a multi-panel plot of normalized 5-mer signatures for all EMS samples (excluding 3d/7d).
    Each subplot shows the normalized 5-mer signature for one sample, with C>T and G>A colored differently.
    X-axis is consistent across all subplots. Sample names are shortened and x-axis labels are removed.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from collections import defaultdict

    # Select EMS samples without 3d/7d
    ems_samples = [s for s in context_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
    if not ems_samples:
        print("No EMS samples found for 5-mer multiplot.")
        return

    # Collect all 5-mers observed in any sample (for consistent x-axis)
    all_c2t_kmers = set()
    all_g2a_kmers = set()
    for sample in ems_samples:
        for kmer in context_counts[sample]:
            if len(kmer) == 5:
                center = kmer[2]
                if center == 'C':
                    all_c2t_kmers.add(kmer)
                elif center == 'G':
                    all_g2a_kmers.add(kmer)
    all_c2t_kmers = sorted(all_c2t_kmers)
    all_g2a_kmers = sorted(all_g2a_kmers)
    all_xticks = all_c2t_kmers + all_g2a_kmers
    x_c2t = np.arange(len(all_c2t_kmers))
    x_g2a = np.arange(len(all_g2a_kmers)) + len(all_c2t_kmers)
    x_all = np.concatenate([x_c2t, x_g2a])

    n_samples = len(ems_samples)
    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(12, 4*n_cols), 3*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, sample in enumerate(ems_samples):
        ax = axes[idx]
        kmer_counts = context_counts[sample]
        c2t_counts = {k: kmer_counts.get(k, 0) for k in all_c2t_kmers}
        g2a_counts = {k: kmer_counts.get(k, 0) for k in all_g2a_kmers}
        total_c2t = sum(c2t_counts.values())
        total_g2a = sum(g2a_counts.values())
        c2t_norm = [c2t_counts[k] / total_c2t if total_c2t > 0 else 0 for k in all_c2t_kmers]
        g2a_norm = [g2a_counts[k] / total_g2a if total_g2a > 0 else 0 for k in all_g2a_kmers]
        ax.bar(x_c2t, c2t_norm, color='orange', label='C>T', width=0.8)
        ax.bar(x_g2a, g2a_norm, color='blue', label='G>A', width=0.8)
        # Use shortened sample name
        short_name = extract_sample_id(sample)
        ax.set_title(short_name)
        if idx % n_cols == 0:
            ax.set_ylabel('Normalized Frequency')
        else:
            ax.set_ylabel('')
        # Remove all x-axis tick labels
        ax.set_xticks([])
        if idx == 0:
            ax.legend()
    # Hide unused axes
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'ems_5mer_signature_multiplot.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_consistently_enriched_5mer_kmers_heatmap(context_counts: dict, genome_kmer_counts: dict, output_dir: str, min_enrichment: float = 1.5, top_n: int = 15):
    """
    Plot a heatmap of raw enrichment ratios for 5-mers that are consistently enriched across all EMS samples.
    """
    ems_samples = [sample for sample in context_counts if 'EMS' in sample and '7d' not in sample and '3d' not in sample]
    if not ems_samples:
        print("No EMS samples found for consistently enriched 5-mer heatmap.")
        return
    
    # Get 5-mer genome counts directly from the dictionary
    genome_5mer_counts = {kmer: count for kmer, count in genome_kmer_counts.items() if len(kmer) == 5}
    cg_genome_kmers = {kmer: count for kmer, count in genome_5mer_counts.items() if kmer[2] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())
    
    enrichment_data = {}
    for sample in ems_samples:
        sample_kmers = context_counts[sample]
        ems_mutation_kmers = {kmer: count for kmer, count in sample_kmers.items() if len(kmer) == 5 and kmer[2] in ['C', 'G']}
        total_mutations = sum(ems_mutation_kmers.values())
        enrichment_data[sample] = {}
        for kmer, observed_count in ems_mutation_kmers.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            enrichment_data[sample][kmer] = enrichment_ratio
    
    df = pd.DataFrame.from_dict(enrichment_data, orient='index')
    # Clean sample names
    df.index = [clean_ems_sample_name(sample) for sample in df.index]
    consistently_enriched_kmers = df.columns[(df.min(axis=0) > min_enrichment)]
    if len(consistently_enriched_kmers) == 0:
        print(f"No 5-mers found with minimum enrichment > {min_enrichment} across all samples.")
        return
    top_kmers = df[consistently_enriched_kmers].median(axis=0).sort_values(ascending=False).head(top_n).index
    df_top = df[top_kmers]
    plt.figure(figsize=(1.5*len(top_kmers), 0.6*len(df_top)+4))
    sns.heatmap(df_top, cmap='YlOrRd', annot=True, fmt=".2f", cbar_kws={'label': 'Enrichment Ratio (Obs/Exp)', 'shrink': 0.8, 'aspect': 20}, annot_kws={'size': 12})
    plt.title(f'Consistently Enriched 5-mers (min enrichment > {min_enrichment})\nTop {top_n} by median enrichment', fontsize=16)
    plt.xlabel('5-mer', fontsize=14)
    plt.ylabel('Sample', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Adjust colorbar font sizes
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=12)
    cbar.set_ylabel('Enrichment Ratio (Obs/Exp)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'consistently_enriched_5mer_heatmap_top{top_n}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_consistently_depleted_5mer_kmers_heatmap(context_counts: dict, genome_kmer_counts: dict, output_dir: str, max_enrichment: float = 0.67, top_n: int = 15):
    """
    Plot a heatmap of raw enrichment ratios for 5-mers that are consistently depleted across all EMS samples.
    """
    ems_samples = [sample for sample in context_counts if 'EMS' in sample and '7d' not in sample and '3d' not in sample]
    if not ems_samples:
        print("No EMS samples found for consistently depleted 5-mer heatmap.")
        return
    
    # Get 5-mer genome counts directly from the dictionary
    genome_5mer_counts = {kmer: count for kmer, count in genome_kmer_counts.items() if len(kmer) == 5}
    cg_genome_kmers = {kmer: count for kmer, count in genome_5mer_counts.items() if kmer[2] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())
    
    enrichment_data = {}
    for sample in ems_samples:
        sample_kmers = context_counts[sample]
        ems_mutation_kmers = {kmer: count for kmer, count in sample_kmers.items() if len(kmer) == 5 and kmer[2] in ['C', 'G']}
        total_mutations = sum(ems_mutation_kmers.values())
        enrichment_data[sample] = {}
        for kmer, observed_count in ems_mutation_kmers.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            enrichment_data[sample][kmer] = enrichment_ratio
    
    df = pd.DataFrame.from_dict(enrichment_data, orient='index')
    # Clean sample names
    df.index = [clean_ems_sample_name(sample) for sample in df.index]
    consistently_depleted_kmers = df.columns[(df.max(axis=0) < max_enrichment)]
    if len(consistently_depleted_kmers) == 0:
        print(f"No 5-mers found with maximum enrichment < {max_enrichment} across all samples.")
        return
    top_kmers = df[consistently_depleted_kmers].median(axis=0).sort_values(ascending=True).head(top_n).index
    df_top = df[top_kmers]
    plt.figure(figsize=(1.5*len(top_kmers), 0.6*len(df_top)+4))
    sns.heatmap(df_top, cmap='Blues', annot=True, fmt=".2f", cbar_kws={'label': 'Enrichment Ratio (Obs/Exp)', 'shrink': 0.8, 'aspect': 20}, annot_kws={'size': 12})
    plt.title(f'Consistently Depleted 5-mers (max enrichment < {max_enrichment})\nTop {top_n} by lowest median enrichment', fontsize=16)
    plt.xlabel('5-mer', fontsize=14)
    plt.ylabel('Sample', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Adjust colorbar font sizes
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=12)
    cbar.set_ylabel('Enrichment Ratio (Obs/Exp)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'consistently_depleted_5mer_heatmap_top{top_n}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_mutation_frequencies_consensus_comparison(
    consensus_data_path: str, 
    nonconsensus_data_path: str, 
    output_dir: str
) -> None:
    '''Create multi-panel bar plot comparing mutation frequencies between consensus and nonconsensus datasets.
    
    Args:
        consensus_data_path (str): Path to consensus mutation frequencies CSV
        nonconsensus_data_path (str): Path to nonconsensus mutation frequencies CSV
        output_dir (str): Directory to save output plot
    '''
    output_path = os.path.join(output_dir, 'ems_mutation_frequencies_consensus_comparison.png')
    
    # Load both datasets
    df_consensus = pd.read_csv(consensus_data_path)
    df_nonconsensus = pd.read_csv(nonconsensus_data_path)
    
    # Get common mutations and samples
    mutations = sorted(set(df_consensus['mutation'].unique()) | set(df_nonconsensus['mutation'].unique()))
    all_samples = set(df_consensus['sample'].unique()) | set(df_nonconsensus['sample'].unique())
    
    # Sort samples consistently
    control_samples = sorted([s for s in all_samples if 'EMS' not in s])
    treated_samples = sorted([s for s in all_samples if 'EMS' in s])
    
    # Sort treated samples by total mutation frequency (using consensus data for ordering)
    treated_totals = df_consensus[df_consensus['ems'] == '+'].groupby('sample')['norm_count'].sum()
    treated_samples = treated_totals.sort_values(ascending=True).index.tolist()
    
    # Group treated samples into three bins by mutation rate
    n_samples = len(treated_samples)
    samples_per_group = n_samples // 3
    remainder = n_samples % 3
    
    # Create groups with even distribution of remainder
    group_sizes = [samples_per_group + (1 if i < remainder else 0) for i in range(3)]
    sample_groups = []
    start_idx = 0
    for size in group_sizes:
        sample_groups.append(treated_samples[start_idx:start_idx + size])
        start_idx += size
    
    group_labels = []
    for i, group in enumerate(['Low', 'Medium', 'High']):
        sample_ids = [extract_sample_id(s) for s in sample_groups[i]]
        # Split sample IDs into roughly equal groups
        n = len(sample_ids)
        split_point = (n + 1) // 2
        if n > 4:  # Only split if more than 4 samples
            group_labels.append(f"EMS {', '.join(sample_ids[:split_point])}")
            group_labels.append(f"EMS {', '.join(sample_ids[split_point:])}")
        else:
            group_labels.append(f"EMS {', '.join(sample_ids)}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, sharey=False)
    
    # Plot setup
    bar_width = 0.95 / (len(control_samples) + len(treated_samples))
    group_spacing = 0.15
    x = np.arange(len(mutations)) * (1 + group_spacing)
    
    # Colors for treated samples
    colors = [plt.cm.Oranges(0.3), plt.cm.Oranges(0.6), plt.cm.Oranges(0.9)]
    
    # Plot nonconsensus data (top panel)
    ax1.set_title('Non-consensus', fontsize=24, fontweight='bold')
    
    # Plot control samples
    for i, sample in enumerate(control_samples):
        sample_data = df_nonconsensus[df_nonconsensus['sample'] == sample].set_index('mutation')['norm_count']
        sample_values = [sample_data.get(mut, 0) for mut in mutations]
        pos = x - (0.475) + (i * bar_width)
        ax1.bar(pos, sample_values, bar_width, color='lightgrey', alpha=0.8)
    
    # Plot treated samples with grouped colors
    current_idx = 0
    for group_idx, group_samples in enumerate(sample_groups):
        for i, sample in enumerate(group_samples):
            sample_data = df_nonconsensus[df_nonconsensus['sample'] == sample].set_index('mutation')['norm_count']
            sample_values = [sample_data.get(mut, 0) for mut in mutations]
            pos = x + (current_idx * bar_width)
            ax1.bar(pos, sample_values, bar_width, color=colors[group_idx], alpha=0.8)
            current_idx += 1
    
    ax1.set_ylabel('Normalized Mutation Rate', fontsize=20)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Set y-axis limits for nonconsensus data
    nonconsensus_values = []
    for sample in control_samples + treated_samples:
        sample_data = df_nonconsensus[df_nonconsensus['sample'] == sample].set_index('mutation')['norm_count']
        sample_values = [sample_data.get(mut, 0) for mut in mutations]
        nonconsensus_values.extend(sample_values)
    
    if nonconsensus_values:
        ax1.set_ylim(0, max(nonconsensus_values) * 1.1)  # Add 10% padding
    
    # Plot consensus data (bottom panel)
    ax2.set_title('Consensus', fontsize=24, fontweight='bold')
    
    # Plot control samples
    for i, sample in enumerate(control_samples):
        sample_data = df_consensus[df_consensus['sample'] == sample].set_index('mutation')['norm_count']
        sample_values = [sample_data.get(mut, 0) for mut in mutations]
        pos = x - (0.475) + (i * bar_width)
        ax2.bar(pos, sample_values, bar_width, color='lightgrey', alpha=0.8)
    
    # Plot treated samples with grouped colors
    current_idx = 0
    for group_idx, group_samples in enumerate(sample_groups):
        for i, sample in enumerate(group_samples):
            sample_data = df_consensus[df_consensus['sample'] == sample].set_index('mutation')['norm_count']
            sample_values = [sample_data.get(mut, 0) for mut in mutations]
            pos = x + (current_idx * bar_width)
            ax2.bar(pos, sample_values, bar_width, color=colors[group_idx], alpha=0.8)
            current_idx += 1
    
    ax2.set_ylabel('Normalized Mutation Rate', fontsize=20)
    ax2.set_xlabel('Mutation Type', fontsize=20)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Set y-axis limits for consensus data
    consensus_values = []
    for sample in control_samples + treated_samples:
        sample_data = df_consensus[df_consensus['sample'] == sample].set_index('mutation')['norm_count']
        sample_values = [sample_data.get(mut, 0) for mut in mutations]
        consensus_values.extend(sample_values)
    
    if consensus_values:
        ax2.set_ylim(0, max(consensus_values) * 1.1)  # Add 10% padding
    
    # Set x-axis ticks
    ax2.set_xticks(x)
    ax2.set_xticklabels(mutations, rotation=45, ha='right', fontsize=18)
    
    # Create shared legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, color='lightgrey', alpha=0.8, label='Controls')
    ]
    
    # Add legend entries for each subgroup
    current_color_idx = 0
    for i in range(len(group_labels)):
        legend_elements.append(
            plt.Rectangle((0,0), 1, 1, 
                         color=colors[current_color_idx], 
                         alpha=0.8,
                         label=group_labels[i])
        )
        if i % 2 == 1:  # Increment color index after each pair
            current_color_idx += 1
    
    # Add legend to the bottom plot (ax2) in the white space
    ax2.legend(
        handles=legend_elements, 
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        framealpha=0.9,
        fontsize=18
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def mutation_type_barplot_consensus_comparison(
    consensus_json_files: list, 
    nonconsensus_json_files: list, 
    base_counts: dict, 
    output_dir: str
) -> None:
    '''Generate mutation type frequency barplot comparing consensus and nonconsensus datasets.
    
    Args:
        consensus_json_files (list): List of paths to consensus mutation JSON files
        nonconsensus_json_files (list): List of paths to nonconsensus mutation JSON files
        base_counts (dict): Dictionary of genome-wide base counts {'A': count, 'T': count, ...}
        output_dir (str): Directory to save output plot
    '''
    
    # Count mutations across all samples for both datasets
    # Consensus: count alternate alleles (actual mutation frequency)
    # Nonconsensus: count unique sites (avoid counting false positives multiple times)
    consensus_freqs = count_mutations_alt_alleles(consensus_json_files)
    nonconsensus_freqs = count_mutations_unique_sites(nonconsensus_json_files)
    
    # Normalize using provided base counts and generate CSVs
    consensus_csv = normalize_mutation_counts(consensus_freqs, base_counts, output_dir, '_consensus')
    nonconsensus_csv = normalize_mutation_counts(nonconsensus_freqs, base_counts, output_dir, '_nonconsensus')
    
    # Generate the comparison plot
    plot_ems_mutation_frequencies_consensus_comparison(consensus_csv, nonconsensus_csv, output_dir)
    
    # Also generate individual plots for backward compatibility
    plot_ems_mutation_frequencies(consensus_csv, output_dir)
    plot_ems_mutation_frequencies_per_sample(consensus_csv, output_dir)

def mutation_type_barplot(json_files: list, base_counts: dict, output_dir: str) -> None:
    '''Generate mutation type frequency barplot.
    
    Args:
        json_files (list): List of paths to mutation JSON files
        base_counts (dict): Dictionary of genome-wide base counts {'A': count, 'T': count, ...}
        output_dir (str): Directory to save output plot
    '''
    
    # Count mutations across all samples
    total_freqs = count_mutations(json_files)
    
    # Normalize using provided base counts and generate CSV
    mutation_frequency_csv = normalize_mutation_counts(total_freqs, base_counts, output_dir, '')
    
    # Generate plots using the CSV
    plot_ems_mutation_frequencies(mutation_frequency_csv, output_dir)
    plot_ems_mutation_frequencies_per_sample(mutation_frequency_csv, output_dir)

def plot_ems_3mer_signature_grouped_bar(context_counts: dict, genome_kmer_counts: dict, output_dir: str, ax=None, fontsize=16, metric='frequency'):
    """
    Create a grouped bar plot of normalized 3-mer signatures for all EMS samples (excluding 3d/7d).
    Each 3-mer context has bars for each sample, grouped by mutation type (C>T vs G>A).
    If ax is provided, plot both C>T and G>A as subpanels in the same axis (stacked bars, not two subplots). If ax is None, create the two-panel figure as before and save PNG.
    The 'metric' argument controls whether to plot 'frequency' (default) or 'enrichment' on the Y axis.
    """
    import numpy as np
    sample_3mer_counts, _ = _extract_3mer_counts_from_5mers(context_counts, genome_kmer_counts)
    ems_samples = [s for s in sample_3mer_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
    if not ems_samples:
        print("No EMS samples found for 3-mer grouped bar plot.")
        return
    all_c2t_kmers = set()
    all_g2a_kmers = set()
    for sample in ems_samples:
        for kmer in sample_3mer_counts[sample]:
            if len(kmer) == 3:
                center = kmer[1]
                if center == 'C':
                    all_c2t_kmers.add(kmer)
                elif center == 'G':
                    all_g2a_kmers.add(kmer)
    all_c2t_kmers = sorted(all_c2t_kmers)
    
    # Order G>A kmers as reverse complements of C>T kmers
    def reverse_complement(kmer):
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join(complement[base] for base in reversed(kmer))
    
    # Create a mapping from C>T kmers to their reverse complement G>A kmers
    c2t_to_g2a = {}
    for c2t_kmer in all_c2t_kmers:
        rc_g2a = reverse_complement(c2t_kmer)
        if rc_g2a in all_g2a_kmers:
            c2t_to_g2a[c2t_kmer] = rc_g2a
    
    # Order G>A kmers to match the reverse complement order of C>T kmers
    all_g2a_kmers_ordered = []
    for c2t_kmer in all_c2t_kmers:
        if c2t_kmer in c2t_to_g2a:
            all_g2a_kmers_ordered.append(c2t_to_g2a[c2t_kmer])
    
    # Add any remaining G>A kmers that don't have C>T counterparts
    remaining_g2a = [kmer for kmer in all_g2a_kmers if kmer not in all_g2a_kmers_ordered]
    all_g2a_kmers_ordered.extend(sorted(remaining_g2a))
    
    all_g2a_kmers = all_g2a_kmers_ordered
    ylabel = 'Enrichment Ratio (Obs/Exp)' if metric == 'enrichment' else 'Normalized Frequency'
    if ax is not None:
        n_c2t = len(all_c2t_kmers)
        n_g2a = len(all_g2a_kmers)
        gap = 0
        x_c2t = np.arange(n_c2t)
        x_g2a = np.arange(n_g2a) + n_c2t + gap
        x_all = np.concatenate([x_c2t, x_g2a])
        # Create tick labels that match the x-axis positions exactly
        all_xticks = []
        for i in range(len(x_all)):
            if i < len(all_c2t_kmers):
                all_xticks.append(all_c2t_kmers[i])
            elif i >= len(all_c2t_kmers) + gap:
                g2a_index = i - len(all_c2t_kmers) - gap
                if g2a_index < len(all_g2a_kmers):
                    all_xticks.append(all_g2a_kmers[g2a_index])
                else:
                    all_xticks.append('')
            else:
                all_xticks.append('')
        width = 0.8 / len(ems_samples)
        for i, sample in enumerate(ems_samples):
            kmer_counts = sample_3mer_counts[sample]
            c2t_counts = {k: kmer_counts.get(k, 0) for k in all_c2t_kmers}
            total_c2t = sum(c2t_counts.values())
            c2t_norm = [c2t_counts[k] / total_c2t if total_c2t > 0 else 0 for k in all_c2t_kmers]
            pos = x_c2t + (i - len(ems_samples)/2 + 0.5) * width
            ax.bar(pos, c2t_norm, width, label=f'{extract_sample_id(sample)} C>T', alpha=0.8, color='orange')
        for i, sample in enumerate(ems_samples):
            kmer_counts = sample_3mer_counts[sample]
            g2a_counts = {k: kmer_counts.get(k, 0) for k in all_g2a_kmers}
            total_g2a = sum(g2a_counts.values())
            g2a_norm = [g2a_counts[k] / total_g2a if total_g2a > 0 else 0 for k in all_g2a_kmers]
            pos = x_g2a + (i - len(ems_samples)/2 + 0.5) * width
            ax.bar(pos, g2a_norm, width, label=f'{extract_sample_id(sample)} G>A', alpha=0.8, color='blue')
        # Ensure tick positions and labels match (already handled above)
        ax.set_xticks(x_all)
        ax.set_xticklabels(all_xticks, rotation=60, ha='center', fontsize=fontsize-4)
        

        
        # Adjust tick label positioning to prevent clipping
        ax.tick_params(axis='x', pad=8)  # Moderate padding for x-axis ticks

        ax.set_ylabel(ylabel, fontsize=fontsize)
        # No x-axis label for panel A in multipanel
        # ax.set_xlabel('3-mer Context', fontsize=fontsize)
        ax.legend(fontsize=fontsize-4, ncol=2, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        return
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(16, 0.3*len(all_c2t_kmers)), 12), sharex=False)
    # Plot C>T mutations (top panel)
    if all_c2t_kmers:
        x = np.arange(len(all_c2t_kmers))
        width = 0.8 / len(ems_samples)
        for i, sample in enumerate(ems_samples):
            kmer_counts = sample_3mer_counts[sample]
            c2t_counts = {k: kmer_counts.get(k, 0) for k in all_c2t_kmers}
            total_c2t = sum(c2t_counts.values())
            c2t_norm = [c2t_counts[k] / total_c2t if total_c2t > 0 else 0 for k in all_c2t_kmers]
            pos = x + (i - len(ems_samples)/2 + 0.5) * width
            ax1.bar(pos, c2t_norm, width, label=extract_sample_id(sample), alpha=0.8)
        ax1.set_ylabel('Normalized Frequency', fontsize=fontsize)
        ax1.set_title('C>T 3-mer Mutation Signatures by Sample', fontsize=fontsize+2, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_c2t_kmers, rotation=45, ha='right', fontsize=fontsize-4)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize-4)
        ax1.grid(axis='y', alpha=0.3)
    # Plot G>A mutations (bottom panel)
    if all_g2a_kmers:
        x = np.arange(len(all_g2a_kmers))
        width = 0.8 / len(ems_samples)
        for i, sample in enumerate(ems_samples):
            kmer_counts = sample_3mer_counts[sample]
            g2a_counts = {k: kmer_counts.get(k, 0) for k in all_g2a_kmers}
            total_g2a = sum(g2a_counts.values())
            g2a_norm = [g2a_counts[k] / total_g2a if total_g2a > 0 else 0 for k in all_g2a_kmers]
            pos = x + (i - len(ems_samples)/2 + 0.5) * width
            ax2.bar(pos, g2a_norm, width, label=extract_sample_id(sample), alpha=0.8)
        ax2.set_ylabel('Normalized Frequency', fontsize=fontsize)
        ax2.set_xlabel('3-mer Context', fontsize=fontsize)
        ax2.set_title('G>A 3-mer Mutation Signatures by Sample', fontsize=fontsize+2, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_g2a_kmers, rotation=45, ha='right', fontsize=fontsize-4)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize-4)
        ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'ems_3mer_signature_grouped_bar.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_5mer_signature_multiplot_9samples(context_counts: dict, output_dir: str, ax=None, fontsize=16, c2t_kmers_sorted=None, g2a_kmers_sorted=None):
    """
    Create a multi-panel plot of normalized 5-mer signatures for exactly 9 EMS samples (excluding 3d/7d).
    Each subplot shows the normalized 5-mer signature for one sample, with C>T and G>A colored differently.
    X-axis is consistent across all subplots, and can be sorted by a provided order.
    If ax is provided, plot to it as a single panel (for multipanel use). If ax is None, create the 3x3 grid as before and save PNG.
    """
    import numpy as np
    ems_samples = [s for s in context_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
    if not ems_samples:
        print("No EMS samples found for 5-mer multiplot.")
        return
    ems_samples = ems_samples[:9]
    # Use provided kmer order if given, else default to sorted
    if c2t_kmers_sorted is None or g2a_kmers_sorted is None:
        all_c2t_kmers = set()
        all_g2a_kmers = set()
        for sample in ems_samples:
            for kmer in context_counts[sample]:
                if len(kmer) == 5:
                    center = kmer[2]
                    if center == 'C':
                        all_c2t_kmers.add(kmer)
                    elif center == 'G':
                        all_g2a_kmers.add(kmer)
        c2t_kmers_sorted = sorted(all_c2t_kmers)
        g2a_kmers_sorted = sorted(all_g2a_kmers)
    x_c2t = np.arange(len(c2t_kmers_sorted))
    x_g2a = np.arange(len(g2a_kmers_sorted)) + len(c2t_kmers_sorted)
    x_all = np.concatenate([x_c2t, x_g2a])
    n_samples = len(ems_samples)
    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))
    if ax is not None:
        for idx, sample in enumerate(ems_samples):
            kmer_counts = context_counts[sample]
            c2t_counts = {k: kmer_counts.get(k, 0) for k in c2t_kmers_sorted}
            g2a_counts = {k: kmer_counts.get(k, 0) for k in g2a_kmers_sorted}
            total_c2t = sum(c2t_counts.values())
            total_g2a = sum(g2a_counts.values())
            c2t_norm = [c2t_counts[k] / total_c2t if total_c2t > 0 else 0 for k in c2t_kmers_sorted]
            g2a_norm = [g2a_counts[k] / total_g2a if total_g2a > 0 else 0 for k in g2a_kmers_sorted]
            ax.plot(x_c2t, c2t_norm, label=f'{extract_sample_id(sample)} C>T', color='orange', alpha=0.5)
            ax.plot(x_g2a, g2a_norm, label=f'{extract_sample_id(sample)} G>A', color='blue', alpha=0.5)
        ax.set_ylabel('Normalized Frequency', fontsize=fontsize+6)
        ax.set_xlabel('5-mer Context', fontsize=fontsize+6)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=fontsize+4)
        ax.tick_params(axis='x', labelsize=fontsize+4)
        ax.legend(fontsize=fontsize+2, ncol=2, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(16, 4*n_cols), 3*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, sample in enumerate(ems_samples):
        ax_ = axes[idx]
        kmer_counts = context_counts[sample]
        c2t_counts = {k: kmer_counts.get(k, 0) for k in c2t_kmers_sorted}
        g2a_counts = {k: kmer_counts.get(k, 0) for k in g2a_kmers_sorted}
        total_c2t = sum(c2t_counts.values())
        total_g2a = sum(g2a_counts.values())
        c2t_norm = [c2t_counts[k] / total_c2t if total_c2t > 0 else 0 for k in c2t_kmers_sorted]
        g2a_norm = [g2a_counts[k] / total_g2a if total_g2a > 0 else 0 for k in g2a_kmers_sorted]
        ax_.bar(x_c2t, c2t_norm, color='orange', label='C>T', width=0.8)
        ax_.bar(x_g2a, g2a_norm, color='blue', label='G>A', width=0.8)
        short_name = extract_sample_id(sample)
        ax_.set_title(short_name, fontsize=fontsize+2)
        if idx % n_cols == 0:
            ax_.set_ylabel('Normalized Frequency', fontsize=fontsize+2)
        else:
            ax_.set_ylabel('')
        ax_.set_xticks([])
        ax_.tick_params(axis='y', labelsize=fontsize+2)
        ax_.tick_params(axis='x', labelsize=fontsize+2)
        if idx == 0:
            ax_.legend(fontsize=fontsize+2)
        ax_.grid(axis='y', alpha=0.3)
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'ems_5mer_signature_multiplot_9samples.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_5mer_signature_average_with_labels(context_counts: dict, genome_kmer_counts: dict, output_dir: str):
    """
    Plot normalized 5-mer context mutation signature for EMS canonical mutations (C>T and G>A), 
    averaged across all samples (excluding 7d/3d), with text labels on the most enriched sequence contexts.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from collections import defaultdict

    c2t_counts = defaultdict(int)
    g2a_counts = defaultdict(int)
    total_c2t = 0
    total_g2a = 0
    
    # Collect data from all samples
    for sample, kmer_counts in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
        for kmer, count in kmer_counts.items():
            if len(kmer) != 5:
                continue
            center = kmer[2]
            if center == 'C':
                c2t_counts[kmer] += count
                total_c2t += count
            elif center == 'G':
                g2a_counts[kmer] += count
                total_g2a += count
    
    # Normalize
    c2t_norm = {k: v / total_c2t for k, v in c2t_counts.items()} if total_c2t > 0 else {}
    g2a_norm = {k: v / total_g2a for k, v in g2a_counts.items()} if total_g2a > 0 else {}
    
    # Sort kmers alphabetically for consistent x-axis
    c2t_kmers = sorted(c2t_norm.keys())
    g2a_kmers = sorted(g2a_norm.keys())
    
    # Create the plot with more compact width
    plt.figure(figsize=(12, 8))  # Fixed width instead of dynamic scaling
    x_c2t = np.arange(len(c2t_kmers))
    x_g2a = np.arange(len(g2a_kmers)) + len(c2t_kmers)  # Offset G>A after C>T
    
    # Bar plots
    c2t_bars = plt.bar(x_c2t, [c2t_norm[k] for k in c2t_kmers], color='orange', label='C>T', width=0.8)
    g2a_bars = plt.bar(x_g2a, [g2a_norm[k] for k in g2a_kmers], color='blue', label='G>A', width=0.8)
    
    # Calculate enrichment ratios per sample, then average (like heatmap functions)
    # Get 5-mer genome counts
    genome_5mer_counts = {kmer: count for kmer, count in genome_kmer_counts.items() if len(kmer) == 5}
    cg_genome_kmers = {kmer: count for kmer, count in genome_5mer_counts.items() if kmer[2] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())
    
    # Calculate enrichment ratios per sample, then average
    c2t_enrichment_data = defaultdict(list)
    g2a_enrichment_data = defaultdict(list)
    
    for sample, kmer_counts in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
            
        # Get C>T and G>A mutations for this sample
        sample_c2t = {kmer: count for kmer, count in kmer_counts.items() 
                     if len(kmer) == 5 and kmer[2] == 'C'}
        sample_g2a = {kmer: count for kmer, count in kmer_counts.items() 
                     if len(kmer) == 5 and kmer[2] == 'G'}
        
        # Use total mutations (C>T + G>A) as denominator, like heatmap function
        total_sample_mutations = sum(sample_c2t.values()) + sum(sample_g2a.values())
        
        # Calculate enrichment for C>T mutations in this sample
        for kmer, observed_count in sample_c2t.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_sample_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_sample_mutations  # Use total mutations, not just C>T
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            c2t_enrichment_data[kmer].append(enrichment_ratio)
        
        # Calculate enrichment for G>A mutations in this sample
        for kmer, observed_count in sample_g2a.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_sample_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_sample_mutations  # Use total mutations, not just G>A
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            g2a_enrichment_data[kmer].append(enrichment_ratio)
    
    # Average enrichment ratios across samples
    c2t_enrichment = {kmer: np.nanmean(ratios) for kmer, ratios in c2t_enrichment_data.items() if ratios}
    g2a_enrichment = {kmer: np.nanmean(ratios) for kmer, ratios in g2a_enrichment_data.items() if ratios}
    

    
    # Find top enriched contexts for labeling (top 5 for each type)
    c2t_sorted = sorted(c2t_enrichment.items(), key=lambda x: x[1], reverse=True)
    top_c2t = c2t_sorted[:5]  # Top 5 most enriched
    
    g2a_sorted = sorted(g2a_enrichment.items(), key=lambda x: x[1], reverse=True)
    top_g2a = g2a_sorted[:5]  # Top 5 most enriched
    
    # Add text labels for top enriched contexts with smart positioning to avoid overlap
    def add_labels_with_offset(kmers_data, kmer_list, x_coords, bar_heights, color, max_offset=50):
        """Add labels with smart positioning to avoid overlap."""
        used_positions = []
        
        for kmer, enrichment in kmers_data:
            if kmer in kmer_list:
                idx = kmer_list.index(kmer)
                if idx < len(x_coords) and idx < len(bar_heights):
                    x_pos = x_coords[idx]
                    y_pos = bar_heights[idx]
                    
                    # Find a good offset that doesn't overlap with existing labels
                    offset = 10
                    while any(abs(x_pos - used_x) < 2 and abs(offset - used_offset) < 20 
                             for used_x, used_offset in used_positions):
                        offset += 15
                        if offset > max_offset:
                            offset = 10  # Reset and try different positioning
                            break
                    
                    used_positions.append((x_pos, offset))
                    
                    plt.annotate(f'{kmer}\n{enrichment:.1f}x', 
                                xy=(x_pos, y_pos), 
                                xytext=(0, offset), 
                                textcoords='offset points',
                                ha='center', va='bottom',
                                fontsize=8, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                                              color='black', alpha=0.6, lw=0.8))
    
    # Add labels for C>T mutations
    add_labels_with_offset(top_c2t, c2t_kmers, x_c2t, [c2t_norm[k] for k in c2t_kmers], 'yellow')
    
    # Add labels for G>A mutations
    add_labels_with_offset(top_g2a, g2a_kmers, x_g2a, [g2a_norm[k] for k in g2a_kmers], 'lightblue')
    
    # Remove x-axis ticks and labels (they're unreadable anyway)
    plt.xticks([])
    plt.xlabel('5-mer Context (C>T | G>A)', fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.title('EMS 5-mer Mutation Signature (All Samples Average)\nTop 4 Most Enriched Contexts Labeled', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'ems_5mer_signature_average_with_labels.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_5mer_signature_average_colored_by_enrichment(context_counts: dict, genome_kmer_counts: dict, output_dir: str):
    """
    Plot normalized 5-mer context mutation signature for EMS canonical mutations (C>T and G>A), 
    averaged across all samples (excluding 7d/3d), with colors based on enrichment values.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from collections import defaultdict

    c2t_counts = defaultdict(int)
    g2a_counts = defaultdict(int)
    total_c2t = 0
    total_g2a = 0
    
    # Collect data from all samples
    for sample, kmer_counts in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
        for kmer, count in kmer_counts.items():
            if len(kmer) != 5:
                continue
            center = kmer[2]
            if center == 'C':
                c2t_counts[kmer] += count
                total_c2t += count
            elif center == 'G':
                g2a_counts[kmer] += count
                total_g2a += count
    
    # Normalize
    c2t_norm = {k: v / total_c2t for k, v in c2t_counts.items()} if total_c2t > 0 else {}
    g2a_norm = {k: v / total_g2a for k, v in g2a_counts.items()} if total_g2a > 0 else {}
    
    # Sort kmers alphabetically for consistent x-axis
    c2t_kmers = sorted(c2t_norm.keys())
    g2a_kmers = sorted(g2a_norm.keys())
    
    # Calculate enrichment ratios per sample, then average (like heatmap functions)
    # Get 5-mer genome counts
    genome_5mer_counts = {kmer: count for kmer, count in genome_kmer_counts.items() if len(kmer) == 5}
    cg_genome_kmers = {kmer: count for kmer, count in genome_5mer_counts.items() if kmer[2] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())
    
    # Calculate enrichment ratios per sample, then average
    c2t_enrichment_data = defaultdict(list)
    g2a_enrichment_data = defaultdict(list)
    
    for sample, kmer_counts in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
            
        # Get C>T and G>A mutations for this sample
        sample_c2t = {kmer: count for kmer, count in kmer_counts.items() 
                     if len(kmer) == 5 and kmer[2] == 'C'}
        sample_g2a = {kmer: count for kmer, count in kmer_counts.items() 
                     if len(kmer) == 5 and kmer[2] == 'G'}
        
        # Use total mutations (C>T + G>A) as denominator, like heatmap function
        total_sample_mutations = sum(sample_c2t.values()) + sum(sample_g2a.values())
        
        # Calculate enrichment for C>T mutations in this sample
        for kmer, observed_count in sample_c2t.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_sample_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_sample_mutations  # Use total mutations, not just C>T
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            c2t_enrichment_data[kmer].append(enrichment_ratio)
        
        # Calculate enrichment for G>A mutations in this sample
        for kmer, observed_count in sample_g2a.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_sample_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_sample_mutations  # Use total mutations, not just G>A
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            g2a_enrichment_data[kmer].append(enrichment_ratio)
    
    # Average enrichment ratios across samples
    c2t_enrichment = {kmer: np.nanmean(ratios) for kmer, ratios in c2t_enrichment_data.items() if ratios}
    g2a_enrichment = {kmer: np.nanmean(ratios) for kmer, ratios in g2a_enrichment_data.items() if ratios}
    
    # Create the plot with more compact width
    plt.figure(figsize=(12, 8))  # Fixed width instead of dynamic scaling
    x_c2t = np.arange(len(c2t_kmers))
    x_g2a = np.arange(len(g2a_kmers)) + len(c2t_kmers)  # Offset G>A after C>T
    
    # Get enrichment values for coloring
    c2t_enrichment_values = [c2t_enrichment.get(kmer, 1.0) for kmer in c2t_kmers]
    g2a_enrichment_values = [g2a_enrichment.get(kmer, 1.0) for kmer in g2a_kmers]
    
    # Create color maps for enrichment values
    max_enrichment = max(max(c2t_enrichment_values), max(g2a_enrichment_values))
    min_enrichment = min(min(c2t_enrichment_values), min(g2a_enrichment_values))
    
    # Normalize enrichment values for coloring (1.0 = white, higher = more saturated)
    def normalize_enrichment(e, max_e):
        if max_e <= 1.0:
            return 0.0
        return min(1.0, max(0.0, (e - 1.0) / (max_e - 1.0)))
    
    c2t_colors = []
    for e in c2t_enrichment_values:
        norm_e = normalize_enrichment(e, max_enrichment)
        # Orange gradient: white to orange
        r = 1.0
        g = max(0.0, min(1.0, 1.0 - 0.6 * norm_e))  # Keep more green for orange
        b = max(0.0, min(1.0, 1.0 - 0.8 * norm_e))
        c2t_colors.append((r, g, b))
    
    g2a_colors = []
    for e in g2a_enrichment_values:
        norm_e = normalize_enrichment(e, max_enrichment)
        # Blue gradient: white to blue
        r = max(0.0, min(1.0, 1.0 - 0.8 * norm_e))
        g = max(0.0, min(1.0, 1.0 - 0.8 * norm_e))
        b = 1.0
        g2a_colors.append((r, g, b))
    
    # Bar plots with colors based on enrichment
    for i, (kmer, freq) in enumerate(zip(c2t_kmers, [c2t_norm[k] for k in c2t_kmers])):
        plt.bar(x_c2t[i], freq, color=c2t_colors[i], width=0.8, alpha=0.8)
    
    for i, (kmer, freq) in enumerate(zip(g2a_kmers, [g2a_norm[k] for k in g2a_kmers])):
        plt.bar(x_g2a[i], freq, color=g2a_colors[i], width=0.8, alpha=0.8)
    
    # Remove x-axis ticks and labels (they're unreadable anyway)
    plt.xticks([])
    plt.xlabel('5-mer Context (C>T | G>A)', fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.title('EMS 5-mer Mutation Signature (All Samples Average)\nColors Indicate Enrichment Values', fontsize=14, fontweight='bold')
    
    # Add legend to main plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', alpha=0.8, label='C>T Mutations'),
        Patch(facecolor='blue', alpha=0.8, label='G>A Mutations')
    ]
    plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
    plt.figtext(0.02, 0.02, 'Color intensity indicates enrichment ratio', 
                fontsize=10, style='italic', ha='left', va='bottom')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'ems_5mer_signature_average_colored_by_enrichment.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ems_5mer_signature_average_top3_peaks_labeled(context_counts: dict, genome_kmer_counts: dict, output_dir: str, ax=None, fontsize=16, metric='frequency'):
    """
    Plot normalized 5-mer context mutation signature for EMS canonical mutations (C>T and G>A),
    averaged across all samples (excluding 7d/3d), with labels for the top 3 by enrichment only.
    If ax is provided, plot to it; otherwise, create a new figure.
    The 'metric' argument controls whether to plot 'frequency' (default) or 'enrichment' on the Y axis.
    Bars are sorted by the chosen metric within each group, but C>T and G>A are kept separate.
    Returns the sorted kmer order for C>T and G>A.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    c2t_counts = defaultdict(int)
    g2a_counts = defaultdict(int)
    total_c2t = 0
    total_g2a = 0
    # Collect data from all samples
    for sample, kmer_counts in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
        for kmer, count in kmer_counts.items():
            if len(kmer) != 5:
                continue
            center = kmer[2]
            if center == 'C':
                c2t_counts[kmer] += count
                total_c2t += count
            elif center == 'G':
                g2a_counts[kmer] += count
                total_g2a += count
    # Normalize
    c2t_norm = {k: v / total_c2t for k, v in c2t_counts.items()} if total_c2t > 0 else {}
    g2a_norm = {k: v / total_g2a for k, v in g2a_counts.items()} if total_g2a > 0 else {}
    # Calculate enrichment ratios per sample, then average (like heatmap functions)
    genome_5mer_counts = {kmer: count for kmer, count in genome_kmer_counts.items() if len(kmer) == 5}
    cg_genome_kmers = {kmer: count for kmer, count in genome_5mer_counts.items() if kmer[2] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())
    c2t_enrichment_data = defaultdict(list)
    g2a_enrichment_data = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
        sample_c2t = {kmer: count for kmer, count in kmer_counts.items() if len(kmer) == 5 and kmer[2] == 'C'}
        sample_g2a = {kmer: count for kmer, count in kmer_counts.items() if len(kmer) == 5 and kmer[2] == 'G'}
        total_sample_mutations = sum(sample_c2t.values()) + sum(sample_g2a.values())
        for kmer, observed_count in sample_c2t.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_sample_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_sample_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            c2t_enrichment_data[kmer].append(enrichment_ratio)
        for kmer, observed_count in sample_g2a.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_sample_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_sample_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            g2a_enrichment_data[kmer].append(enrichment_ratio)
    c2t_enrichment = {kmer: np.nanmean(ratios) for kmer, ratios in c2t_enrichment_data.items() if ratios}
    g2a_enrichment = {kmer: np.nanmean(ratios) for kmer, ratios in g2a_enrichment_data.items() if ratios}
    # Choose metric for sorting and plotting
    if metric == 'enrichment':
        c2t_metric = c2t_enrichment
        g2a_metric = g2a_enrichment
        ylabel = 'Enrichment Ratio (Obs/Exp)'
    else:
        c2t_metric = c2t_norm
        g2a_metric = g2a_norm
        ylabel = 'Normalized Frequency'
    # Sort kmers by metric within each group
    c2t_kmers_sorted = [k for k, v in sorted(c2t_metric.items(), key=lambda x: x[1], reverse=True)]
    g2a_kmers_sorted = [k for k, v in sorted(g2a_metric.items(), key=lambda x: x[1], reverse=True)]
    # Prepare y values for plotting
    c2t_y = [c2t_metric[k] for k in c2t_kmers_sorted]
    g2a_y = [g2a_metric[k] for k in g2a_kmers_sorted]
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    x_c2t = np.arange(len(c2t_kmers_sorted))
    x_g2a = np.arange(len(g2a_kmers_sorted)) + len(c2t_kmers_sorted)
    c2t_bars = ax.bar(x_c2t, c2t_y, color='orange', label='C>T', width=0.8)
    g2a_bars = ax.bar(x_g2a, g2a_y, color='blue', label='G>A', width=0.8)
    # For labeling, use only the top 3 enrichment peaks for each type
    c2t_freq_data = [(kmer, c2t_norm.get(kmer, 0), c2t_enrichment.get(kmer, 1.0)) for kmer in c2t_kmers_sorted]
    g2a_freq_data = [(kmer, g2a_norm.get(kmer, 0), g2a_enrichment.get(kmer, 1.0)) for kmer in g2a_kmers_sorted]
    c2t_top3_enrichment = sorted(c2t_freq_data, key=lambda x: x[2], reverse=True)[:3]
    g2a_top3_enrichment = sorted(g2a_freq_data, key=lambda x: x[2], reverse=True)[:3]
    def add_labels_for_enrichment_only(top_enrichment, kmer_list, x_coords, bar_heights):
        fig = ax.figure
        base_offset_y = 110
        base_offset_x = 0
        label_spacing_x = 250  # pixels
        label_spacing_y = 0    # keep all at the same height
        for i, (kmer, freq, enrichment) in enumerate(top_enrichment):
            if kmer in kmer_list:
                idx = kmer_list.index(kmer)
                if idx < len(x_coords) and idx < len(bar_heights):
                    x_pos = x_coords[idx]
                    y_pos = bar_heights[idx]
                    label_text = f'{kmer}\n{enrichment:.1f}x'
                    offset_x = base_offset_x + i * label_spacing_x
                    offset_y = base_offset_y + i * label_spacing_y
                    ax.annotate(label_text,
                        xy=(x_pos, y_pos),
                        xytext=(offset_x, offset_y),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=fontsize+4, fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=1),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='black', alpha=0.7, lw=1.0))
    add_labels_for_enrichment_only(c2t_top3_enrichment, c2t_kmers_sorted, x_c2t, c2t_y)
    add_labels_for_enrichment_only(g2a_top3_enrichment, g2a_kmers_sorted, x_g2a, g2a_y)
    # Set ylim to leave space for labels above bars
    max_y = max(c2t_y + g2a_y) if (c2t_y + g2a_y) else 1
    ax.set_ylim(0, max_y * 1.55)
    ax.set_xticks([])
    ax.set_xlabel('5-mer Context (C>T | G>A)', fontsize=fontsize+8)
    ax.set_ylabel(ylabel, fontsize=fontsize+8)
    if ax is None:
        ax.set_title('5-mer Mutation Signature: Top Peaks', fontsize=fontsize+12, fontweight='bold', pad=40)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', alpha=0.8, label='C>T Mutations'),
        Patch(facecolor='blue', alpha=0.8, label='G>A Mutations'),
        Patch(facecolor='lightblue', alpha=0.7, label='Enrichment Values')
    ]
    ax.legend(handles=legend_elements, fontsize=fontsize+4, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    if ax is None:
        plt.tight_layout()
        outpath = os.path.join(output_dir, 'ems_5mer_signature_average_top3_peaks_labeled.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return c2t_kmers_sorted, g2a_kmers_sorted

def get_sorted_5mer_kmer_order(context_counts: dict, genome_kmer_counts: dict, metric='frequency'):
    """
    Return the sorted 5-mer kmer order for C>T and G>A, as used in the average/top-peaks plot, without plotting.
    """
    import numpy as np
    from collections import defaultdict
    c2t_counts = defaultdict(int)
    g2a_counts = defaultdict(int)
    total_c2t = 0
    total_g2a = 0
    for sample, kmer_counts in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
        for kmer, count in kmer_counts.items():
            if len(kmer) != 5:
                continue
            center = kmer[2]
            if center == 'C':
                c2t_counts[kmer] += count
                total_c2t += count
            elif center == 'G':
                g2a_counts[kmer] += count
                total_g2a += count
    c2t_norm = {k: v / total_c2t for k, v in c2t_counts.items()} if total_c2t > 0 else {}
    g2a_norm = {k: v / total_g2a for k, v in g2a_counts.items()} if total_g2a > 0 else {}
    genome_5mer_counts = {kmer: count for kmer, count in genome_kmer_counts.items() if len(kmer) == 5}
    cg_genome_kmers = {kmer: count for kmer, count in genome_5mer_counts.items() if kmer[2] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())
    c2t_enrichment_data = defaultdict(list)
    g2a_enrichment_data = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
        if '7d' in sample or '3d' in sample:
            continue
        sample_c2t = {kmer: count for kmer, count in kmer_counts.items() if len(kmer) == 5 and kmer[2] == 'C'}
        sample_g2a = {kmer: count for kmer, count in kmer_counts.items() if len(kmer) == 5 and kmer[2] == 'G'}
        total_sample_mutations = sum(sample_c2t.values()) + sum(sample_g2a.values())
        for kmer, observed_count in sample_c2t.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_sample_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_sample_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            c2t_enrichment_data[kmer].append(enrichment_ratio)
        for kmer, observed_count in sample_g2a.items():
            genome_count = cg_genome_kmers.get(kmer, 0)
            if genome_count == 0 or total_sample_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_sample_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            g2a_enrichment_data[kmer].append(enrichment_ratio)
    c2t_enrichment = {kmer: np.nanmean(ratios) for kmer, ratios in c2t_enrichment_data.items() if ratios}
    g2a_enrichment = {kmer: np.nanmean(ratios) for kmer, ratios in g2a_enrichment_data.items() if ratios}
    if metric == 'enrichment':
        c2t_metric = c2t_enrichment
        g2a_metric = g2a_enrichment
    else:
        c2t_metric = c2t_norm
        g2a_metric = g2a_norm
    c2t_kmers_sorted = [k for k, v in sorted(c2t_metric.items(), key=lambda x: x[1], reverse=True)]
    g2a_kmers_sorted = [k for k, v in sorted(g2a_metric.items(), key=lambda x: x[1], reverse=True)]
    return c2t_kmers_sorted, g2a_kmers_sorted

def plot_multipanel_5mer_3mer_signature(context_counts, genome_kmer_counts, output_dir, fontsize=20, metric='frequency'):
    """
    Create a multipanel figure with:
    (A) 3-mer grouped bar (top)
    (B) 5-mer average signature with labels (bottom)
    The 'metric' argument controls whether the 5-mer average signature panel shows 'frequency' or 'enrichment'.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(32, 20))
    gs = GridSpec(2, 1, height_ratios=[1.2, 2.2], hspace=0.3, wspace=0.1)

    # Panel A: 3-mer grouped bar (top, no legend)
    axA = fig.add_subplot(gs[0])
    plot_ems_3mer_signature_grouped_bar(
        context_counts, genome_kmer_counts, output_dir, ax=axA, fontsize=fontsize+4, metric=metric
    )
    # Ensure tick labels are visible by adjusting margins
    axA.margins(x=0.05)
    axA.margins(y=0.05)  # Smaller bottom margin for tick labels
    if axA.get_legend() is not None:
        axA.get_legend().remove()
    axA.set_xlabel(axA.get_xlabel(), fontsize=fontsize+10)
    axA.set_ylabel(axA.get_ylabel(), fontsize=fontsize+10)
    # Don't override tick parameters - let the plotting function handle them
    # axA.tick_params(axis='both', labelsize=fontsize)
    for _, spine in axA.spines.items():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Panel B: 5-mer average signature with labels (bottom)
    axB = fig.add_subplot(gs[1])
    plot_ems_5mer_signature_average_top3_peaks_labeled(
        context_counts, genome_kmer_counts, output_dir, ax=axB, fontsize=fontsize+4, metric=metric
    )
    axB.set_xlabel(axB.get_xlabel(), fontsize=fontsize+10)
    axB.set_ylabel(axB.get_ylabel(), fontsize=fontsize+10)
    axB.tick_params(axis='both', labelsize=fontsize+8)
    for _, spine in axB.spines.items():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Add large panel labels (A, B) using fig.text
    fig.text(0.04, 0.92, 'A', fontsize=fontsize+22, fontweight='bold', va='top', ha='left')
    fig.text(0.04, 0.47, 'B', fontsize=fontsize+22, fontweight='bold', va='top', ha='left')

    outpath = os.path.join(output_dir, f'multipanel_5mer_3mer_signature_{metric}.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_5mer_multiplot_standalone(context_counts, genome_kmer_counts, output_dir, fontsize=20):
    """
    Create a standalone 5-mer multiplot figure with all treated EMS samples as overlaid line plots.
    Four stacked panels:
      1. C>T normalized frequency
      2. G>A normalized frequency
      3. C>T enrichment
      4. G>A enrichment
    Kmers are ordered: C>T alphabetical, G>A as reverse complements of C>T.
    Only one legend at the top (color = mutation type).
    The figure is now compressed in width and extended in height, with minimal x-axis labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    # Get all treated EMS samples
    ems_samples = [s for s in context_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
    if not ems_samples:
        print("No EMS samples found for 5-mer multiplot.")
        return
    # Get all 5-mers
    all_c2t_kmers = set()
    all_g2a_kmers = set()
    for sample in ems_samples:
        for kmer in context_counts[sample]:
            if len(kmer) == 5:
                center = kmer[2]
                if center == 'C':
                    all_c2t_kmers.add(kmer)
                elif center == 'G':
                    all_g2a_kmers.add(kmer)
    all_c2t_kmers = sorted(all_c2t_kmers)
    # Order G>A kmers as reverse complements of C>T kmers
    def reverse_complement(kmer):
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join(complement[base] for base in reversed(kmer))
    c2t_to_g2a = {}
    for c2t_kmer in all_c2t_kmers:
        rc_g2a = reverse_complement(c2t_kmer)
        if rc_g2a in all_g2a_kmers:
            c2t_to_g2a[c2t_kmer] = rc_g2a
    all_g2a_kmers_ordered = []
    for c2t_kmer in all_c2t_kmers:
        if c2t_kmer in c2t_to_g2a:
            all_g2a_kmers_ordered.append(c2t_to_g2a[c2t_kmer])
    remaining_g2a = [kmer for kmer in all_g2a_kmers if kmer not in all_g2a_kmers_ordered]
    all_g2a_kmers_ordered.extend(sorted(remaining_g2a))
    all_g2a_kmers = all_g2a_kmers_ordered
    n_c2t = len(all_c2t_kmers)
    n_g2a = len(all_g2a_kmers)
    x_c2t = np.arange(n_c2t)
    x_g2a = np.arange(n_g2a)
    # Genome 5-mer counts for enrichment
    genome_5mer_counts = {kmer: count for kmer, count in genome_kmer_counts.items() if len(kmer) == 5}
    cg_genome_kmers = {kmer: count for kmer, count in genome_5mer_counts.items() if kmer[2] in ['C', 'G']}
    total_genome_kmers = sum(cg_genome_kmers.values())
    # Set up multiplot: compress width, extend height
    fig, axes = plt.subplots(4, 1, figsize=(20, 18), sharex=False)
    ax1, ax2, ax3, ax4 = axes
    # --- C>T Normalized Frequency ---
    for sample in ems_samples:
        kmer_counts = context_counts[sample]
        c2t_counts = {k: kmer_counts.get(k, 0) for k in all_c2t_kmers}
        total_c2t = sum(c2t_counts.values())
        c2t_norm = [c2t_counts[k] / total_c2t if total_c2t > 0 else 0 for k in all_c2t_kmers]
        ax1.plot(x_c2t, c2t_norm, color='orange', alpha=0.5)
    ax1.set_ylabel('Normalized Frequency', fontsize=fontsize)
    ax1.set_title('C>T 5-mer Mutation Signature (All EMS Samples)', fontsize=fontsize+2, fontweight='bold')
    # --- G>A Normalized Frequency ---
    for sample in ems_samples:
        kmer_counts = context_counts[sample]
        g2a_counts = {k: kmer_counts.get(k, 0) for k in all_g2a_kmers}
        total_g2a = sum(g2a_counts.values())
        g2a_norm = [g2a_counts[k] / total_g2a if total_g2a > 0 else 0 for k in all_g2a_kmers]
        ax2.plot(x_g2a, g2a_norm, color='blue', alpha=0.5)
    ax2.set_ylabel('Normalized Frequency', fontsize=fontsize)
    ax2.set_title('G>A 5-mer Mutation Signature (All EMS Samples)', fontsize=fontsize+2, fontweight='bold')
    # --- C>T Enrichment ---
    for sample in ems_samples:
        kmer_counts = context_counts[sample]
        c2t_enrich = []
        total_c2t = sum(kmer_counts.get(k, 0) for k in all_c2t_kmers)
        for k in all_c2t_kmers:
            observed = kmer_counts.get(k, 0)
            genome_count = cg_genome_kmers.get(k, 0)
            expected_freq = genome_count / total_genome_kmers if total_genome_kmers > 0 else 0
            expected = expected_freq * total_c2t if total_c2t > 0 else 0
            enrich = (observed / expected) if expected > 0 else np.nan
            c2t_enrich.append(enrich)
        ax3.plot(x_c2t, c2t_enrich, color='orange', alpha=0.5)
    ax3.set_ylabel('Enrichment Ratio (Obs/Exp)', fontsize=fontsize)
    ax3.set_title('C>T 5-mer Enrichment (All EMS Samples)', fontsize=fontsize+2, fontweight='bold')
    # --- G>A Enrichment ---
    for sample in ems_samples:
        kmer_counts = context_counts[sample]
        g2a_enrich = []
        total_g2a = sum(kmer_counts.get(k, 0) for k in all_g2a_kmers)
        for k in all_g2a_kmers:
            observed = kmer_counts.get(k, 0)
            genome_count = cg_genome_kmers.get(k, 0)
            expected_freq = genome_count / total_genome_kmers if total_genome_kmers > 0 else 0
            expected = expected_freq * total_g2a if total_g2a > 0 else 0
            enrich = (observed / expected) if expected > 0 else np.nan
            g2a_enrich.append(enrich)
        ax4.plot(x_g2a, g2a_enrich, color='blue', alpha=0.5)
    ax4.set_ylabel('Enrichment Ratio (Obs/Exp)', fontsize=fontsize)
    ax4.set_title('G>A 5-mer Enrichment (All EMS Samples)', fontsize=fontsize+2, fontweight='bold')
    # --- X ticks ---
    # Remove all x-axis tick labels for cleaner appearance
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks([])
    ax4.set_xlabel('5-mer Context', fontsize=fontsize)
    # --- Legend ---
    legend_elements = [Patch(facecolor='orange', label='C>T'), Patch(facecolor='blue', label='G>A')]
    ax1.legend(handles=legend_elements, fontsize=fontsize-4, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)
    ax3.grid(axis='y', alpha=0.3)
    ax4.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    outpath = os.path.join(output_dir, '5mer_multiplot_standalone.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved standalone 5-mer multiplot (4-panel) as: {outpath}")

def extract_trimer_counts_from_5mers(context_counts, genome_kmer_counts):
    """
    Extract trimer counts at -2/-1/0 (left) and 0/+1/+2 (right) positions from 5-mer data.
    Only count trimers where the center position corresponds to a C or G base in the original 5-mer.
    Returns:
        sample_trimer_counts: dict[sample][position]['trimer'] = count
        genome_trimer_counts: dict[position]['trimer'] = count
    """
    sample_trimer_counts = {}
    genome_trimer_counts = {'left': {}, 'right': {}}
    
    # Genome trimer counts from 5-mers
    for kmer5, gcount in genome_kmer_counts.items():
        if len(kmer5) == 5:
            # For left trimers (-2/-1/0): center should be at position 2 of 5-mer (the mutated base)
            if kmer5[2] in ['C', 'G']:
                left = kmer5[0:3]
                genome_trimer_counts['left'][left] = genome_trimer_counts['left'].get(left, 0) + gcount
            
            # For right trimers (0/+1/+2): center should be at position 2 of 5-mer (the mutated base)
            if kmer5[2] in ['C', 'G']:
                right = kmer5[2:5]
                genome_trimer_counts['right'][right] = genome_trimer_counts['right'].get(right, 0) + gcount
    
    # Sample trimer counts from 5-mers
    for sample, sample_counts in context_counts.items():
        sample_trimer_counts[sample] = {'left': {}, 'right': {}}
        for kmer5, scount in sample_counts.items():
            if len(kmer5) == 5:
                # For left trimers (-2/-1/0): center should be at position 2 of 5-mer (the mutated base)
                if kmer5[2] in ['C', 'G']:
                    left = kmer5[0:3]
                    sample_trimer_counts[sample]['left'][left] = sample_trimer_counts[sample]['left'].get(left, 0) + scount
                
                # For right trimers (0/+1/+2): center should be at position 2 of 5-mer (the mutated base)
                if kmer5[2] in ['C', 'G']:
                    right = kmer5[2:5]
                    sample_trimer_counts[sample]['right'][right] = sample_trimer_counts[sample]['right'].get(right, 0) + scount
    
    return sample_trimer_counts, genome_trimer_counts


def debug_trimer_extraction(context_counts, genome_kmer_counts, output_dir):
    """
    Debug function to check trimer extraction and show statistics.
    """
    import pandas as pd
    
    # Test extraction
    sample_trimer_counts, genome_trimer_counts = extract_trimer_counts_from_5mers(context_counts, genome_kmer_counts)
    
    # Print some statistics
    print("=== Trimer Extraction Debug ===")
    
    for position in ['left', 'right']:
        print(f"\n{position.upper()} position ({'-2/-1/0' if position == 'left' else '0/+1/+2'}):")
        
        # Check genome counts
        genome_counts = genome_trimer_counts[position]
        total_genome = sum(genome_counts.values())
        print(f"  Total genome trimers: {total_genome}")
        print(f"  Unique genome trimers: {len(genome_counts)}")
        
        # Check which trimers have C/G at center
        cg_center_count = 0
        for trimer in genome_counts:
            center_pos = 2 if position == 'left' else 0  # Center position in trimer
            if trimer[center_pos] in ['C', 'G']:
                cg_center_count += genome_counts[trimer]
        
        print(f"  Trimers with C/G at center: {cg_center_count}")
        print(f"  Trimers with C/G at center (%): {cg_center_count/total_genome*100:.1f}%")
        
        # Show top 10 trimers by count
        top_trimers = sorted(genome_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top 10 trimers by count:")
        for trimer, count in top_trimers:
            center_pos = 2 if position == 'left' else 0
            center_base = trimer[center_pos]
            print(f"    {trimer} (center: {center_base}): {count}")
    
    return sample_trimer_counts, genome_trimer_counts

def plot_enriched_trimers(sample_trimer_counts, genome_trimer_counts, output_dir, position, min_enrichment=1.5, top_n=20):
    """
    Plot enrichment of trimers at a given position (left or right) across EMS samples.
    Args:
        sample_trimer_counts: dict[sample][position]['trimer'] = count
        genome_trimer_counts: dict[position]['trimer'] = count
        output_dir: output directory
        position: 'left' or 'right'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    # Only EMS samples (excluding 3d/7d)
    ems_samples = [s for s in sample_trimer_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
    if not ems_samples:
        print(f"No EMS samples found for trimer enrichment at {position} position.")
        return
    genome_counts = genome_trimer_counts[position]
    total_genome_positions = sum(genome_counts.values())
    # Aggregate observed trimer counts across EMS samples
    observed_counts = {}
    total_observed_mutations = 0
    for sample in ems_samples:
        for trimer, count in sample_trimer_counts[sample][position].items():
            observed_counts[trimer] = observed_counts.get(trimer, 0) + count
            total_observed_mutations += count
    # Calculate enrichment based on expected vs observed counts
    enrichment = {}
    for trimer in genome_counts:
        obs_count = observed_counts.get(trimer, 0)
        exp_count = genome_counts[trimer]
        
        # Expected mutations = (genome_frequency_of_trimer) × (total_observed_mutations)
        expected_mutations = (exp_count / total_genome_positions) * total_observed_mutations if total_genome_positions > 0 else 0
        
        # Enrichment = observed_mutations / expected_mutations
        enrichment[trimer] = (obs_count / expected_mutations) if expected_mutations > 0 else np.nan
    # Sort by enrichment
    sorted_enriched = sorted(enrichment.items(), key=lambda x: x[1], reverse=True)
    top_enriched = [x for x in sorted_enriched if not np.isnan(x[1])][:top_n]
    # Plot
    plt.figure(figsize=(1.2*top_n, 6))
    trimers = [x[0] for x in top_enriched]
    ratios = [x[1] for x in top_enriched]
    bars = plt.bar(trimers, ratios, color='darkorange', alpha=0.8)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7)
    plt.ylabel('Enrichment Ratio (Obs/Exp)')
    plt.xlabel(f'Trimer ({position} of mutated base)')
    plt.title(f'Top {top_n} Enriched Trimers at {position} (-2/-1/0 or 0/+1/+2)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    outpath = os.path.join(output_dir, f'enriched_trimers_{position}.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trimer enrichment plot for {position} to {outpath}")
    # Also save enrichment values as CSV
    df = pd.DataFrame({'trimer': trimers, 'enrichment': ratios})
    df.to_csv(os.path.join(output_dir, f'enriched_trimers_{position}.csv'), index=False)

def plot_trimer_enrichment_multipanel(sample_trimer_counts, genome_trimer_counts, output_dir, top_n=20):
    """
    Create a multipanel figure with two heatmaps: left (-2/-1/0) and right (0/+1/+2) trinucleotide enrichment across EMS samples.
    Sample names are simplified, and titles/labels use 'trinucleotide'.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    positions = ['left', 'right']
    pos_labels = {'left': '-2/-1/0', 'right': '0/+1/+2'}
    dfs = {}
    for position in positions:
        ems_samples = [s for s in sample_trimer_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
        genome_counts = genome_trimer_counts[position]
        total_genome_positions = sum(genome_counts.values())
        enrichment_matrix = {}
        for sample in ems_samples:
            observed = sample_trimer_counts[sample][position]
            total_observed_mutations = sum(observed.values())
            enrichment = {}
            for trimer in genome_counts:
                obs_count = observed.get(trimer, 0)
                exp_count = genome_counts[trimer]
                
                # Expected mutations = (genome_frequency_of_trimer) × (total_observed_mutations)
                expected_mutations = (exp_count / total_genome_positions) * total_observed_mutations if total_genome_positions > 0 else 0
                
                # Enrichment = observed_mutations / expected_mutations
                enrichment[trimer] = (obs_count / expected_mutations) if expected_mutations > 0 else np.nan
            enrichment_matrix[sample] = enrichment
        df = pd.DataFrame(enrichment_matrix).T
        # Simplify sample names
        df.index = [clean_ems_sample_name(s) for s in df.index]
        # Select top N trinucleotides by median enrichment
        top_trimers = df.median(axis=0).sort_values(ascending=False).head(top_n).index
        dfs[position] = df[top_trimers]
    # Multipanel plot
    fig = plt.figure(figsize=(1.2*top_n, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 1])
    for i, position in enumerate(positions):
        ax = fig.add_subplot(gs[i])
        sns.heatmap(dfs[position], cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Enrichment Ratio (Obs/Exp)'}, ax=ax)
        ax.set_title(f'Trinucleotide Enrichment Heatmap ({pos_labels[position]})', fontsize=18)
        ax.set_xlabel('Trinucleotide', fontsize=14)
        ax.set_ylabel('Sample', fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'trinucleotide_enrichment_multipanel.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multipanel trinucleotide enrichment heatmap to {outpath}")

def output_trimer_enrichment_table(sample_trimer_counts, genome_trimer_counts, output_dir):
    """
    Output comprehensive tables of all trimers and their enrichment/depletion values for both positions.
    Creates separate CSV files for left (-2/-1/0) and right (0/+1/+2) positions with all trimers.
    """
    import pandas as pd
    import numpy as np
    
    positions = ['left', 'right']
    pos_labels = {'left': '-2/-1/0', 'right': '0/+1/+2'}
    
    for position in positions:
        # Only EMS samples (excluding 3d/7d)
        ems_samples = [s for s in sample_trimer_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
        if not ems_samples:
            print(f"No EMS samples found for trimer enrichment at {position} position.")
            continue
            
        genome_counts = genome_trimer_counts[position]
        total_genome_positions = sum(genome_counts.values())
        
        # Aggregate observed trimer counts across EMS samples
        observed_counts = {}
        total_observed_mutations = 0
        for sample in ems_samples:
            for trimer, count in sample_trimer_counts[sample][position].items():
                observed_counts[trimer] = observed_counts.get(trimer, 0) + count
                total_observed_mutations += count
        
        # Calculate enrichment for ALL trimers
        enrichment_data = []
        for trimer in sorted(genome_counts.keys()):
            obs_count = observed_counts.get(trimer, 0)
            exp_count = genome_counts[trimer]
            
            # Expected mutations = (genome_frequency_of_trimer) × (total_observed_mutations)
            expected_mutations = (exp_count / total_genome_positions) * total_observed_mutations if total_genome_positions > 0 else 0
            
            # Enrichment = observed_mutations / expected_mutations
            enrichment = (obs_count / expected_mutations) if expected_mutations > 0 else np.nan
            
            # Calculate additional statistics
            genome_freq = exp_count / total_genome_positions if total_genome_positions > 0 else 0
            observed_freq = obs_count / total_observed_mutations if total_observed_mutations > 0 else 0
            
            enrichment_data.append({
                'trimer': trimer,
                'observed_count': obs_count,
                'expected_count': expected_mutations,
                'genome_count': genome_counts[trimer],
                'genome_frequency': genome_freq,
                'observed_frequency': observed_freq,
                'enrichment_ratio': enrichment,
                'log2_enrichment': np.log2(enrichment) if enrichment > 0 and not np.isnan(enrichment) else np.nan
            })
        
        # Create DataFrame and sort by enrichment
        df = pd.DataFrame(enrichment_data)
        df = df.sort_values('enrichment_ratio', ascending=False)
        
        # Add enrichment/depletion classification
        df['enrichment_status'] = df['enrichment_ratio'].apply(
            lambda x: 'enriched' if x > 1.5 else ('depleted' if x < 0.67 else 'neutral')
        )
        
        # Save to CSV
        outpath = os.path.join(output_dir, f'trimer_enrichment_table_{position}_{pos_labels[position].replace("/", "_")}.csv')
        df.to_csv(outpath, index=False)
        print(f"Saved comprehensive trimer enrichment table for {position} ({pos_labels[position]}) to {outpath}")
        
        # Print summary statistics
        enriched_count = len(df[df['enrichment_status'] == 'enriched'])
        depleted_count = len(df[df['enrichment_status'] == 'depleted'])
        neutral_count = len(df[df['enrichment_status'] == 'neutral'])
        
        print(f"  Summary for {position} ({pos_labels[position]}):")
        print(f"    Enriched trimers (>1.5x): {enriched_count}")
        print(f"    Depleted trimers (<0.67x): {depleted_count}")
        print(f"    Neutral trimers: {neutral_count}")
        print(f"    Total trimers: {len(df)}")

def main():
    parser = argparse.ArgumentParser(description='Generate plots from analysis results')
    parser.add_argument('-r', '--results_dir', type=str, required=True,
                       help='Path to results directory containing analysis outputs')
    parser.add_argument('-n', '--nonconsensus_results_dir', type=str,
                       help='Path to nonconsensus results directory for comparison plots')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='Path to output directory for plots')
    parser.add_argument('--dnds-csv', type=str, 
                       help='Path to dN/dS analysis CSV file (for random mutation analysis)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for and process different result files
    results_dir = Path(args.results_dir)
    
    # Mutation frequency plots
    nuc_mut_dir = results_dir / 'nuc_muts'
    jsons = list(nuc_mut_dir.glob('*.json'))
    if jsons:  # Only proceed if JSON files exist
        basecounts_file = results_dir / 'results' / 'basecounts.json'
        if basecounts_file.exists():
            print("Generating mutation frequency plots...")
            with open(basecounts_file) as f:
                base_counts = json.load(f)  # Now this is genome-wide base counts
            # Remove basecounts.json from jsons list
            jsons = [j for j in jsons if j.stem != 'basecounts']
            if jsons:  # Check if there are mutation JSONs after filtering
                if args.nonconsensus_results_dir:
                    # Generate consensus comparison plots
                    nonconsensus_dir = Path(args.nonconsensus_results_dir)
                    nonconsensus_nuc_mut_dir = nonconsensus_dir / 'nuc_muts'
                    nonconsensus_jsons = list(nonconsensus_nuc_mut_dir.glob('*.json'))
                    if nonconsensus_jsons:
                        # Remove basecounts.json from nonconsensus jsons list
                        nonconsensus_jsons = [j for j in nonconsensus_jsons if j.stem != 'basecounts']
                        if nonconsensus_jsons:
                            print("Generating consensus vs nonconsensus comparison plots...")
                            mutation_type_barplot_consensus_comparison(jsons, nonconsensus_jsons, base_counts, args.output_dir)
                        else:
                            print("No nonconsensus mutation JSONs found, generating single dataset plots...")
                            mutation_type_barplot(jsons, base_counts, args.output_dir)
                    else:
                        print("No nonconsensus mutation JSONs found, generating single dataset plots...")
                        mutation_type_barplot(jsons, base_counts, args.output_dir)
                else:
                    # Generate single dataset plots
                    mutation_type_barplot(jsons, base_counts, args.output_dir)
    
    # Kmer context plots
    contextcounts_file = results_dir / 'results' / 'contextcounts.json'
    genome_kmer_counts_file = results_dir / 'results' / 'genome_kmer_counts.json'
    if contextcounts_file.exists() and genome_kmer_counts_file.exists():
        print("Generating kmer context plots (5-mer and 3-mer only)...")
        with open(contextcounts_file) as f:
            context_counts = json.load(f)
        with open(genome_kmer_counts_file) as f:
            genome_kmer_counts = json.load(f)
        
        # Generate 5-mer and 3-mer plots only (no 7-mer)
        plot_consistently_enriched_5mer_kmers_heatmap(context_counts, genome_kmer_counts, args.output_dir)
        plot_consistently_depleted_5mer_kmers_heatmap(context_counts, genome_kmer_counts, args.output_dir)
        plot_consistently_enriched_3mer_kmers_heatmap(context_counts, genome_kmer_counts, args.output_dir)
        plot_consistently_depleted_3mer_kmers_heatmap(context_counts, genome_kmer_counts, args.output_dir)
        #plot_ems_5mer_signature_per_sample(context_counts, args.output_dir)
        #plot_ems_5mer_signature_all_samples(context_counts, args.output_dir)
        #plot_ems_3mer_signature_per_sample(context_counts, genome_kmer_counts, args.output_dir)
        #plot_ems_3mer_signature_all_samples(context_counts, genome_kmer_counts, args.output_dir)
        #plot_ems_3mer_signature_multiplot(context_counts, genome_kmer_counts, args.output_dir)
        #plot_ems_5mer_signature_multiplot(context_counts, args.output_dir)
        #plot_ems_3mer_signature_grouped_bar(context_counts, genome_kmer_counts, args.output_dir)
        #plot_ems_5mer_signature_multiplot_9samples(context_counts, args.output_dir)
        #plot_ems_5mer_signature_average_with_labels(context_counts, genome_kmer_counts, args.output_dir)
        plot_ems_5mer_signature_average_colored_by_enrichment(context_counts, genome_kmer_counts, args.output_dir)
        plot_ems_5mer_signature_average_top3_peaks_labeled(context_counts, genome_kmer_counts, args.output_dir)
        plot_5mer_multiplot_standalone(context_counts, genome_kmer_counts, args.output_dir)
        plot_multipanel_5mer_3mer_signature(context_counts, genome_kmer_counts, args.output_dir, metric='frequency')
        plot_multipanel_5mer_3mer_signature(context_counts, genome_kmer_counts, args.output_dir, metric='enrichment')
        # --- NEW: Trimer enrichment at -2/-1/0 and 0/+1/+2 ---
        sample_trimer_counts, genome_trimer_counts = extract_trimer_counts_from_5mers(context_counts, genome_kmer_counts)
        # Debug trimer extraction
        debug_trimer_extraction(context_counts, genome_kmer_counts, args.output_dir)
        plot_enriched_trimers(sample_trimer_counts, genome_trimer_counts, args.output_dir, position='left', min_enrichment=1.5, top_n=20)
        plot_enriched_trimers(sample_trimer_counts, genome_trimer_counts, args.output_dir, position='right', min_enrichment=1.5, top_n=20)
        plot_trimer_enrichment_multipanel(sample_trimer_counts, genome_trimer_counts, args.output_dir, top_n=20)
        output_trimer_enrichment_table(sample_trimer_counts, genome_trimer_counts, args.output_dir)
    else:
        print("Warning: contextcounts.json or genome_kmer_counts.json not found in results directory")
    
    # dN/dS ratio plots - only use plot_dnds_analysis
    if args.dnds_csv and os.path.exists(args.dnds_csv):
        print("Generating dN/dS analysis plots...")
        plot_dnds_analysis(
            results_csv=args.dnds_csv,
            output_dir=args.output_dir,
            title="dN/dS Analysis of EMS Mutations"
        )

    # Add per-gene dN/dS plots
    if args.results_dir:
        print("Generating per-gene dN/dS plots...")
        plot_per_gene_dnds(args.results_dir, args.output_dir)

    # Add gene mutation statistics
    gene_stats_path = os.path.join(args.results_dir, 'results', 'gene_statistics.json')
    top_n = 30
    print(gene_stats_path)
    if os.path.exists(gene_stats_path):
        print("Generating gene mutation statistics plots...")
        plot_gene_mutation_statistics(gene_stats_path, args.output_dir, top_n)

if __name__ == "__main__":
    main()