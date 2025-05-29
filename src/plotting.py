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

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def count_mutations(json_files: list) -> dict:
    '''Count mutation frequencies across all samples.
    
    Args:
        json_files (list): List of paths to mutation JSON files
    
    Returns:
        dict: Dictionary of mutation frequencies per sample
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
                
                # store the mutation frequency in the total frequencies dictionary
                if mutation_type not in total_freqs[sample]:
                    total_freqs[sample][mutation_type] = mutation_counts[gene]['mutations'][site]
                else:
                    total_freqs[sample][mutation_type] += mutation_counts[gene]['mutations'][site] 
               
    return total_freqs

def plot_kmer_context(context_counts: dict, genome_kmer_counts: dict, output_dir: str) -> None:
    '''Plot the kmer context barplot.'''
    output_path = os.path.join(output_dir, 'kmer_context.png')
    observed_kmers = {}
    ems_samples = [sample for sample in context_counts if 'EMS' in sample]

    for sample in ems_samples:
        sample_data = context_counts[sample]
        for kmer, count in sample_data.items():
            if kmer in genome_kmer_counts and genome_kmer_counts[kmer] > 0:
                freq = count / genome_kmer_counts[kmer]
                if kmer not in observed_kmers:
                    observed_kmers[kmer] = []
                observed_kmers[kmer].append(freq)

    # Average across samples
    averaged_kmers = {kmer: np.mean(freqs) for kmer, freqs in observed_kmers.items() if freqs}

    # Sort and get top 30
    top_kmers = dict(sorted(averaged_kmers.items(), key=lambda x: x[1], reverse=True)[:30])

    plt.figure(figsize=(12, 6))
    plt.bar(top_kmers.keys(), top_kmers.values(), color='darkorange')
    plt.xlabel('Kmer Context')
    plt.ylabel('Normalized Frequency (Observed/Genome)')
    plt.title('Top 30 Most Frequent Kmer Contexts (EMS Samples)')
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

def normalize_mutation_counts(total_freqs: dict, base_counts: dict, output_dir: str) -> str:
    '''Normalize mutation counts by base counts and save to CSV.
    
    Args:
        total_freqs (dict): Dictionary of mutation frequencies per sample
        base_counts (dict): Dictionary of genome-wide base counts {'A': count, 'T': count, ...}
        output_dir (str): Directory to save output CSV
        
    Returns:
        str: Path to the saved CSV file
    '''
    
    output_path = os.path.join(output_dir, 'mutation_frequencies.csv')
    
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
    mutation_frequency_csv = normalize_mutation_counts(total_freqs, base_counts, output_dir)
    
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
    '''Plot normalized kmer context for each sample.'''
    output_path = os.path.join(output_dir, 'normalized_kmer_context_per_sample.png')
    sample_data = {}
    all_kmers = set()

    for sample, data in context_counts.items():
        sample_data[sample] = {}
        for kmer, count in data.items():
            if kmer in genome_kmer_counts and genome_kmer_counts[kmer] > 0:
                norm_freq = count / genome_kmer_counts[kmer]
                sample_data[sample][kmer] = norm_freq
                all_kmers.add(kmer)

    if not sample_data:
        print("No normalized kmer data found")
        return

    # Get top 20 most variable kmers across all samples
    kmer_variances = {}
    for kmer in all_kmers:
        values = [sample_data[sample].get(kmer, 0) for sample in sample_data]
        if len(values) > 1:
            kmer_variances[kmer] = np.var(values)

    top_kmers = sorted(kmer_variances.items(), key=lambda x: x[1], reverse=True)[:20]
    top_kmer_names = [kmer for kmer, _ in top_kmers]

    ems_samples = [s for s in sample_data.keys() if 'EMS' in s]
    control_samples = [s for s in sample_data.keys() if 'EMS' not in s]

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
    ax1.set_title('Normalized Kmer Context Frequencies (Top 20 Most Variable)')
    ax1.set_xlabel('Kmer Context')
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
        ax2.set_xlabel('Kmer Context')
        ax2.set_ylabel('Mean Normalized Frequency')
        ax2.set_title('Average Normalized Kmer Frequencies: Control vs EMS')
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_kmer_names, rotation=45, ha='right')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_ems_mutation_kmer_bias(context_counts: dict, genome_kmer_counts: dict, output_dir: str) -> None:
    '''Analyze kmer bias for each EMS sample individually.'''
    if not genome_kmer_counts:
        print("No genome kmer counts found for bias analysis")
        return

    for sample, sample_kmers in context_counts.items():
        if 'EMS' not in sample:
            continue

        output_path = os.path.join(output_dir, f'ems_mutation_kmer_bias_{sample}.png')
        ems_mutation_kmers = sample_kmers  # This is already a dict of {kmer: count}

        # Calculate expected vs observed frequencies
        total_mutations = sum(ems_mutation_kmers.values())
        total_genome_kmers = sum(genome_kmer_counts.values())

        bias_analysis = []
        for kmer, observed_count in ems_mutation_kmers.items():
            genome_count = genome_kmer_counts.get(kmer, 0)
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

        # Sort by enrichment
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
            plt.title(f'Top 20 Enriched Kmers in {sample}')
            plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        # Print summary
        print(f"\nEMS Mutation Kmer Bias Analysis for {sample}:")
        print(f"Total mutations analyzed: {total_mutations:,}")
        print(f"Significantly enriched kmers: {len(significant_kmers)}")
        if significant_kmers:
            print(f"\nTop 5 significantly enriched kmers:")
            for i, kmer_data in enumerate(significant_kmers[:5]):
                print(f"  {i+1}. {kmer_data['kmer']}: {kmer_data['enrichment_ratio']:.2f}x enriched (p={kmer_data['p_value']:.2e})")
        print(f"Detailed results saved to {results_path}")

def analyze_ems_cg_kmer_bias(context_counts: dict, genome_kmer_counts: dict, output_dir: str) -> None:
    '''Analyze EMS mutation bias specifically for kmers with C or G as center base.
    
    Args:
        context_counts (dict): Dictionary containing context counts and normalized analysis
        genome_kmer_counts (dict): Genome-wide kmer counts
        output_dir (str): Directory to save output plot and analysis
    '''
    output_path = os.path.join(output_dir, 'ems_cg_kmer_bias.png')
    
    # Extract EMS mutation data and use provided genome_kmer_counts
    ems_mutation_kmers = {}
    cg_genome_kmers = genome_kmer_counts
    if not cg_genome_kmers:
        print("No genome kmer counts found for C/G kmer bias analysis")
        return
    
    # Collect mutation kmers from EMS samples (only C/G center)
    for sample, data in context_counts.items():
        if 'EMS' in sample and sample != 'genome_kmer_counts':
            # Use raw mutation counts (not normalized)
            sample_kmers = data if 'normalized_analysis' not in data else data
            for kmer, count in sample_kmers.items():
                if isinstance(count, int) and len(kmer) == 5 and kmer[2] in ['C', 'G']:  # Changed from k[1] to k[2]
                    ems_mutation_kmers[kmer] = ems_mutation_kmers.get(kmer, 0) + count
    
    if not ems_mutation_kmers:
        print("No EMS mutation C/G kmer data found")
        return
    
    # Calculate expected vs observed frequencies for C/G kmers only
    total_mutations = sum(ems_mutation_kmers.values())
    total_cg_genome_kmers = sum(cg_genome_kmers.values())
    
    bias_analysis = []
    
    for kmer in cg_genome_kmers:
        observed_count = ems_mutation_kmers.get(kmer, 0)
        expected_freq = cg_genome_kmers[kmer] / total_cg_genome_kmers
        expected_count = expected_freq * total_mutations
        
        # Calculate enrichment ratio
        if expected_count > 0:
            enrichment_ratio = observed_count / expected_count
        else:
            enrichment_ratio = 0
        
        # Perform binomial test for significance
        from scipy.stats import binomtest
        if total_mutations > 0 and expected_freq > 0:
            result = binomtest(observed_count, total_mutations, expected_freq)
            p_value = result.pvalue
        else:
            p_value = 1.0
        
        bias_analysis.append({
            'kmer': kmer,
            'center_base': kmer[2],  # Changed from kmer[1] to kmer[2]
            'observed': observed_count,
            'expected': expected_count,
            'genome_count': cg_genome_kmers[kmer],
            'enrichment_ratio': enrichment_ratio,
            'p_value': p_value,
            'significant': bool(p_value < 0.05 and enrichment_ratio > 1.5)
        })
    
    # Sort by enrichment ratio
    bias_analysis.sort(key=lambda x: x['enrichment_ratio'], reverse=True)
    
    # Separate C and G center kmers
    c_kmers = [x for x in bias_analysis if x['center_base'] == 'C']
    g_kmers = [x for x in bias_analysis if x['center_base'] == 'G']
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top enriched C-center kmers
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
        ax1.set_title('Top Enriched C-Center 5-mers in EMS Mutations')
        ax1.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    
    # 2. Top enriched G-center kmers
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
        ax2.set_title('Top Enriched G-Center 5-mers in EMS Mutations')
        ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    
    # 3. Observed vs Expected counts for top C-center kmers
    if top_c_enriched:
        kmers = [x['kmer'] for x in top_c_enriched[:10]]
        observed = [x['observed'] for x in top_c_enriched[:10]]
        expected = [x['expected'] for x in top_c_enriched[:10]]
        
        x = np.arange(len(kmers))
        width = 0.35
        
        ax3.bar(x - width/2, observed, width, label='Observed', color='red', alpha=0.7)
        ax3.bar(x + width/2, expected, width, label='Expected', color='blue', alpha=0.7)
        ax3.set_xlabel('C-Center 5-mer')
        ax3.set_ylabel('Mutation Count')
        ax3.set_title('Observed vs Expected Counts (Top C-Center 5-mers)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(kmers, rotation=45, ha='right')
        ax3.legend()
    
    # 4. Observed vs Expected counts for top G-center kmers
    if top_g_enriched:
        kmers = [x['kmer'] for x in top_g_enriched[:10]]
        observed = [x['observed'] for x in top_g_enriched[:10]]
        expected = [x['expected'] for x in top_g_enriched[:10]]
        
        x = np.arange(len(kmers))
        width = 0.35
        
        ax4.bar(x - width/2, observed, width, label='Observed', color='red', alpha=0.7)
        ax4.bar(x + width/2, expected, width, label='Expected', color='blue', alpha=0.7)
        ax4.set_xlabel('G-Center 5-mer')
        ax4.set_ylabel('Mutation Count')
        ax4.set_title('Observed vs Expected Counts (Top G-Center 5-mers)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(kmers, rotation=45, ha='right')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate summary statistics
    c_significant = [x for x in c_kmers if x['significant']]
    g_significant = [x for x in g_kmers if x['significant']]
    
    # Save detailed results
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
    
    # Print summary
    print(f"\nEMS C/G 5-mer Bias Analysis:")
    print(f"Total C/G mutations analyzed: {total_mutations:,}")
    print(f"C-center 5-mers analyzed: {len(c_kmers)}")
    print(f"G-center 5-mers analyzed: {len(g_kmers)}")
    print(f"Significantly enriched C-center 5-mers: {len(c_significant)}")
    print(f"Significantly enriched G-center 5-mers: {len(g_significant)}")
    
    c_ratios = [x['enrichment_ratio'] for x in c_kmers]
    g_ratios = [x['enrichment_ratio'] for x in g_kmers]
    print(f"Median C-center enrichment: {np.median(c_ratios):.2f}" if c_ratios else "No C-center data")
    print(f"Median G-center enrichment: {np.median(g_ratios):.2f}" if g_ratios else "No G-center data")
    
    if c_significant:
        print(f"\nTop 3 significantly enriched C-center 5-mers:")
        for i, kmer_data in enumerate(c_significant[:3]):
            print(f"  {i+1}. {kmer_data['kmer']}: {kmer_data['enrichment_ratio']:.2f}x enriched (p={kmer_data['p_value']:.2e})")
    
    if g_significant:
        print(f"\nTop 3 significantly enriched G-center 5-mers:")
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

def plot_normalized_kmer_enrichment_heatmap(context_counts: dict, genome_kmer_counts: dict, output_dir: str, top_n: int = 30):
    """
    Plot a heatmap of normalized kmer enrichment (Observed/Expected) for each EMS sample.
    """
    import seaborn as sns
    import pandas as pd

    # Collect all EMS samples
    ems_samples = [sample for sample in context_counts if 'EMS' in sample]
    if not ems_samples:
        print("No EMS samples found for enrichment heatmap.")
        return

    # Build a DataFrame: rows=samples, columns=kmers, values=enrichment ratio
    enrichment_data = {}
    all_kmers = set()
    for sample in ems_samples:
        sample_kmers = context_counts[sample]
        total_mutations = sum(sample_kmers.values())
        total_genome_kmers = sum(genome_kmer_counts.values())
        enrichment_data[sample] = {}
        for kmer, observed_count in sample_kmers.items():
            genome_count = genome_kmer_counts.get(kmer, 0)
            if genome_count == 0 or total_mutations == 0:
                continue
            expected_freq = genome_count / total_genome_kmers
            expected_count = expected_freq * total_mutations
            enrichment_ratio = (observed_count / expected_count) if expected_count > 0 else np.nan
            enrichment_data[sample][kmer] = enrichment_ratio
            all_kmers.add(kmer)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(enrichment_data, orient='index').fillna(0)

    # Select top N kmers by variance across samples (most variable/enriched)
    kmer_variances = df.var(axis=0)
    top_kmers = kmer_variances.sort_values(ascending=False).head(top_n).index
    df_top = df[top_kmers]

    # Z-score normalization (optional, for better heatmap contrast)
    df_top_z = (df_top - df_top.mean()) / df_top.std()

    plt.figure(figsize=(1.2*top_n, 0.5*len(df_top)+4))
    sns.heatmap(df_top_z, cmap='RdBu_r', center=0, annot=False, cbar_kws={'label': 'Z-score Enrichment'})
    plt.title(f'Normalized Kmer Enrichment (Observed/Expected, Z-score)\nTop {top_n} Most Variable Kmers (EMS Samples)')
    plt.xlabel('Kmer')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'normalized_kmer_enrichment_heatmap_top{top_n}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate plots from analysis results')
    parser.add_argument('-r', '--results_dir', type=str, required=True,
                       help='Path to results directory containing analysis outputs')
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
                mutation_type_barplot(jsons, base_counts, args.output_dir)
    
    # Kmer context plots
    contextcounts_file = results_dir / 'results' / 'contextcounts.json'
    genome_kmer_counts_file = results_dir / 'results' / 'genome_kmer_counts.json'
    if contextcounts_file.exists() and genome_kmer_counts_file.exists():
        print("Generating kmer context plots...")
        with open(contextcounts_file) as f:
            context_counts = json.load(f)
        with open(genome_kmer_counts_file) as f:
            genome_kmer_counts = json.load(f)
        
        # Generate kmer plots
        plot_kmer_context(context_counts, genome_kmer_counts, args.output_dir)
        plot_normalized_kmer_context_per_sample(context_counts, genome_kmer_counts, args.output_dir)
        analyze_ems_mutation_kmer_bias(context_counts, genome_kmer_counts, args.output_dir)
        analyze_ems_cg_kmer_bias(context_counts, genome_kmer_counts, args.output_dir)
        plot_normalized_kmer_enrichment_heatmap(context_counts, genome_kmer_counts, args.output_dir)
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

if __name__ == "__main__":
    main()