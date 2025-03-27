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

def plot_kmer_context(data_dict: dict, genome_kmer_counts: dict, output_dir: str) -> None:
    '''Plot the kmer context barplot.
    
    Args:
        data_dict (dict): Dictionary of kmer context data
        output_dir (str): Directory to save output plot
    '''
    
    # Set output path for the plot 
    output_path = os.path.join(output_dir, 'kmer_context.png')

    # initialize a dictionary to store the observed kmers
    observed_kmers = {}
    ems_samples = []
    
    # normalize and average the observed kmer rate across all treated samples
    for sample in data_dict:
        if 'EMS' in sample:
            ems_samples.append(sample)
            for kmer in data_dict[sample]:
                if kmer not in observed_kmers:
                    observed_kmers[kmer] = data_dict[sample][kmer] / genome_kmer_counts[kmer]
                else:
                    observed_kmers[kmer] += data_dict[sample][kmer] / genome_kmer_counts[kmer] 
                     
    for kmer in observed_kmers:
        observed_kmers[kmer] = observed_kmers[kmer] / len(ems_samples)
            
    # plot the kmer context barplot 
    # Sort kmers by count and get top 10
    top_kmers = dict(sorted(observed_kmers.items(), key=lambda x: x[1], reverse=True)[:30])
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    plt.bar(top_kmers.keys(), top_kmers.values(), color='darkorange')
    
    # Customize plot
    plt.xlabel('Kmer Context')
    plt.ylabel('Frequency')
    plt.title('Top 30 Most Frequent Kmer Contexts')
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save
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

def normalize_mutation_counts(total_freqs: dict, base_counts: dict, output_file_dir: str) -> str:
    '''Normalize mutation counts by total reference base counts and write to CSV.
    
    Args:
        total_freqs (dict): Dictionary of mutation frequencies per sample
        base_counts (dict): Dictionary of total base counts per sample from basecounts.json
        output_file_dir (str): Directory to write output CSV
        
    Returns:
        str: Path to output CSV file
    '''
    output_file = os.path.join(output_file_dir, 'normalized_ems_mutations_freqs.csv')
    header = ['sample','ems','norm_count','mutation']
    
    with open(output_file, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)
        for sample in total_freqs:
            if 'NT' in sample or 'Minus' in sample or 'Pre' in sample:
                ems = '-'
            else:
                ems = '+'
            for mut_type in total_freqs[sample]:
                ref = mut_type[0]  # Get reference base from mutation type
                # Normalize by total count of reference base from basecounts
                norm_freq = total_freqs[sample][mut_type] / base_counts[sample][ref]
                writer.writerow([sample, ems, norm_freq, mut_type])
    
    return output_file

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

def mutation_type_barplot(json_file_dir: list, base_counts: dict, output_file_dir: str):
    '''Generate mutation type frequency barplot using basecounts for normalization.
    
    Args:
        json_file_dir (list): List of paths to mutation JSON files
        base_counts (dict): Dictionary of total base counts per sample
        output_file_dir (str): Directory to write output files
    '''
    # Count mutation frequencies
    total_frequencies = count_mutations(json_file_dir)
    
    # Normalize using provided base counts
    mutation_frequence_csv = normalize_mutation_counts(total_frequencies, base_counts, output_file_dir)
    
    # Generate plot
    plot_ems_mutation_frequencies(mutation_frequence_csv, output_file_dir)
    plot_ems_mutation_frequencies_per_sample(mutation_frequence_csv, output_file_dir)   

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
                base_counts = json.load(f)
            # Remove basecounts.json from jsons list
            jsons = [j for j in jsons if j.stem != 'basecounts']
            if jsons:  # Check if there are mutation JSONs after filtering
                mutation_type_barplot(jsons, base_counts, args.output_dir)
    
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