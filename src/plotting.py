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
    '''Create combined dN/dS ratio plot for all samples.
    
    Args:
        aa_mutations_files (list): List of paths to amino acid mutation JSON files
        output_dir (str): Directory to save output plots
    '''
    # Data structures to hold all samples
    all_data = {
        'gene_lengths': [],
        'coverages': [],
        'syn_ratios': [],
        'is_control': []  # True for control samples, False for treated
    }
    
    # Process each sample
    for aa_file in aa_mutations_files:
        with open(aa_file) as f:
            aa_mutations = json.load(f)
            
        # Determine if control sample
        is_control = any(x in aa_file.stem for x in ['NT', 'Minus', 'Pre'])
        
        # Extract data
        for gene in aa_mutations:
            if 'dnds_norm' in aa_mutations[gene] and 'gene_len' in aa_mutations[gene]:
                all_data['gene_lengths'].append(aa_mutations[gene]['gene_len'])
                all_data['coverages'].append(aa_mutations[gene].get('avg_cov', 0))
                all_data['syn_ratios'].append(aa_mutations[gene]['dnds_norm'])
                all_data['is_control'].append(is_control)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot dN/dS vs gene length
    for is_control, color, label in [(True, 'lightgrey', 'Control'), (False, 'darkorange', 'EMS Treated')]:
        mask = [x == is_control for x in all_data['is_control']]
        ax1.scatter(
            [l for l, m in zip(all_data['gene_lengths'], mask) if m],
            [r for r, m in zip(all_data['syn_ratios'], mask) if m],
            alpha=0.5,
            color=color,
            label=label
        )
    
    ax1.set_xlabel('Gene Length (bp)')
    ax1.set_ylabel('dN/dS Ratio')
    ax1.set_title('dN/dS Ratio vs Gene Length - All Samples')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax1.legend()
    
    # Plot dN/dS vs coverage
    for is_control, color, label in [(True, 'lightgrey', 'Control'), (False, 'darkorange', 'EMS Treated')]:
        mask = [x == is_control for x in all_data['is_control']]
        ax2.scatter(
            [c for c, m in zip(all_data['coverages'], mask) if m],
            [r for r, m in zip(all_data['syn_ratios'], mask) if m],
            alpha=0.5,
            color=color,
            label=label
        )
    
    ax2.set_xlabel('Average Coverage')
    ax2.set_ylabel('dN/dS Ratio')
    ax2.set_title('dN/dS Ratio vs Coverage - All Samples')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax2.legend()
    
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
    '''Create plot comparing genome-wide dN/dS ratios vs mutation rates.
    Also outputs a CSV comparing key metrics per sample.
    
    Args:
        results_dir (str): Directory containing analysis results
        output_dir (str): Directory to save output plots and CSV
    '''
    # Load the data
    with open(f"{results_dir}/results/genome_dnds.json") as f:
        dnds_data = json.load(f)
    with open(f"{results_dir}/results/intergeniccounts.json") as f:
        intergenic_data = json.load(f)
        
    # Prepare data for plotting and CSV
    plot_data = {
        'sample': [],
        'dnds': [],
        'intergenic_rate': [],
        'syn_sites_mutated_rate': [],
        'non_syn_sites_mutated_rate': [],  # Added non-syn rate
        'is_control': []
    }
    
    # Prepare CSV data
    csv_data = []
    
    for sample in dnds_data:
        if sample in intergenic_data:
            # Calculate rates
            total_mutations = sum(intergenic_data[sample]['mutations'].values())
            total_sites = intergenic_data[sample]['total_sites']
            intergenic_rate = total_mutations / total_sites if total_sites > 0 else 0
            
            # Calculate mutation rates
            syn_sites_mutated_rate = (dnds_data[sample]['syn_sites_mutated'] / 
                                    dnds_data[sample]['syn_sites']) if dnds_data[sample]['syn_sites'] > 0 else 0
            non_syn_sites_mutated_rate = (dnds_data[sample]['non_syn_sites_mutated'] / 
                                        dnds_data[sample]['non_syn_sites']) if dnds_data[sample]['non_syn_sites'] > 0 else 0
            
            # Store for plotting
            plot_data['sample'].append(sample)
            plot_data['dnds'].append(dnds_data[sample]['dnds'])
            plot_data['intergenic_rate'].append(intergenic_rate)
            plot_data['syn_sites_mutated_rate'].append(syn_sites_mutated_rate)
            plot_data['non_syn_sites_mutated_rate'].append(non_syn_sites_mutated_rate)
            plot_data['is_control'].append(any(x in sample for x in ['NT', 'Minus', 'Pre']))
            
            # Store for CSV
            csv_data.append({
                'Sample': sample,
                'dN/dS': dnds_data[sample]['dnds'],
                'Intergenic_Rate': intergenic_rate,
                'Syn_Sites_Mutated_Rate': syn_sites_mutated_rate,
                'Non_Syn_Sites_Mutated_Rate': non_syn_sites_mutated_rate,
                'Total_Intergenic_Mutations': total_mutations,
                'Total_Intergenic_Sites': total_sites,
                'Syn_Sites_Mutated': dnds_data[sample]['syn_sites_mutated'],
                'Total_Syn_Sites': dnds_data[sample]['syn_sites'],
                'Non_Syn_Sites_Mutated': dnds_data[sample]['non_syn_sites_mutated'],
                'Total_Non_Syn_Sites': dnds_data[sample]['non_syn_sites'],
                'Is_Control': 'Yes' if any(x in sample for x in ['NT', 'Minus', 'Pre']) else 'No'
            })
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: dN/dS vs Intergenic Rate
    for is_control, color, label in [(True, 'lightgrey', 'Control'), (False, 'darkorange', 'EMS Treated')]:
        mask = [x == is_control for x in plot_data['is_control']]
        ax1.scatter(
            [r for r, m in zip(plot_data['intergenic_rate'], mask) if m],
            [d for d, m in zip(plot_data['dnds'], mask) if m],
            alpha=0.7,
            color=color,
            label=label,
            s=100
        )
    
    # Add sample labels to first plot
    for i, txt in enumerate(plot_data['sample']):
        abbreviated_name = extract_sample_id(txt)
        ax1.annotate(
            abbreviated_name,
            (plot_data['intergenic_rate'][i], plot_data['dnds'][i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=8,
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5)
        )
    
    ax1.set_xlabel('Intergenic Mutation Rate')
    ax1.set_ylabel('Genome-wide dN/dS Ratio')
    ax1.set_title('dN/dS vs Intergenic Mutation Rate')
    
    # Add correlation coefficient to first plot
    r1 = np.corrcoef(plot_data['intergenic_rate'], plot_data['dnds'])[0,1]
    ax1.text(0.05, 0.95, f'Correlation: {r1:.3f}', 
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 2: dN/dS vs Synonymous Sites Mutated Rate
    for is_control, color, label in [(True, 'lightgrey', 'Control'), (False, 'darkorange', 'EMS Treated')]:
        mask = [x == is_control for x in plot_data['is_control']]
        ax2.scatter(
            [r for r, m in zip(plot_data['syn_sites_mutated_rate'], mask) if m],
            [d for d, m in zip(plot_data['dnds'], mask) if m],
            alpha=0.7,
            color=color,
            label=label,
            s=100
        )
    
    # Add sample labels to second plot
    for i, txt in enumerate(plot_data['sample']):
        abbreviated_name = extract_sample_id(txt)
        ax2.annotate(
            abbreviated_name,
            (plot_data['syn_sites_mutated_rate'][i], plot_data['dnds'][i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=8,
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5)
        )
    
    ax2.set_xlabel('Synonymous Sites Mutated Rate')
    ax2.set_ylabel('Genome-wide dN/dS Ratio')
    ax2.set_title('dN/dS vs Synonymous Sites Mutated Rate')
    
    # Add correlation coefficient to second plot
    r2 = np.corrcoef(plot_data['syn_sites_mutated_rate'], plot_data['dnds'])[0,1]
    ax2.text(0.05, 0.95, f'Correlation: {r2:.3f}', 
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Plot 3: dN/dS vs Non-synonymous Sites Mutated Rate
    for is_control, color, label in [(True, 'lightgrey', 'Control'), (False, 'darkorange', 'EMS Treated')]:
        mask = [x == is_control for x in plot_data['is_control']]
        ax3.scatter(
            [r for r, m in zip(plot_data['non_syn_sites_mutated_rate'], mask) if m],
            [d for d, m in zip(plot_data['dnds'], mask) if m],
            alpha=0.7,
            color=color,
            label=label,
            s=100
        )
    
    # Add sample labels to third plot
    for i, txt in enumerate(plot_data['sample']):
        abbreviated_name = extract_sample_id(txt)
        ax3.annotate(
            abbreviated_name,
            (plot_data['non_syn_sites_mutated_rate'][i], plot_data['dnds'][i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=8,
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5)
        )
    
    ax3.set_xlabel('Non-synonymous Sites Mutated Rate')
    ax3.set_ylabel('Genome-wide dN/dS Ratio')
    ax3.set_title('dN/dS vs Non-synonymous Sites Mutated Rate')
    
    # Add correlation coefficient to third plot
    r3 = np.corrcoef(plot_data['non_syn_sites_mutated_rate'], plot_data['dnds'])[0,1]
    ax3.text(0.05, 0.95, f'Correlation: {r3:.3f}', 
             transform=ax3.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dnds_vs_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save CSV data
    csv_path = os.path.join(output_dir, 'mutation_rates_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

def main():
    parser = argparse.ArgumentParser(description='Generate plots from analysis results')
    parser.add_argument('-r', '--results_dir', type=str, required=True,
                       help='Path to results directory containing analysis outputs')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='Path to output directory for plots')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for and process different result files
    results_dir = Path(args.results_dir)
    
    # Mutation frequency plots
    nuc_mut_dir = results_dir / 'nuc_muts'
    jsons = list(nuc_mut_dir.glob('*.json'))
    if jsons:  # Only proceed if JSON files exist
        basecounts_file = results_dir / 'basecounts.json'
        if basecounts_file.exists():
            print("Generating mutation frequency plots...")
            with open(basecounts_file) as f:
                base_counts = json.load(f)
            # Remove basecounts.json from jsons list
            jsons = [j for j in jsons if j.stem != 'basecounts']
            if jsons:  # Check if there are mutation JSONs after filtering
                mutation_type_barplot(jsons, base_counts, args.output_dir)

    # dN/dS ratio plots
    aa_mut_dir = results_dir / 'aa_muts'
    aa_mut_files = list(aa_mut_dir.glob('*.json'))
    if aa_mut_files:
        # Filter out basecounts or other non-mutation files
        aa_mut_files = [f for f in aa_mut_files if f.stem != 'basecounts']
        if aa_mut_files:
            # Generate individual sample plots
            for aa_file in aa_mut_files:
                print(f"Generating dN/dS plots for {aa_file.stem}...")
                with open(aa_file) as f:
                    aa_mutations = json.load(f)
                plot_dnds_ratio(aa_mutations, args.output_dir, aa_file.stem)
            
            # Generate combined plot
            print("\nGenerating combined dN/dS plot...")
            plot_combined_dnds_ratios(aa_mut_files, args.output_dir)
            
            # Generate sample-level comparison plot
            print("Generating sample-level dN/dS comparison plot...")
            plot_sample_dnds_vs_coverage(aa_mut_files, args.output_dir)
            
            # Generate dN/dS vs intergenic plot if both files exist
            genome_dnds = results_dir / 'results' / 'genome_dnds.json'
            intergenic = results_dir / 'results' / 'intergeniccounts.json'
            if genome_dnds.exists() and intergenic.exists():
                print("Generating dN/dS vs intergenic mutation plot...")
                plot_dnds_vs_intergenic(str(results_dir), args.output_dir)

    # Kmer context plots
    kmer_file = results_dir / 'kmer_context.json'
    kmer_counts_file = results_dir / 'kmer_counts.json'
    if kmer_file.exists() and kmer_counts_file.exists():
        # Verify files contain data
        if os.path.getsize(kmer_file) > 0 and os.path.getsize(kmer_counts_file) > 0:
            print("Generating kmer context plots...")
            with open(kmer_file) as f:
                kmer_context = json.load(f)
            with open(kmer_counts_file) as f:
                genome_kmer_counts = json.load(f)
            plot_kmer_context(kmer_context, genome_kmer_counts, args.output_dir)
        
if __name__ == "__main__":
    main()