import json
import argparse
import matplotlib.pyplot as plt
from glob import glob
import os
import csv
import seaborn as sns
import pandas as pd
import numpy as np

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
        sample = mutation_json.split('/')[-1].split('.')[0]
        
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

def plot_kmer_context(data_dict: dict, output_dir: str) -> None:
    '''Plot the kmer context barplot.
    
    Args:
        data_dict (dict): Dictionary of kmer context data
        output_dir (str): Directory to save output plot
    '''
    
    # Set output path for the plot 
    output_path = os.path.join(output_dir, 'kmer_context.png')

    # initialize a dictionary to store the observed kmers
    observed_kmers = {}
    
    # sum observed kmers across all treated samples
    for sample in data_dict:
        if 'EMS' in sample:
            for kmer in data_dict[sample]:
                if kmer not in observed_kmers:
                    observed_kmers[kmer] = data_dict[sample][kmer]
                else:
                    observed_kmers[kmer] += data_dict[sample][kmer]
    
    # plot the kmer context barplot 
    # Sort kmers by count and get top 10
    top_kmers = dict(sorted(observed_kmers.items(), key=lambda x: x[1], reverse=True)[:30])
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    plt.bar(top_kmers.keys(), top_kmers.values(), color='darkorange')
    
    # Customize plot
    plt.xlabel('Kmer Context')
    plt.ylabel('Count')
    plt.title('Top 10 Most Frequent Kmer Contexts')
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
          
def plot_ems_mutation_frequencies(data_path: str, output_dir: str) -> None:
    '''Create bar plot of mutation frequencies using matplotlib.
    
    Args:
        data_path (str): Path to CSV file containing mutation frequencies
        output_dir (str): Directory to save output plot
    '''
    # Set output path for the plot
    output_path = os.path.join(output_dir, 'ems_mutation_frequencies.png')
    
    # Read the mutation frequencies from the CSV file
    df = pd.read_csv(data_path)
    
    # Split data into control and treated groups
    control = df[df['ems'] == '-'].groupby('mutation')['norm_count'].mean()
    treated = df[df['ems'] == '+'].groupby('mutation')['norm_count'].mean()
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Create x-axis positions
    mutations = control.index
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

def main():
    parser = argparse.ArgumentParser(description='Plot data from a JSON file.')
    parser.add_argument('-j', '--result_json', type=str, help='Path to the final results JSON file')
    parser.add_argument('-m', '--mutation_jsons', type=str, help='Path to the nucleotide mutation JSON file directory')
    parser.add_argument('-k', '--kmer_context', type=str, help='Path to the kmer context JSON file')
    parser.add_argument('-o', '--output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()

    if args.result_json:
        json_file_path = args.result_json
        result_data = load_json(json_file_path)
    
    if args.mutation_jsons:
        mutation_json_file_dir = args.mutation_jsons
        
        # collect all the mutation json files
        jsons = glob(mutation_json_file_dir + '/*.json')
        
        # load the base counts json file into a dictionary  
        with open(mutation_json_file_dir + '/../basecounts.json') as f:
            base_counts = json.load(f)
        
        # plot the mutation type barplot
        mutation_type_barplot(jsons, base_counts, args.output_dir)
    
    if args.kmer_context:
        kmer_context_file_path = args.kmer_context
        kmer_context_data = load_json(kmer_context_file_path)
        
        # plot the kmer context barplot
        plot_kmer_context(kmer_context_data, args.output_dir)
        
        
if __name__ == "__main__":
    main()