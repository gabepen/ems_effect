import json
import argparse
import matplotlib.pyplot as plt
from glob import glob
import os
import csv
import seaborn as sns
import pandas as pd


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

def plot_ems_mutation_frequencies(data_path: str, output_dir: str) -> None:
    
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
        
if __name__ == "__main__":
    main()