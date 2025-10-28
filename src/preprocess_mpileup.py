import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
import json
from modules.parse import SeqContext
from BCBio.GFF import GFFExaminer
from BCBio import GFF
import yaml
from multiprocessing import Pool, cpu_count
from typing import Set, Dict, List

def analyze_read_position_bias(reads, alt):
    """
    Analyze positional bias using KS test to compare distributions of
    ALT base positions vs other base positions
    Returns (p-value, KS statistic) tuple - lower p-values indicate more significant positional bias
    """
    alt_positions = []  # positions of alt bases
    other_positions = []  # positions of other bases
    total_bases = 0  # count of all valid bases (excluding markers)
    current_pos = 0
    i = 0
    
    while i < len(reads):
        if reads[i] in "^":  # Start of read
            i += 2  # Skip quality character
            continue
        elif reads[i] in "$":  # End of read
            i += 1
            continue
        elif reads[i] in "+-":  # Handle indels
            i += 1
            indel_len = ""
            while reads[i].isdigit():
                indel_len += reads[i]
                i += 1
            i += int(indel_len)
            continue
        elif reads[i] in "*":  # Skip deletions
            i += 1
            continue
            
        # Record position of base
        if reads[i] in ".,AGCT":  # Only count actual bases
            total_bases += 1
            if reads[i] in alt.upper() + alt.lower():
                alt_positions.append(current_pos)
            else:
                other_positions.append(current_pos)
            
        current_pos += 1
        i += 1
    
    # Calculate proportion of alt bases
    alt_proportion = len(alt_positions) / total_bases if total_bases > 0 else 0
    
    # Check for fixation events (high alt proportion)
    if alt_proportion > 0.9:  # More than 90% alt bases
        return -1, -1  # Return values indicating fixation
    
    # If we have very few alt bases relative to read length,
    # analyze their distribution to detect clustering
    if alt_proportion < 0.1:  # Less than 10% alt bases
        if len(alt_positions) < 2:
            return 1.0, 0.0  # Single alt base - can't be clustered
            
        # Calculate read length (excluding markers)
        read_length = current_pos
        
        # Calculate expected mean distance between alt bases if randomly distributed
        expected_distance = read_length / (len(alt_positions) + 1)
        
        # Calculate actual mean distance between consecutive alt bases
        alt_positions.sort()
        distances = [alt_positions[i+1] - alt_positions[i] for i in range(len(alt_positions)-1)]
        actual_mean_distance = sum(distances) / len(distances)
        
        # If actual distances are significantly larger than expected (>60% of expected),
        # consider them well-spread (not clustered)
        if actual_mean_distance > (0.6 * expected_distance):
            return 1.0, 0.0  # Return values indicating no clustering
        else:
            return 0.0, 1.0  # Return values indicating clustering

    # Only perform KS test if we have enough alt bases for meaningful comparison
    if len(alt_positions) >= 2 and len(other_positions) >= 2:
        statistic, pvalue = stats.ks_2samp(alt_positions, other_positions)
        return pvalue, statistic
    
    # Not enough data points for meaningful test
    return 1.0, 0.0  # Return values indicating no significant bias


def get_most_common_alt(reads, ref):
    """
    Find the most common alternate allele in the reads.
    Returns tuple of (alt_base, count)
    """
    counts = defaultdict(int)
    i = 0
    while i < len(reads):
        if reads[i] in ".,": # Reference match
            i += 1
            continue
        elif reads[i] in "^": # Start of read
            i += 2  # Skip quality character
            continue
        elif reads[i] in "$": # End of read
            i += 1
            continue
        elif reads[i] in "+-":  # Skip indels
            i += 1
            indel_len = ""
            while reads[i].isdigit():
                indel_len += reads[i]
                i += 1
            i += int(indel_len)
            continue
        elif reads[i] in "*": # Skip deletions
            i += 1
            continue
            
        # Count the base if it's not the reference
        base = reads[i].upper()
        if base in 'AGCT' and base != ref.upper():
            counts[base] += 1
        i += 1
    
    if not counts:
        return (None, 0)
    
    # Return the most common alternate allele and its count
    return max(counts.items(), key=lambda x: x[1])


def get_depth_threshold(mpileup_file: str, percentile: float = 5.0) -> int:
    """
    Calculate depth threshold based on distribution of depths in mpileup file.
    
    Args:
        mpileup_file (str): Path to mpileup file
        percentile (float): Percentile threshold for depth filtering (0-100)
        
    Returns:
        int: Depth threshold that excludes bottom percentile of depths
    """
    print("Collecting depth distribution for filtering...")
    depths = []
    with open(mpileup_file) as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:  # Sample every 5,000th position to save memory
                entry = line.split()
                if len(entry) >= 4:  # Ensure the line has enough fields
                    depth = int(entry[3])
                    depths.append(depth)
    
    # Calculate depth threshold based on percentile
    if depths:
        depth_threshold = max(1, np.percentile(depths, percentile))
        print(f"Depth threshold set to {depth_threshold} (excludes bottom {percentile}%)")
        return depth_threshold
    else:
        print("Warning: Could not calculate depth distribution, using default threshold of 1")
        return 1


def count_bases(reads, ref):
    """
    Count occurrences of different bases in reads string
    Returns tuple of (counts_dict, most_common_alt, total_alt_count) where:
    - counts_dict has counts of A,C,G,T,N
    - most_common_alt is the most frequent non-reference base
    - total_alt_count is the sum of all non-reference bases
    """
    counts = defaultdict(int)
    i = 0
    while i < len(reads):
        if reads[i] in "^":  # Start of read
            i += 2
            continue
        elif reads[i] in "$":  # End of read
            i += 1
            continue
        elif reads[i] in "+-":  # Handle indels
            i += 1
            indel_len = ""
            while reads[i].isdigit():
                indel_len += reads[i]
                i += 1
            i += int(indel_len)
            continue
        elif reads[i] in "*":  # Skip deletions
            i += 1
            continue
            
        base = reads[i].upper()
        if base in 'ACGTN':
            counts[base] += 1
        i += 1
    
    # Calculate total alternate base count
    total_alt_count = sum(counts[b] for b in 'ACGT' if b != ref.upper())
    
    # Find most common alt allele
    alt_bases = {b: counts[b] for b in 'ACGT' if b != ref.upper()}
    most_common_alt = max(alt_bases.items(), key=lambda x: x[1]) if alt_bases else (None, 0)
    
    return counts, most_common_alt[0], total_alt_count


def analyze_n_alt_ratios(plot_data):
    """
    Analyze the ratios between N and Alt frequencies to identify common patterns.
    
    Args:
        plot_data (pd.DataFrame): DataFrame containing N_freq and Alt_freq columns
        
    Returns:
        dict: Analysis results including common ratios and their frequencies
    """
    # Calculate N:Alt ratios
    ratios = plot_data['N_freq'] / plot_data['Alt_freq']
    
    # Round ratios to nearest 0.5 to identify common patterns
    rounded_ratios = np.round(ratios * 2) / 2
    
    # Get most common ratios and their counts
    ratio_counts = rounded_ratios.value_counts().head(10)
    
    # Calculate quality metrics
    quality_score = plot_data['depth'] / (plot_data['N_count'] + plot_data['Alt_count'])
    
    # Group data by rounded ratios and calculate mean quality
    ratio_quality = pd.DataFrame({
        'ratio': rounded_ratios,
        'quality': quality_score
    }).groupby('ratio').agg({
        'quality': ['mean', 'std', 'count']
    }).reset_index()
    
    # Convert to JSON-serializable format
    ratio_counts_dict = {str(k): int(v) for k, v in ratio_counts.items()}
    ratio_quality_dict = {
        'ratios': ratio_quality['ratio'].tolist(),
        'quality_mean': ratio_quality['quality']['mean'].tolist(),
        'quality_std': ratio_quality['quality']['std'].tolist(),
        'quality_count': ratio_quality['quality']['count'].tolist()
    }
    
    return {
        'common_ratios': ratio_counts_dict,
        'ratio_quality': ratio_quality_dict
    }


def plot_n_vs_alt_distributions(sample_data, output_dir):
    """
    Create individual plots for each sample comparing N and alternate base distributions
    
    Args:
        sample_data (dict): Dictionary with sample names as keys and lists of (depth, n_count, alt_count) as values
        output_dir (str): Directory to save plots
    """
    plots_dir = os.path.join(output_dir, 'n_alt_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Store ratio analysis results across samples
    all_samples_ratios = {}
    
    for sample, measurements in sample_data.items():
        # Convert sample data to DataFrame
        plot_data = pd.DataFrame([{
            'depth': depth,
            'N_freq': n_count / depth if depth > 0 else 0,
            'Alt_freq': alt_count / depth if depth > 0 else 0,
            'N_count': n_count,
            'Alt_count': alt_count
        } for depth, n_count, alt_count in measurements if n_count > 0 and alt_count > 0])
        
        if plot_data.empty:
            print(f"No data with both N and Alt bases for sample {sample}. Skipping plot.")
            continue
            
        # Remove outliers from both frequencies
        plot_data = remove_outliers(plot_data, 'N_freq')
        plot_data = remove_outliers(plot_data, 'Alt_freq')
        
        # Analyze N:Alt ratios
        ratio_analysis = analyze_n_alt_ratios(plot_data)
        all_samples_ratios[sample] = ratio_analysis
        
        # Create figure with gridspec
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)
        
        # Main scatter plot with hexbin
        ax_main = fig.add_subplot(gs[1:, :2])
        hb = ax_main.hexbin(plot_data['N_freq'], plot_data['Alt_freq'], 
                          gridsize=30, cmap='YlOrRd')
        ax_main.set_xlabel('N Frequency')
        ax_main.set_ylabel('Alt Frequency')
        
        # Add ratio guidelines
        max_freq = max(plot_data['N_freq'].max(), plot_data['Alt_freq'].max())
        for ratio in [0.5, 1, 2, 3, 4, 5]:
            ax_main.plot([0, max_freq], [0, max_freq/ratio], 
                        '--', color='blue', alpha=0.3, 
                        label=f'1:{ratio} ratio')
        ax_main.legend()
        
        plt.colorbar(hb, ax=ax_main, label='Count')
        
        # Top histogram for N frequency
        ax_hist_x = fig.add_subplot(gs[0, :2])
        sns.histplot(data=plot_data, x='N_freq', bins=50, ax=ax_hist_x)
        ax_hist_x.set_ylabel('Count')
        
        # Right histogram for Alt frequency
        ax_hist_y = fig.add_subplot(gs[1:, 2])
        sns.histplot(data=plot_data, y='Alt_freq', bins=50, ax=ax_hist_y)
        ax_hist_y.set_xlabel('Count')
        
        # Ratio distribution plot
        ax_ratio = fig.add_subplot(gs[0, 2])
        ratio_series = pd.Series(ratio_analysis['common_ratios'])
        ratio_series.plot(kind='bar', ax=ax_ratio)
        ax_ratio.set_title('Most Common N:Alt Ratios')
        ax_ratio.set_xlabel('N:Alt Ratio')
        ax_ratio.set_ylabel('Count')
        plt.xticks(rotation=45)
        
        # Stats in text box
        stats_text = (
            f"Summary Statistics:\n\n"
            f"N Frequency:\n"
            f"  Mean: {plot_data['N_freq'].mean():.3f}\n"
            f"  Median: {plot_data['N_freq'].median():.3f}\n"
            f"  Std: {plot_data['N_freq'].std():.3f}\n\n"
            f"Alt Frequency:\n"
            f"  Mean: {plot_data['Alt_freq'].mean():.3f}\n"
            f"  Median: {plot_data['Alt_freq'].median():.3f}\n"
            f"  Std: {plot_data['Alt_freq'].std():.3f}\n\n"
            f"Most Common N:Alt Ratios:\n" +
            "\n".join(f"  {float(ratio):.1f}: {count}" 
                     for ratio, count in sorted(
                         ratio_analysis['common_ratios'].items(), 
                         key=lambda x: x[1], 
                         reverse=True
                     )[:5]) +
            f"\n\nCorrelation: {plot_data['N_freq'].corr(plot_data['Alt_freq']):.3f}\n"
            f"Total Sites: {len(plot_data):,}\n"
            f"Mean Depth: {plot_data['depth'].mean():.1f}"
        )
        
        # Add text box
        plt.figtext(0.95, 0.95, stats_text,
                   fontsize=10,
                   fontfamily='monospace',
                   bbox=dict(facecolor='white', alpha=0.8),
                   verticalalignment='top',
                   horizontalalignment='right')
        
        # Overall title
        fig.suptitle(f'N vs Alt Base Analysis - {sample}\n(outliers removed)', 
                    fontsize=14, y=0.95)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{sample}_n_alt_analysis.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    # Save ratio analysis results
    with open(os.path.join(output_dir, 'n_alt_ratio_analysis.json'), 'w') as f:
        json.dump({
            sample: {
                'common_ratios': ratios['common_ratios'],
                'ratio_quality': ratios['ratio_quality']
            }
            for sample, ratios in all_samples_ratios.items()
        }, f, indent=2)


def load_config(config_path):
    """
    Load configuration file to get paths for genome FASTA and GFF.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Tuple[str, str]: Paths to genome FASTA and GFF files.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    genome_fasta = config['references']['genomic_fna']
    genome_gff = config['references']['annotation']
    return genome_fasta, genome_gff


def filter_genic_positions(filtered_positions: Set[int], seq_context: SeqContext) -> Dict[str, List[int]]:
    """
    Organize filtered positions by gene, converting to relative positions within each gene.
    
    Args:
        filtered_positions (Set[int]): Set of filtered genomic positions
        seq_context (SeqContext): SeqContext object for gene information
        
    Returns:
        Dict[str, List[int]]: Dictionary mapping GeneIDs to lists of relative filtered positions
    """
    genic_positions = {}
    
    with open(seq_context.annot) as handle:
        for rec in GFF.parse(handle):
            for feat in rec.features:
                if feat.type == 'gene':
                    # Get GeneID from Dbxref
                    gene_id = feat.qualifiers['Dbxref'][0].split(':')[-1]
                    gene_start = feat.location.start
                    gene_end = feat.location.end
                    
                    # Find filtered positions within this gene
                    gene_filtered = [
                        pos - gene_start  # Convert to relative position
                        for pos in filtered_positions 
                        if gene_start <= pos < gene_end
                    ]
                    
                    if gene_filtered:
                        genic_positions[gene_id] = sorted(gene_filtered)
    
    return genic_positions


def process_line(line, depth_threshold, seq_context, keep_low_alt=False, alt_threshold=3):
    """Modified to optionally retain low-alt sites (1-2 alt reads) that pass other filters."""
    fields = line.strip().split('\t')
    if len(fields) < 6:
        return None, (set(), set(), set()), None  # (depth_filtered, n_filtered, bias_filtered)
    
    # Check depth threshold
    depth = int(fields[3])
    if depth < depth_threshold:
        return None, (set([int(fields[1])]), set(), set()), None
    
    chrom = fields[0]
    pos = int(fields[1]) 
    ref = fields[2]
    reads = fields[4]
    
    # Count bases and get most common alt in one pass
    base_counts, alt, alt_base_count = count_bases(reads, ref)
    
    if not alt or alt_base_count == 0:  # No alternate allele found
        return None, (set(), set(), set()), None

    if alt_base_count < alt_threshold:
        if not keep_low_alt:
            n_filtered = set([pos])
            return None, (set(), n_filtered, set()), None
        # Evaluate remaining filters and keep if pass (as non-mutation)
        
        # N contamination filter
        if base_counts['N'] > alt_base_count * 10:
            n_filtered = set([pos])
            return None, (set(), n_filtered, set()), None
        
        # Position bias test
        pvalue, statistic = analyze_read_position_bias(reads, alt)
        if (pvalue == -1 and statistic == -1) or (statistic < 0.25 and pvalue > 0.01):
            # Keep in filtered output, but no mutation_type
            return (chrom, pos, ref, fields[3], reads, fields[5], None), (set(), set(), set()), None
        else:
            bias_filtered = set([pos])
            return None, (set(), set(), bias_filtered), None
    
    # Initialize filter sets
    depth_filtered = set()
    n_filtered = set()
    bias_filtered = set()
    
    # Calculate bias score using KS test
    pvalue, statistic = analyze_read_position_bias(reads, alt)
    
    # Track and save C>T and G>A mutations
    mutation_type = None
    if ref.upper() == 'C' and alt == 'T':
        mutation_type = 'C>T'
    elif ref.upper() == 'G' and alt == 'A':
        mutation_type = 'G>A'
    
    measurement = None
    
    # Only collect data for sites that pass the KS test
    if (pvalue == -1 and statistic == -1) or (statistic < 0.25 and pvalue > 0.01):
        n_count = base_counts['N']
        measurement = (depth, n_count, alt_base_count)
    
    # Filter if more Ns than alternate bases
    
    if base_counts['N'] > alt_base_count*10:
        n_filtered.add(pos)
        return None, (depth_filtered, n_filtered, bias_filtered), measurement
    
    # Handle fixation events
    if pvalue == -1 and statistic == -1:
        return (chrom, pos, ref, fields[3], reads, fields[5], mutation_type), (depth_filtered, n_filtered, bias_filtered), measurement
    
    # Write results if passes position bias test (low KS statistic, high p-value)
    if statistic < 0.25 and pvalue > 0.01:
        return (chrom, pos, ref, fields[3], reads, fields[5], mutation_type), (depth_filtered, n_filtered, bias_filtered), measurement
    else:
        bias_filtered.add(pos)
        return None, (depth_filtered, n_filtered, bias_filtered), measurement


def analyze_high_n_ratios(sample_data, output_dir):
    """
    Analyze and plot the distribution of high N:Alt ratios across samples.
    
    Args:
        sample_data (dict): Dictionary with sample names as keys and lists of (depth, n_count, alt_count) as values
        output_dir (str): Directory to save plots
    """
    # Create data structure to hold ratio distributions for each sample
    sample_ratios = {}
    high_ratio_counts = {}
    
    for sample, measurements in sample_data.items():
        ratios = []
        high_ratio_count = 0
        total_positions = 0
        
        for depth, n_count, alt_count in measurements:
            if alt_count > 0:  # Avoid division by zero
                ratio = n_count / alt_count
                ratios.append(ratio)
                total_positions += 1
                if ratio > 5:  # Count positions with very high N:Alt ratios
                    high_ratio_count += 1
        
        if ratios:
            sample_ratios[sample] = ratios
            high_ratio_counts[sample] = {
                'count': high_ratio_count,
                'total': total_positions,
                'percentage': (high_ratio_count / total_positions) * 100 if total_positions > 0 else 0
            }
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Box plot of ratio distributions
    box_data = [ratios for ratios in sample_ratios.values()]
    ax1.boxplot(box_data, labels=list(sample_ratios.keys()), showfliers=False)
    ax1.set_ylabel('N:Alt Ratio')
    ax1.set_title('Distribution of N:Alt Ratios Across Samples\n(excluding outliers)')
    plt.xticks(rotation=45)
    
    # Bar plot of high ratio percentages
    samples = list(high_ratio_counts.keys())
    percentages = [data['percentage'] for data in high_ratio_counts.values()]
    counts = [data['count'] for data in high_ratio_counts.values()]
    
    bars = ax2.bar(samples, percentages)
    ax2.set_ylabel('Percentage of Positions with N:Alt Ratio > 5')
    ax2.set_title('Prevalence of High N:Alt Ratio Positions')
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'n_alt_ratio_distribution.png'))
    plt.close()
    
    # Save detailed statistics
    stats = {
        sample: {
            'high_ratio_positions': data['count'],
            'total_positions': data['total'],
            'percentage': data['percentage'],
            'ratio_stats': {
                'mean': np.mean(sample_ratios[sample]),
                'median': np.median(sample_ratios[sample]),
                'std': np.std(sample_ratios[sample]),
                'q1': np.percentile(sample_ratios[sample], 25),
                'q3': np.percentile(sample_ratios[sample], 75)
            }
        }
        for sample, data in high_ratio_counts.items()
    }
    
    with open(os.path.join(output_dir, 'high_n_ratio_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main(mpileup_dir, output_dir, config_path, keep_low_alt=False, write_sites=False, alt_threshold=3):
    """
    Process all mpileup files in a directory and detect sequence bias.
    
    Args:
        mpileup_dir (str): Path to directory containing mpileup files
        output_dir (str): Path to output directory
        config_path (str): Path to configuration file
        keep_low_alt (bool): If True, keep sites with 1-2 alt reads that pass other filters in filtered output
        write_sites (bool): If True, write a sites file listing all positions with depth >= threshold regardless of alternate evidence
        alt_threshold (int): Minimum number of alternate alleles to consider a mutation (default: 3)
    """
    # Load genome paths from config
    genome_fasta, genome_gff = load_config(config_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SeqContext for gene checking
    seq_context = SeqContext(genome_fasta, genome_gff)
    
    # Add dictionary to store N vs Alt data for all samples
    sample_n_alt_data = {}
    filtered_positions = {}  # Dictionary to store filtered positions for each sample
    
    # Process each mpileup file in directory
    for mpileup_file in glob.glob(os.path.join(mpileup_dir, "*.txt")):
        sample_name = os.path.basename(mpileup_file).replace(".txt", "")
        output_file = os.path.join(output_dir, f"{sample_name}_filtered.txt")
        mutation_file = os.path.join(output_dir, f"{sample_name}_mutations.txt")
        sites_file = os.path.join(output_dir, f"{sample_name}_sites.txt") if write_sites else None
        
        print(f"\nProcessing {sample_name}...")
        
        # Calculate depth threshold for this file
        depth_threshold = get_depth_threshold(mpileup_file)
        
        # Initialize data structures
        sample_n_alt_data[sample_name] = []
        filtered_positions[sample_name] = set()
        
        # Initialize tracking sets
        depth_filtered_sites = set()
        n_filtered_sites = set()
        bias_filtered_sites = set()
        low_alt_filtered_sites = set()
        
        with open(mpileup_file) as f_in, \
             open(output_file, 'w') as f_out, \
             open(mutation_file, 'w') as f_mut, \
             (open(sites_file, 'w') if write_sites else open(os.devnull, 'w')) as f_sites:
            
            lines = f_in.readlines()
            
            # Write sites file: callable positions = pass coverage and, if alt present, pass N-content and bias filters
            if write_sites:
                for line in lines:
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    try:
                        depth = int(parts[3])
                    except ValueError:
                        continue
                    if depth < depth_threshold:
                        continue
                    ref = parts[2]
                    reads = parts[4]
                    # Count bases and alt evidence
                    base_counts, alt_base, alt_count = count_bases(reads, ref)
                    if alt_count == 0:
                        # No alt evidence: include as callable (passed coverage)
                        f_sites.write(line)
                        continue
                    # N-content filter relative to alt evidence
                    if base_counts.get('N', 0) > alt_count * 10:
                        continue
                    # Positional bias filter
                    pvalue, statistic = analyze_read_position_bias(reads, alt_base if alt_base else '')
                    passes_bias = (pvalue == -1 and statistic == -1) or (statistic < 0.25 and pvalue > 0.01)
                    if passes_bias:
                        f_sites.write(line)
            
            with Pool(cpu_count()) as pool:
                results = pool.starmap(
                    process_line,
                    [(line, depth_threshold, seq_context, keep_low_alt, alt_threshold) for line in lines]
                )
            
            # Counter for lines with mutations written to output
            written_mutation_lines = 0

            # Process results and accumulate filtered positions
            for result, (depth_filtered, n_filtered, bias_filtered), measurement in results:
                if measurement is not None:
                    sample_n_alt_data[sample_name].append(measurement)
                
                # Accumulate filtered positions by reason
                depth_filtered_sites.update(depth_filtered)
                bias_filtered_sites.update(bias_filtered)
                filtered_positions[sample_name].update(depth_filtered)
                filtered_positions[sample_name].update(bias_filtered)
                if n_filtered and not depth_filtered and not bias_filtered:
                    # low-alt filter, do not add to filtered_positions
                    low_alt_filtered_sites.update(n_filtered)
                else:
                    n_filtered_sites.update(n_filtered)
                    filtered_positions[sample_name].update(n_filtered)
                
                if result:
                    chrom, pos, ref, depth, reads, qual, mutation_type = result
                    f_out.write(f"{chrom}\t{pos}\t{ref}\t{depth}\t{reads}\t{qual}\n")
                    if mutation_type:
                        f_mut.write(f"{chrom}\t{pos}\t{ref}\t{depth}\t{reads}\t{qual}\t{mutation_type}\n")
                    written_mutation_lines += 1
        
        print(f"Results for {sample_name}:")
        print(f"Total sites processed: {len(lines)}")
        print(f"Sites filtered due to depth: {len(depth_filtered_sites)}")
        print(f"Sites filtered due to excess N bases: {len(n_filtered_sites)}")
        print(f"Sites filtered due to position bias: {len(bias_filtered_sites)}")
        print(f"Sites filtered due to <4 alternate alleles: {len(low_alt_filtered_sites)}")
        print(f"Sites with mutations remaining: {len(lines) - len(depth_filtered_sites) - len(n_filtered_sites) - len(bias_filtered_sites) - len(low_alt_filtered_sites)}")
        print(f"Lines with mutations written to output: {written_mutation_lines}")
    
    
    # Filter and save genic positions to JSON
    try:
        for sample, positions in filtered_positions.items():
            genic_filtered = filter_genic_positions(positions, seq_context)
            json_path = os.path.join(output_dir, f"{sample}_filtered_positions.json")
            with open(json_path, 'w') as json_file:
                json.dump(genic_filtered, json_file, indent=2)
    except Exception as e:
        print(f"Error filtering genic positions: {e}")
    
    # After processing all samples, create the visualization
    plot_n_vs_alt_distributions(sample_n_alt_data, output_dir)
    
    # After processing all samples and before creating other visualizations
    high_ratio_stats = analyze_high_n_ratios(sample_n_alt_data, output_dir)
    
    # Print summary
    print("\nHigh N:Alt Ratio Analysis:")
    for sample, stats in high_ratio_stats.items():
        print(f"\n{sample}:")
        print(f"  Positions with N:Alt ratio > 5: {stats['high_ratio_positions']:,}")
        print(f"  Percentage of total positions: {stats['percentage']:.2f}%")
        print(f"  Median N:Alt ratio: {stats['ratio_stats']['median']:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process mpileup files to detect sequence bias')
    parser.add_argument('mpileup_dir', help='Path to directory containing mpileup files')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('config_path', help='Path to configuration file')
    parser.add_argument('--keep-low-alt', action='store_true', help='Keep sites with 1-2 alt reads that pass other filters in filtered output')
    parser.add_argument('--write-sites', action='store_true', help='Also write <sample>_sites.txt containing all positions with depth >= threshold regardless of alternate evidence')
    parser.add_argument('--alt-threshold', type=int, default=3, help='Minimum number of alternate alleles to consider a mutation (default: 3) (for bias and n-content filters)')
    args = parser.parse_args()
    
    main(args.mpileup_dir, args.output_dir, args.config_path, keep_low_alt=args.keep_low_alt, write_sites=args.write_sites, alt_threshold=args.alt_threshold)
