import numpy as np
from collections import defaultdict
from scipy import stats

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
    
    # Need enough data points for meaningful test
    if len(other_positions) < 2:
        # this is likely a fixation event
        return -1, -1 # Return trackable p-value and KS statistic
        
    # Calculate proportion of alt bases
    alt_proportion = len(alt_positions) / total_bases
    
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
        
        # If actual distances are significantly larger than expected (>75% of expected),
        # consider them well-spread (not clustered)
        if actual_mean_distance > (0.75 * expected_distance):
            return 1.0, 0.0  # Return values indicating no clustering

    # Perform two-sample KS test
    statistic, pvalue = stats.ks_2samp(alt_positions, other_positions)
    
    return pvalue, statistic

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
    Returns tuple of (counts_dict, most_common_alt, alt_count) where:
    - counts_dict has counts of A,C,G,T,N and total valid bases
    - most_common_alt is the most frequent non-reference base
    - alt_count is the count of the most common alt base
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
    
    counts['total'] = sum(counts[b] for b in 'ACGT')
    
    # Find most common alt allele
    alt_bases = {b: counts[b] for b in 'ACGT' if b != ref.upper()}
    most_common_alt = max(alt_bases.items(), key=lambda x: x[1]) if alt_bases else (None, 0)
    
    return counts, most_common_alt[0], most_common_alt[1]

def main(mpileup_dir, output_dir):
    """
    Process all mpileup files in a directory and detect sequence bias.
    
    Args:
        mpileup_dir (str): Path to directory containing mpileup files
        output_dir (str): Path to output directory
    """
    import os
    import glob
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each mpileup file in directory
    for mpileup_file in glob.glob(os.path.join(mpileup_dir, "*.txt")):
        sample_name = os.path.basename(mpileup_file).replace(".txt", "")
        output_file = os.path.join(output_dir, f"{sample_name}_filtered.txt")
        mutation_file = os.path.join(output_dir, f"{sample_name}_mutations.txt")
        
        print(f"\nProcessing {sample_name}...")
        
        # Calculate depth threshold for this file
        depth_threshold = get_depth_threshold(mpileup_file)
        
        fixation_count = 0
        filtered_count = 0
        total_sites = 0
        depth_filtered = 0
        sites_written = 0
        ref_only_count = 0
        n_filtered = 0  # Counter for N-filtered sites
        ct_mutations = 0  # Counter for C>T mutations
        ga_mutations = 0  # Counter for G>A mutations
       
        with open(mpileup_file) as f_in, \
             open(output_file, 'w') as f_out, \
             open(mutation_file, 'w') as f_mut:
            
            
            # Process each line
            for line in f_in:
                total_sites += 1
                fields = line.strip().split('\t')
                if len(fields) < 6:
                    continue
                    
                # Check depth threshold
                depth = int(fields[3])
                if depth < depth_threshold:
                    depth_filtered += 1
                    continue
                    
                chrom = fields[0]
                pos = fields[1] 
                ref = fields[2]
                reads = fields[4]
                
                # Count bases and get most common alt in one pass
                base_counts, alt, alt_count = count_bases(reads, ref)
                
                # Filter if more Ns than alternate bases
                alt_base_count = sum(base_counts[b] for b in 'ACGT' if b != ref.upper())
                if base_counts['N'] > alt_base_count:
                    n_filtered += 1
                    continue
                
                if not alt or alt_count == 0:  # No alternate allele found
                    ref_only_count += 1
                    f_out.write(f"{chrom}\t{pos}\t{ref}\t{fields[3]}\t{reads}\t{fields[5]}\n")
                    continue
                
                # Track and save C>T and G>A mutations
                mutation_type = None
                if ref.upper() == 'C' and alt == 'T':
                    ct_mutations += 1
                    mutation_type = 'C>T'
                elif ref.upper() == 'G' and alt == 'A':
                    ga_mutations += 1
                    mutation_type = 'G>A'
                
                # Calculate bias score using KS test
                pvalue, statistic = analyze_read_position_bias(reads, alt)
                
                # Count fixation events
                if pvalue == -1 and statistic == -1:
                    fixation_count += 1
                    f_out.write(f"{chrom}\t{pos}\t{ref}\t{fields[3]}\t{reads}\t{fields[5]}\n")
                    if mutation_type:  # Save mutation if it's C>T or G>A
                        f_mut.write(f"{chrom}\t{pos}\t{ref}\t{fields[3]}\t{reads}\t{fields[5]}\t{mutation_type}\n")
                    continue
                
                # Write results if passes filters
                if statistic < 0.25 and pvalue > 0.01:
                    f_out.write(f"{chrom}\t{pos}\t{ref}\t{fields[3]}\t{reads}\t{fields[5]}\n")
                    if mutation_type:  # Save mutation if it's C>T or G>A
                        f_mut.write(f"{chrom}\t{pos}\t{ref}\t{fields[3]}\t{reads}\t{fields[5]}\t{mutation_type}\n")
                    sites_written += 1
                else:
                    filtered_count += 1
        
        print(f"Results for {sample_name}:")
        print(f"Total sites processed: {total_sites}")
        print(f"Sites filtered by depth (< {depth_threshold}): {depth_filtered}")
        print(f"Sites filtered due to excess N bases: {n_filtered}")
        print(f"Fixation events found: {fixation_count}")
        print(f"Sites filtered due to clustering: {filtered_count}")
        print(f"Sites written to output: {sites_written}")
        print(f"Sites with no alternate allele: {ref_only_count}")
        print(f"C>T mutations in output: {ct_mutations}")
        print(f"G>A mutations in output: {ga_mutations}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process mpileup files to detect sequence bias')
    parser.add_argument('mpileup_dir', help='Path to directory containing mpileup files')
    parser.add_argument('output_dir', help='Path to output directory')
    args = parser.parse_args()
    
    main(args.mpileup_dir, args.output_dir)
