#!/usr/bin/env python3
"""
Collect mutation counts from mpileup files with filtering.

This script synthesizes the simplified mpileup parsing approach with control filtering:
1. Depth percentile filtering (top and bottom 10% filtered)
2. Majority-ref filtering 
3. Remove sites that appear in >1 control file

Outputs mutation counts per site for EMS mutation rate estimation.
"""
import argparse
import numpy as np
import glob
import os
from collections import Counter
import multiprocessing as mp
from functools import partial
from scipy import stats


def parse_bases(bases, ref):
    """Parse mpileup bases string into uppercase bases, mapping . and , to ref."""
    i = 0
    L = len(bases)
    ref_up = ref.upper()
    out = []
    while i < L:
        c = bases[i]
        if c == '^':  # start of read
            i += 2
            continue
        if c == '$':  # end of read
            i += 1
            continue
        if c == '*':  # deletion
            i += 1
            continue
        if c in '+-':  # indel
            i += 1
            num_str = ''
            while i < L and bases[i].isdigit():
                num_str += bases[i]
                i += 1
            indel_len = int(num_str) if num_str else 0
            i += indel_len
            continue
        if c == '.' or c == ',':
            out.append(ref_up)
            i += 1
            continue
        if c.isalpha():
            out.append(c.upper())
        i += 1
    return out


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


def build_exclusion_mask_from_counts(control_count_files, min_alt=1):
    """Build exclusion mask from control .counts files (much faster than mpileup parsing)."""
    site_hits = Counter()
    total_files = len(control_count_files)
    
    for i, filename in enumerate(control_count_files):
        print(f"  Processing control counts file {i+1}/{total_files}: {os.path.basename(filename)}")
        line_count = 0
        with open(filename) as f:
            header = next(f, None)  # Skip header
            for line in f:
                line_count += 1
                if line_count % 100000 == 0:
                    print(f"    Processed {line_count:,} lines...")
                    
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                    
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = fields
                
                # Convert to int
                ref_count = int(ref_count)
                depth = int(depth)
                
                # Only consider G/C sites
                if ref not in {"G", "C"}:
                    continue
                
                # Calculate EMS mutations
                ems_count = depth - ref_count
                
                if ems_count >= min_alt:
                    site_hits[(chrom, pos)] += 1
        
        print(f"    Completed {os.path.basename(filename)}: {line_count:,} lines processed")
    
    excluded_sites = {site for site, count in site_hits.items() if count > 1}
    print(f"  Built exclusion mask: {len(excluded_sites)} sites to exclude")
    return excluded_sites


def save_exclusion_mask(excluded_sites, output_file):
    """Save exclusion mask to file."""
    with open(output_file, 'w') as f:
        f.write("chrom\tpos\n")
        for chrom, pos in sorted(excluded_sites):
            f.write(f"{chrom}\t{pos}\n")
    print(f"Exclusion mask saved to: {output_file}")


def load_exclusion_mask(mask_file):
    """Load exclusion mask from file."""
    excluded_sites = set()
    with open(mask_file) as f:
        header = next(f, None)  # Skip header
        for line in f:
            chrom, pos = line.strip().split('\t')
            excluded_sites.add((chrom, pos))
    return excluded_sites


def process_mpileup_file(infile, excluded_sites, low_pct=10, high_pct=90, apply_position_bias=False):
    """
    Process mpileup file and return filtered sites with mutation counts.
    
    Returns:
        List of tuples: (chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth)
    """
    sites = []
    
    # First pass: collect all sites
    with open(infile) as inf:
        for line in inf:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            chrom, pos, ref = fields[0], fields[1], fields[2].upper()
            bases = fields[4]

            parsed = parse_bases(bases, ref)
            depth = len(parsed)
            if depth == 0:
                continue

            counts = {b: parsed.count(b) for b in "ACGT"}
            ref_count = counts.get(ref, 0)

            # majority-ref filter
            nonref_count = depth - ref_count
            if nonref_count / depth > 0.5:
                continue

            sites.append((chrom, pos, ref,
                          ref_count,
                          counts["A"], counts["C"], counts["G"], counts["T"],
                          depth, bases))  # Include original bases string for bias analysis

    if not sites:
        return [], None, None

    # Second pass: compute depth percentiles and filter
    depths = [d for *_, d, _ in sites]
    low_thresh = np.percentile(depths, low_pct)
    high_thresh = np.percentile(depths, high_pct)

    # Apply filters (exclusion mask is optional)
    filtered = []
    for site in sites:
        chrom, pos, ref, ref_count, A, C, G, T, depth, bases = site
        
        # Depth percentile filter
        if not (low_thresh <= depth <= high_thresh):
            continue
            
        # Optional control exclusion filter
        if excluded_sites and (chrom, pos) in excluded_sites:
            continue
        
        # Position bias filter (if enabled)
        if apply_position_bias:
            # Find most common alt allele for bias analysis
            alt_counts = {b: counts[b] for b in "ACGT" if b != ref}
            if alt_counts:
                most_common_alt = max(alt_counts.items(), key=lambda x: x[1])[0]
                pvalue, statistic = analyze_read_position_bias(bases, most_common_alt)
                
                # Apply bias filter: pass if fixation event OR (low KS stat AND high p-value)
                passes_bias = (pvalue == -1 and statistic == -1) or (statistic < 0.25 and pvalue > 0.01)
                if not passes_bias:
                    continue
            
        filtered.append((chrom, pos, ref, ref_count, A, C, G, T, depth))
    
    return filtered, low_thresh, high_thresh


def process_single_file(args_tuple):
    """Process a single mpileup file. Used by multiprocessing."""
    infile, outfile, excluded_sites, low_pct, high_pct, apply_position_bias = args_tuple
    
    try:
        filtered_sites, low_thresh, high_thresh = process_mpileup_file(
            infile, excluded_sites, low_pct, high_pct, apply_position_bias
        )
        
        # Write output
        with open(outfile, "w") as outf:
            outf.write("chrom\tpos\tref\tref_count\tA_count\tC_count\tG_count\tT_count\tdepth\n")
            for chrom, pos, ref, ref_count, A, C, G, T, depth in filtered_sites:
                outf.write(f"{chrom}\t{pos}\t{ref}\t{ref_count}\t{A}\t{C}\t{G}\t{T}\t{depth}\n")
        
        return {
            'file': os.path.basename(infile),
            'success': True,
            'sites_kept': len(filtered_sites),
            'low_thresh': low_thresh,
            'high_thresh': high_thresh
        }
    except Exception as e:
        return {
            'file': os.path.basename(infile),
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Collect mutation counts from mpileup files with filtering (multiprocessing)"
    )
    parser.add_argument("--input-dir", required=True, 
                        help="Directory containing mpileup files")
    parser.add_argument("--output-dir", required=True, 
                        help="Output directory for .counts files")
    parser.add_argument("--control-pattern", type=str, default="*NT*.txt",
                        help="Pattern to match control files for exclusion mask generation (default: *NT*.txt)")
    parser.add_argument("--generate-exclusion-mask", action="store_true",
                        help="Generate exclusion mask from control files after processing")
    parser.add_argument("--exclusion-mask-file", type=str, default="exclusion_mask.tsv",
                        help="Output file for exclusion mask (default: exclusion_mask.tsv)")
    parser.add_argument("--min-alt", type=int, default=1,
                        help="Minimum alt allele count for exclusion mask generation (default 1)")
    parser.add_argument("--low", type=float, default=10.0,
                        help="Low percentile cutoff (default 10)")
    parser.add_argument("--high", type=float, default=90.0,
                        help="High percentile cutoff (default 90)")
    parser.add_argument("--n-processes", type=int, default=None,
                        help="Number of processes to use (default: number of CPUs)")
    parser.add_argument("--position-bias", action="store_true",
                        help="Apply position bias filtering using KS test (default: False)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all mpileup files in input directory
    mpileup_files = glob.glob(os.path.join(args.input_dir, "*_variants.txt"))
    if not mpileup_files:
        print(f"No .txt files found in {args.input_dir}")
        return
    
    print(f"Found {len(mpileup_files)} mpileup files")
    
    # Process all files without exclusion mask (treat controls and samples the same)
    excluded_sites = None  # No exclusion mask during processing
    
    # Prepare arguments for multiprocessing
    process_args = []
    for infile in mpileup_files:
        basename = os.path.basename(infile)
        outfile = os.path.join(args.output_dir, basename.replace(".txt", ".counts"))
        process_args.append((infile, outfile, excluded_sites, args.low, args.high, args.position_bias))
    
    # Process files in parallel
    if args.n_processes:
        n_processes = args.n_processes
    elif mp.cpu_count() > len(mpileup_files):
        n_processes = len(mpileup_files)
    else:
        n_processes = mp.cpu_count()
    print(f"Processing {len(mpileup_files)} files using {n_processes} processes")
    
    with mp.Pool(n_processes) as pool:
        results = pool.map(process_single_file, process_args)
    
    # Report results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Successfully processed: {len(successful)} files")
    print(f"Failed: {len(failed)} files")
    
    if successful:
        total_sites = sum(r['sites_kept'] for r in successful)
        print(f"Total sites kept across all files: {total_sites}")
    
    if failed:
        print(f"\nFailed files:")
        for result in failed:
            print(f"  {result['file']}: {result['error']}")
    
    print(f"\nOutput files written to: {args.output_dir}")
    
    # Generate exclusion mask if requested
    if args.generate_exclusion_mask:
        print(f"\n=== GENERATING EXCLUSION MASK ===")
        control_count_files = glob.glob(os.path.join(args.output_dir, args.control_pattern.replace(".txt", ".counts")))
        
        if control_count_files:
            print(f"Found {len(control_count_files)} control count files")
            excluded_sites = build_exclusion_mask_from_counts(control_count_files, min_alt=args.min_alt)
            mask_file = os.path.join(args.output_dir, args.exclusion_mask_file)
            save_exclusion_mask(excluded_sites, mask_file)
        else:
            print(f"No control count files found matching pattern: {args.control_pattern.replace('.txt', '.counts')}")


if __name__ == "__main__":
    main()
