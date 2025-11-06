#!/usr/bin/env python3
"""
Plot 5mer mutation rate analysis for strand-collapsed (C-centered only) mutations.

This script creates various plots for analyzing 5mer context mutation rates in EMS mutations,
focusing on normalized mutation rates rather than enrichment ratios. It also includes
positional base effects analysis to determine which nucleotide positions within 5mers
drive the observed mutation rates.
"""
import argparse
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import re
import gzip
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Try to import logomaker for sequence logos
try:
    import logomaker
    LOGOMAKER_AVAILABLE = True
except ImportError:
    LOGOMAKER_AVAILABLE = False
    print("Warning: logomaker not available. Sequence logo generation will be skipped.")
    print("Install with: pip install logomaker")

# Global color palette configuration
# Change these to test different color schemes from plot_colors.json
# Options:
#   - palette_type: 'qualitative', 'diverging', 'sequential'
#   - palette_name: 
#       * qualitative: '6a', '5a', '4a', '4b', '3a', '3b', '2a', '2b'
#       * diverging: 'd1', 'd2', 'd3', 'd4'
#       * sequential: 's1', 's2', 's3', 'm1', 'm2', 'm3'
#       * viridis: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
COLOR_PALETTE_TYPE = 'viridis'
COLOR_PALETTE_NAME = 'plasma'  # Change to 'd2', 'd3', 'plasma', 'viridis', etc.


def load_json(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_sample_id(filename):
    """Extract sample ID from filename.
    
    Handles patterns like:
    - sample_variants_5mer_contexts.json -> sample_variants
    - sample_5mer_contexts.json -> sample
    """
    basename = os.path.basename(filename)
    # Remove _5mer_contexts.json extension
    sample_id = basename.replace('_5mer_contexts.json', '')
    # Also handle _variants suffix if present (but keep it for now as it might be needed)
    return sample_id


def load_exclusion_mask_tsv(mask_file):
    """Load exclusion mask from TSV file (chrom\tpos format).
    
    Returns set of (chrom, pos) tuples to exclude.
    """
    excluded = set()
    if not mask_file or not os.path.exists(mask_file):
        return excluded
    
    try:
        with open(mask_file, 'r') as f:
            header = next(f, None)  # Skip header
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                excluded.add((parts[0], parts[1]))
    except Exception as e:
        print(f"Warning: Error loading exclusion mask from {mask_file}: {e}")
    
    return excluded


def build_exclusion_mask_from_controls(counts_dir, min_alt=1):
    """Build exclusion mask from control .counts files.
    
    Excludes sites that appear in more than one control (with at least min_alt mutations).
    Returns set of (chrom, pos) tuples to exclude.
    """
    from collections import Counter
    
    site_hits = Counter()
    control_files = glob.glob(os.path.join(counts_dir, "*NT*.counts"))
    control_files.extend(glob.glob(os.path.join(counts_dir, "*CONTROL*.counts")))
    control_files.extend(glob.glob(os.path.join(counts_dir, "*control*.counts")))
    
    if not control_files:
        print("Warning: No control files found for building exclusion mask")
        return set()
    
    print(f"Building exclusion mask from {len(control_files)} control files...")
    
    for control_file in control_files:
        try:
            with open(control_file, 'r') as f:
                header = next(f, None)  # Skip header
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 9:
                        continue
                    
                    chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = parts[:9]
                    
                    # Only consider G/C sites
                    if ref not in {"G", "C"}:
                        continue
                    
                    try:
                        ref_count = int(ref_count)
                        depth = int(depth)
                    except (ValueError, TypeError):
                        continue
                    
                    # Calculate EMS mutations (for C->T or G->A)
                    if ref == 'C':
                        ems_count = int(T_count) if T_count.isdigit() else 0
                    else:  # ref == 'G'
                        ems_count = int(A_count) if A_count.isdigit() else 0
                    
                    if ems_count >= min_alt:
                        site_hits[(chrom, pos)] += 1
        except Exception as e:
            print(f"Warning: Error processing control file {control_file}: {e}")
            continue
    
    # Exclude sites that appear in more than one control
    excluded_sites = {site for site, count in site_hits.items() if count > 1}
    print(f"  Found {len(excluded_sites)} sites to exclude (appear in multiple controls)")
    
    return excluded_sites




def load_sample_depths(counts_dir, exclusion_mask=None, min_alt=0):
    """Load total depth per sample from .counts files, excluding masked sites and sites below min_alt.
    
    Args:
        counts_dir: Directory containing .counts files
        exclusion_mask: Optional set of (chrom, pos) tuples to exclude
        min_alt: Minimum alt allele count required for a site to be included (default: 0, no filtering)
    
    Returns dict mapping sample_id -> total_depth (sum of all site depths, excluding masked sites and sites below min_alt)
    """
    sample_depths = {}
    count_files = glob.glob(os.path.join(counts_dir, "*.counts"))
    
    if not count_files:
        print(f"Warning: No .counts files found in {counts_dir}")
        return sample_depths
    
    print(f"Loading depth information from {len(count_files)} .counts files...")
    if exclusion_mask:
        print(f"  Applying exclusion mask: {len(exclusion_mask)} sites excluded")
    if min_alt > 0:
        print(f"  Filtering sites: requiring min_alt >= {min_alt}")
    
    for count_file in count_files:
        sample_name = os.path.basename(count_file).replace('.counts', '')
        total_depth = 0
        excluded_count = 0
        min_alt_filtered = 0
        
        try:
            with open(count_file, 'r') as f:
                header = next(f, None)  # Skip header
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 9:
                        continue
                    
                    chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count = parts[:8]
                    
                    # Apply exclusion mask
                    if exclusion_mask and (chrom, pos) in exclusion_mask:
                        excluded_count += 1
                        continue
                    
                    # Filter by min_alt for C->T or G->A mutations
                    if min_alt > 0 and ref in {"G", "C"}:
                        if ref == 'C':
                            ems_count = int(T_count) if T_count.isdigit() else 0
                        else:  # ref == 'G'
                            ems_count = int(A_count) if A_count.isdigit() else 0
                        
                        if ems_count < min_alt:
                            min_alt_filtered += 1
                            continue
                    
                    try:
                        depth = int(parts[8])  # depth is 9th column (0-indexed: 8)
                        if depth > 0:
                            total_depth += depth
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"Warning: Error reading depth from {count_file}: {e}")
            continue
        
        if total_depth > 0:
            sample_depths[sample_name] = total_depth
            filter_info = []
            if excluded_count > 0:
                filter_info.append(f"{excluded_count:,} sites excluded by mask")
            if min_alt_filtered > 0:
                filter_info.append(f"{min_alt_filtered:,} sites filtered by min_alt")
            if filter_info:
                print(f"  {sample_name}: total depth = {total_depth:,} ({', '.join(filter_info)})")
            else:
                print(f"  {sample_name}: total depth = {total_depth:,}")
    
    return sample_depths


def is_control_sample(sample_id):
    """Check if sample is a control (NT) sample."""
    return 'NT' in sample_id


def calculate_aggregate_depth_normalized_rates(context_counts, sample_depths, use_collapsed=True):
    """Calculate aggregate depth-normalized mutation rates with background subtraction.
    
    Args:
        context_counts: dict mapping sample_id -> kmer_counts dict
        sample_depths: dict mapping sample_id -> total_depth
        use_collapsed: if True, reverse complement G-centered kmers to C-centered
    
    Returns:
        dict mapping kmer -> depth-normalized rate (treated - control)
    """
    # Separate treated and control samples
    treated_samples = []
    control_samples = []
    
    for sample_id in context_counts.keys():
        if is_control_sample(sample_id):
            control_samples.append(sample_id)
        else:
            treated_samples.append(sample_id)
    
    print(f"\nCalculating aggregate depth-normalized rates:")
    print(f"  Treated samples: {len(treated_samples)}")
    print(f"  Control samples: {len(control_samples)}")
    
    if not treated_samples:
        print("Warning: No treated samples found!")
        return {}
    
    # Aggregate mutation counts and depths per kmer for treated samples
    treated_kmer_counts = defaultdict(int)
    treated_total_depth = 0
    
    for sample_id in treated_samples:
        kmer_counts = context_counts[sample_id]
        depth = sample_depths.get(sample_id, 0)
        
        if depth <= 0:
            print(f"Warning: Sample {sample_id} has no depth information, skipping")
            continue
        
        treated_total_depth += depth
        
        for kmer, count in kmer_counts.items():
            if len(kmer) == 5:
                if use_collapsed:
                    # Collapse G-centered to C-centered
                    if kmer[2] == 'G':
                        kmer = reverse_complement(kmer)
                    if kmer[2] == 'C':
                        treated_kmer_counts[kmer] += count
                else:
                    # Use as-is
                    treated_kmer_counts[kmer] += count
    
    # Aggregate mutation counts and depths per kmer for control samples
    control_kmer_counts = defaultdict(int)
    control_total_depth = 0
    
    for sample_id in control_samples:
        kmer_counts = context_counts[sample_id]
        depth = sample_depths.get(sample_id, 0)
        
        if depth <= 0:
            print(f"Warning: Sample {sample_id} has no depth information, skipping")
            continue
        
        control_total_depth += depth
        
        for kmer, count in kmer_counts.items():
            if len(kmer) == 5:
                if use_collapsed:
                    # Collapse G-centered to C-centered
                    if kmer[2] == 'G':
                        kmer = reverse_complement(kmer)
                    if kmer[2] == 'C':
                        control_kmer_counts[kmer] += count
                else:
                    # Use as-is
                    control_kmer_counts[kmer] += count
    
    print(f"  Treated total depth: {treated_total_depth:,}")
    print(f"  Control total depth: {control_total_depth:,}")
    
    # Calculate depth-normalized rates (mutations per base sequenced)
    treated_rates = {}
    control_rates = {}
    
    all_kmers = set(treated_kmer_counts.keys()) | set(control_kmer_counts.keys())
    
    for kmer in all_kmers:
        # Treated rate: mutations / depth
        treated_count = treated_kmer_counts.get(kmer, 0)
        if treated_total_depth > 0:
            treated_rates[kmer] = treated_count / treated_total_depth
        else:
            treated_rates[kmer] = 0.0
        
        # Control rate: mutations / depth
        control_count = control_kmer_counts.get(kmer, 0)
        if control_total_depth > 0:
            control_rates[kmer] = control_count / control_total_depth
        else:
            control_rates[kmer] = 0.0
    
    # Subtract background (control) rates from treated rates
    ems_rates = {}
    for kmer in all_kmers:
        ems_rates[kmer] = treated_rates[kmer] - control_rates[kmer]
    
    print(f"  Calculated rates for {len(ems_rates)} kmers")
    print(f"  Rate range: {min(ems_rates.values()):.2e} to {max(ems_rates.values()):.2e}")
    
    return ems_rates


def calculate_aggregate_depth_and_genome_normalized_rates(context_counts, sample_depths, genome_counts, use_collapsed=True):
    """Calculate aggregate depth-normalized and genome-frequency-normalized mutation rates with background subtraction.
    
    Args:
        context_counts: dict mapping sample_id -> kmer_counts dict
        sample_depths: dict mapping sample_id -> total_depth
        genome_counts: dict mapping kmer -> genome_count (for genomic frequency normalization)
        use_collapsed: if True, reverse complement G-centered kmers to C-centered
    
    Returns:
        dict mapping kmer -> depth-normalized and genome-frequency-normalized rate (treated - control)
    """
    # Separate treated and control samples
    treated_samples = []
    control_samples = []
    
    for sample_id in context_counts.keys():
        if is_control_sample(sample_id):
            control_samples.append(sample_id)
        else:
            treated_samples.append(sample_id)
    
    print(f"\nCalculating aggregate depth and genome-frequency-normalized rates:")
    print(f"  Treated samples: {len(treated_samples)}")
    print(f"  Control samples: {len(control_samples)}")
    
    if not treated_samples:
        print("Warning: No treated samples found!")
        return {}
    
    # Aggregate mutation counts and depths per kmer for treated samples
    treated_kmer_counts = defaultdict(int)
    treated_total_depth = 0
    
    for sample_id in treated_samples:
        kmer_counts = context_counts[sample_id]
        depth = sample_depths.get(sample_id, 0)
        
        if depth <= 0:
            print(f"Warning: Sample {sample_id} has no depth information, skipping")
            continue
        
        treated_total_depth += depth
        
        for kmer, count in kmer_counts.items():
            if len(kmer) == 5:
                if use_collapsed:
                    # Collapse G-centered to C-centered
                    if kmer[2] == 'G':
                        kmer = reverse_complement(kmer)
                    if kmer[2] == 'C':
                        treated_kmer_counts[kmer] += count
                else:
                    # Use as-is
                    treated_kmer_counts[kmer] += count
    
    # Aggregate mutation counts and depths per kmer for control samples
    control_kmer_counts = defaultdict(int)
    control_total_depth = 0
    
    for sample_id in control_samples:
        kmer_counts = context_counts[sample_id]
        depth = sample_depths.get(sample_id, 0)
        
        if depth <= 0:
            print(f"Warning: Sample {sample_id} has no depth information, skipping")
            continue
        
        control_total_depth += depth
        
        for kmer, count in kmer_counts.items():
            if len(kmer) == 5:
                if use_collapsed:
                    # Collapse G-centered to C-centered
                    if kmer[2] == 'G':
                        kmer = reverse_complement(kmer)
                    if kmer[2] == 'C':
                        control_kmer_counts[kmer] += count
                else:
                    # Use as-is
                    control_kmer_counts[kmer] += count
    
    print(f"  Treated total depth: {treated_total_depth:,}")
    print(f"  Control total depth: {control_total_depth:,}")
    
    # Collapse genome counts if needed
    collapsed_genome_counts = {}
    for kmer, count in genome_counts.items():
        if len(kmer) == 5:
            if use_collapsed:
                if kmer[2] == 'G':
                    kmer = reverse_complement(kmer)
                if kmer[2] == 'C':
                    collapsed_genome_counts[kmer] = collapsed_genome_counts.get(kmer, 0) + count
            else:
                collapsed_genome_counts[kmer] = collapsed_genome_counts.get(kmer, 0) + count
    
    total_genome = sum(collapsed_genome_counts.values())
    print(f"  Total genome kmers: {total_genome:,}")
    
    # Calculate depth-normalized and genome-frequency-normalized rates
    treated_rates = {}
    control_rates = {}
    
    all_kmers = set(treated_kmer_counts.keys()) | set(control_kmer_counts.keys())
    
    for kmer in all_kmers:
        treated_count = treated_kmer_counts.get(kmer, 0)
        control_count = control_kmer_counts.get(kmer, 0)
        
        # Get genomic frequency
        genome_count = collapsed_genome_counts.get(kmer, 0)
        if genome_count > 0 and total_genome > 0:
            genome_freq = genome_count / total_genome
        else:
            genome_freq = 0.0
        
        # Treated rate: mutations / (genome_freq * depth)
        if treated_total_depth > 0 and genome_freq > 0:
            expected_treated = genome_freq * treated_total_depth
            treated_rates[kmer] = treated_count / expected_treated if expected_treated > 0 else 0.0
        else:
            treated_rates[kmer] = 0.0
        
        # Control rate: mutations / (genome_freq * depth)
        if control_total_depth > 0 and genome_freq > 0:
            expected_control = genome_freq * control_total_depth
            control_rates[kmer] = control_count / expected_control if expected_control > 0 else 0.0
        else:
            control_rates[kmer] = 0.0
    
    # Subtract background (control) rates from treated rates
    ems_rates = {}
    for kmer in all_kmers:
        ems_rates[kmer] = treated_rates[kmer] - control_rates[kmer]
    
    print(f"  Calculated rates for {len(ems_rates)} kmers")
    print(f"  Rate range: {min(ems_rates.values()):.2e} to {max(ems_rates.values()):.2e}")
    
    return ems_rates


def load_5mer_contexts(kmers_dir, use_gene_only=False, no_strand_collapse=False):
    """Load 5mer context counts from all 5mer context JSON files in directory."""
    context_counts = {}
    kmer_files = glob.glob(os.path.join(kmers_dir, "*_5mer_contexts.json"))
    
    print(f"Found {len(kmer_files)} 5mer context files")
    
    for kmer_file in kmer_files:
        sample_id = extract_sample_id(kmer_file)
        try:
            data = load_json(kmer_file)
            
            if no_strand_collapse:
                # Handle non-strand-collapsed format with c_centered and g_centered
                if isinstance(data, dict) and 'c_centered' in data:
                    if use_gene_only:
                        c_counts = data['c_centered'].get('gene', {})
                        g_counts = data['g_centered'].get('gene', {})
                    else:
                        c_counts = data['c_centered'].get('total', {})
                        g_counts = data['g_centered'].get('total', {})
                    
                    # Combine C and G centered into a single dictionary for compatibility
                    combined_counts = {}
                    combined_counts.update(c_counts)
                    combined_counts.update(g_counts)
                    
                    context_counts[sample_id] = combined_counts
                    print(f"Loaded {len(c_counts)} C-centered and {len(g_counts)} G-centered 5mers for sample {sample_id}")
                else:
                    # Fallback: treat strand-collapsed data as C-centered only
                    print(f"Warning: Expected non-strand-collapsed format for {sample_id}, treating as strand-collapsed data")
                    if isinstance(data, dict) and 'total' in data:
                        if use_gene_only:
                            kmer_counts = data.get('gene', {})
                        else:
                            kmer_counts = data.get('total', {})
                    else:
                        kmer_counts = data
                    
                    context_counts[sample_id] = kmer_counts
                    print(f"Loaded {len(kmer_counts)} C-centered 5mers for sample {sample_id} (strand-collapsed data)")
            else:
                # Handle strand-collapsed format
                if isinstance(data, dict) and 'total' in data:
                    # New format with gene/intergenic separation
                    if use_gene_only:
                        kmer_counts = data.get('gene', {})
                        print(f"Loaded {len(kmer_counts)} gene-only 5mers for sample {sample_id}")
                    else:
                        kmer_counts = data.get('total', {})
                        print(f"Loaded {len(kmer_counts)} total 5mers for sample {sample_id}")
                else:
                    # Old format - just use the data directly
                    kmer_counts = data
                    print(f"Loaded {len(kmer_counts)} 5mers for sample {sample_id} (old format)")
                
            context_counts[sample_id] = kmer_counts
        except Exception as e:
            print(f"Error loading {kmer_file}: {e}")
    
    return context_counts


def generate_genome_5mer_counts(genome_fasta, kmer_size=5):
    """Generate 5mer counts from genome sequence (supports both regular and gzipped files)."""
    genome_kmers = defaultdict(int)
    
    # Check if file is gzipped
    if genome_fasta.endswith('.gz'):
        with gzip.open(genome_fasta, 'rt') as f:
            genome_seq = ""
            for line in f:
                if not line.startswith('>'):
                    genome_seq += line.strip().upper()
    else:
        with open(genome_fasta, 'r') as f:
            genome_seq = ""
            for line in f:
                if not line.startswith('>'):
                    genome_seq += line.strip().upper()
    
    # Count all 5mers in genome
    for i in range(len(genome_seq) - kmer_size + 1):
        kmer = genome_seq[i:i + kmer_size]
        if 'N' not in kmer:  # Skip kmers with N
            genome_kmers[kmer] += 1
    
    return dict(genome_kmers)


def generate_genome_5mer_counts_separate(genome_fasta, kmer_size=5):
    """Generate separate C-centered and G-centered 5mer counts from genome sequence."""
    c_centered_kmers = defaultdict(int)
    g_centered_kmers = defaultdict(int)
    
    # Check if file is gzipped
    if genome_fasta.endswith('.gz'):
        with gzip.open(genome_fasta, 'rt') as f:
            genome_seq = ""
            for line in f:
                if not line.startswith('>'):
                    genome_seq += line.strip().upper()
    else:
        with open(genome_fasta, 'r') as f:
            genome_seq = ""
            for line in f:
                if not line.startswith('>'):
                    genome_seq += line.strip().upper()
    
    # Count C-centered and G-centered 5mers separately
    for i in range(len(genome_seq) - kmer_size + 1):
        kmer = genome_seq[i:i + kmer_size]
        if 'N' not in kmer:  # Skip kmers with N
            if kmer[2] == 'C':
                c_centered_kmers[kmer] += 1
            elif kmer[2] == 'G':
                g_centered_kmers[kmer] += 1
    
    return {
        'c_centered': dict(c_centered_kmers),
        'g_centered': dict(g_centered_kmers)
    }


def filter_centered_kmers(kmer_counts, center_base='C'):
    """Filter to only 5mers centered on the specified base (default 'C')."""
    centered = {}
    for kmer, count in kmer_counts.items():
        if len(kmer) == 5 and kmer[2] == center_base:
            centered[kmer] = count
    return centered


def calculate_mutation_rates(sample_counts, genome_counts):
    """Calculate normalized mutation rates for sample 5mer counts."""
    mutation_rates = {}
    total_sample = sum(sample_counts.values())
    total_genome = sum(genome_counts.values())
    
    for kmer, observed_count in sample_counts.items():
        if kmer in genome_counts and genome_counts[kmer] > 0 and total_sample > 0:
            # Normalize by genome frequency to get mutation rate per kmer occurrence
            genome_freq = genome_counts[kmer] / total_genome
            mutation_rate = observed_count / (genome_freq * total_sample) if genome_freq > 0 else 0
            mutation_rates[kmer] = mutation_rate
        else:
            mutation_rates[kmer] = 0
    
    return mutation_rates


def calculate_normalized_mutation_rates(sample_counts, genome_counts):
    """Calculate mutation rates normalized by total mutations per sample."""
    mutation_rates = {}
    total_sample = sum(sample_counts.values())
    
    for kmer, observed_count in sample_counts.items():
        if total_sample > 0:
            # Simple normalization: mutations per kmer / total mutations
            mutation_rate = observed_count / total_sample
            mutation_rates[kmer] = mutation_rate
        else:
            mutation_rates[kmer] = 0
    
    return mutation_rates


def reverse_complement(seq):
    """Get reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def clean_sample_label(sample_name: str) -> str:
    """Produce a cleaned display label from a raw sample file stem.
    
    Supported patterns (any prefix/suffix around tokens):
      *_NT#_#d_*   -> NT#        (drop day)
      *_EMS-#_*    -> EMS-#
      *_EMS#_#d_*  -> EMS#_#d
    If no pattern matches, return the original name.
    """
    name = sample_name
    # *_NT#_#d_* -> NT#
    m = re.search(r'(?:^|_)NT(\d+)_\d+d(?:_|$)', name, flags=re.IGNORECASE)
    if m:
        return f"NT{m.group(1)}"
    # *_NT-#_* -> NT-#
    m = re.search(r'(?:^|_)NT-(\d+)(?:_|$)', name, flags=re.IGNORECASE)
    if m:
        return f"NT-{m.group(1)}"
    # *_EMS-#_* -> EMS-#
    m = re.search(r'(?:^|_)EMS-(\d+)(?:_|$)', name, flags=re.IGNORECASE)
    if m:
        return f"EMS-{m.group(1)}"
    # *_EMS#_#d_* -> EMS#_#d
    m = re.search(r'(?:^|_)EMS(\d+)_(\d+)d(?:_|$)', name, flags=re.IGNORECASE)
    if m:
        return f"EMS{m.group(1)}_{m.group(2)}d"
    return sample_name


def extract_ems_number(sample_name):
    """Extract EMS number from sample name for sorting.
    
    Examples:
        EMS-1 -> 1
        EMS-10 -> 10
        EMS1_14d -> 1
        sample_EMS-5_rep1 -> 5
    
    Returns the numeric value if found, otherwise returns float('inf') to sort to end.
    """
    # Try EMS-# pattern
    m = re.search(r'EMS-?(\d+)', sample_name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return float('inf')  # Put non-EMS samples at the end


def sort_samples_by_ems_number(samples):
    """Sort sample list by EMS number."""
    return sorted(samples, key=extract_ems_number)


def analyze_positional_effects_from_rates(context_counts, use_corrected_p=True):
    """
    Analyze which bases at which positions drive 5mer mutation rates.
    This function adapts the positional analysis from positional_5mer_analysis.py
    to work with mutation rate data instead of treatment effect coefficients.
    """
    print("Analyzing positional base effects from mutation rates...")
    
    # Initialize storage for each position and base
    position_effects = {pos: {'A': [], 'T': [], 'G': [], 'C': []} for pos in range(5)}
    
    # Get EMS samples (excluding 3d/7d)
    ems_samples = [s for s in context_counts.keys() if 'EMS' in s and '3d' not in s and '7d' not in s]
    if not ems_samples:
        print("No EMS samples found for positional analysis")
        return None, None
    
    # Calculate mutation rates for each 5mer across all samples
    all_5mers = set()
    for sample_counts in context_counts.values():
        all_5mers.update(k for k in sample_counts.keys() if len(k) == 5)
    
    # Calculate average mutation rates across samples
    kmer_rates = {}
    for kmer in all_5mers:
        rates = []
        for sample in ems_samples:
            sample_counts = context_counts[sample]
            total_sample = sum(sample_counts.values())
            if total_sample > 0:
                rate = sample_counts.get(kmer, 0) / total_sample
                rates.append(rate)
        kmer_rates[kmer] = {
            'mean_rate': np.mean(rates) if rates else 0,
            'std_rate': np.std(rates) if rates else 0,
            'rates': rates
        }
    
    # Group effects by position and base
    for kmer, rate_data in kmer_rates.items():
        if len(kmer) == 5:  # Ensure it's a proper 5mer
            mean_rate = rate_data['mean_rate']
            std_rate = rate_data['std_rate']
            
            for pos in range(5):
                base = kmer[pos]
                if base in position_effects[pos]:
                    position_effects[pos][base].append({
                        'kmer': kmer,
                        'mean_rate': mean_rate,
                        'std_rate': std_rate,
                        'rates': rate_data['rates']
                    })
    
    # Calculate summary statistics for each position and base
    position_summary = {}
    for pos in range(5):
        position_summary[pos] = {}
        print(f"\nPosition {pos} analysis:")
        
        for base in ['A', 'T', 'G', 'C']:
            effects = position_effects[pos][base]
            if effects:
                mean_rates = [e['mean_rate'] for e in effects]
                std_rates = [e['std_rate'] for e in effects]
                all_rates = [rate for e in effects for rate in e['rates']]
                
                # Calculate statistics
                stats_dict = {
                    'count': len(effects),
                    'mean_rate': np.mean(mean_rates),
                    'median_rate': np.median(mean_rates),
                    'std_rate': np.std(mean_rates),
                    'min_rate': np.min(mean_rates),
                    'max_rate': np.max(mean_rates),
                    'mean_std': np.mean(std_rates),
                    'high_rate_count': sum(1 for r in mean_rates if r > np.percentile(mean_rates, 90)),
                    'low_rate_count': sum(1 for r in mean_rates if r < np.percentile(mean_rates, 10)),
                    'mean_rates': mean_rates,
                    'all_rates': all_rates
                }
                
                position_summary[pos][base] = stats_dict
                
                print(f"  {base}: n={stats_dict['count']:3d}, "
                      f"mean_rate={stats_dict['mean_rate']:.6f}, "
                      f"high_rate={stats_dict['high_rate_count']:2d}, "
                      f"low_rate={stats_dict['low_rate_count']:2d}")
            else:
                position_summary[pos][base] = None
    
    return position_summary, position_effects


def test_positional_significance_from_rates(position_summary):
    """
    Test if certain positions or bases have significantly different mutation rates.
    """
    print("\nTesting positional significance from mutation rates...")
    
    significance_tests = {}
    
    # Test each position: are the mean rates significantly different between bases?
    for pos in range(5):
        base_rates = {}
        for base in ['A', 'T', 'G', 'C']:
            if position_summary[pos][base] is not None:
                base_rates[base] = position_summary[pos][base]['all_rates']
        
        if len(base_rates) >= 2:  # Need at least 2 groups to test
            # Kruskal-Wallis test (non-parametric ANOVA)
            groups = list(base_rates.values())
            try:
                h_stat, p_value = stats.kruskal(*groups)
                significance_tests[f'position_{pos}_bases'] = {
                    'test': 'kruskal_wallis',
                    'statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'description': f'Position {pos}: difference between bases'
                }
                print(f"Position {pos} base differences: H={h_stat:.3f}, p={p_value:.4e}")
            except ValueError:
                # May fail if all values are identical
                pass
    
    # Test purine vs pyrimidine at each position
    for pos in range(5):
        purines = []  # A, G
        pyrimidines = []  # C, T
        
        for base in ['A', 'G']:
            if position_summary[pos][base] is not None:
                purines.extend(position_summary[pos][base]['all_rates'])
        
        for base in ['C', 'T']:
            if position_summary[pos][base] is not None:
                pyrimidines.extend(position_summary[pos][base]['all_rates'])
        
        if len(purines) > 0 and len(pyrimidines) > 0:
            try:
                u_stat, p_value = stats.mannwhitneyu(purines, pyrimidines, alternative='two-sided')
                significance_tests[f'position_{pos}_purine_vs_pyrimidine'] = {
                    'test': 'mann_whitney',
                    'statistic': u_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'purine_mean': np.mean(purines),
                    'pyrimidine_mean': np.mean(pyrimidines),
                    'description': f'Position {pos}: purine vs pyrimidine'
                }
                print(f"Position {pos} purine vs pyrimidine: U={u_stat:.1f}, p={p_value:.4e}, "
                      f"pur_mean={np.mean(purines):.6f}, pyr_mean={np.mean(pyrimidines):.6f}")
            except ValueError:
                pass
    
    return significance_tests


def create_positional_plots_from_rates(position_summary, output_dir):
    """
    Create visualization plots for positional effects from mutation rates.
    """
    print("Creating positional effect plots from mutation rates...")
    
    # Prepare data for plotting
    plot_data = []
    for pos in range(5):
        for base in ['A', 'T', 'G', 'C']:
            if position_summary[pos][base] is not None:
                stats = position_summary[pos][base]
                plot_data.append({
                    'Position': pos,
                    'Base': base,
                    'Mean_Rate': stats['mean_rate'],
                    'Count': stats['count'],
                    'High_Rate_Count': stats['high_rate_count'],
                    'Low_Rate_Count': stats['low_rate_count']
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('5mer Positional Base Effects Analysis (Mutation Rates)', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean mutation rate by position and base
    ax1 = axes[0, 0]
    pivot_rate = df_plot.pivot(index='Position', columns='Base', values='Mean_Rate')
    sns.heatmap(pivot_rate, annot=True, fmt='.6f', cmap='Reds', 
                cbar_kws={'label': 'Mean Mutation Rate'}, ax=ax1)
    ax1.set_title('Mean Mutation Rate by Position and Base')
    ax1.set_xlabel('Base')
    ax1.set_ylabel('5mer Position')
    
    # Plot 2: Count of 5mers by position and base
    ax2 = axes[0, 1]
    pivot_count = df_plot.pivot(index='Position', columns='Base', values='Count')
    sns.heatmap(pivot_count, annot=True, fmt='.0f', cmap='Blues',
                cbar_kws={'label': 'Number of 5mers'}, ax=ax2)
    ax2.set_title('Number of 5mers by Position and Base')
    ax2.set_xlabel('Base')
    ax2.set_ylabel('5mer Position')
    
    # Plot 3: High rate count
    ax3 = axes[1, 0]
    pivot_high = df_plot.pivot(index='Position', columns='Base', values='High_Rate_Count')
    sns.heatmap(pivot_high, annot=True, fmt='.0f', cmap='Oranges',
                cbar_kws={'label': 'High Rate Count'}, ax=ax3)
    ax3.set_title('High Rate 5mers (90th percentile)')
    ax3.set_xlabel('Base')
    ax3.set_ylabel('5mer Position')
    
    # Plot 4: Bar plot showing high and low rate counts
    ax4 = axes[1, 1]
    x = np.arange(len(df_plot))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, df_plot['High_Rate_Count'], width, 
                    label='High Rate (>90th percentile)', alpha=0.7, color='red')
    bars2 = ax4.bar(x + width/2, df_plot['Low_Rate_Count'], width,
                    label='Low Rate (<10th percentile)', alpha=0.7, color='blue')
    
    ax4.set_xlabel('Position-Base')
    ax4.set_ylabel('Count')
    ax4.set_title('High vs Low Rate Counts')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{row['Position']}-{row['Base']}" for _, row in df_plot.iterrows()], 
                       rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'positional_analysis_mutation_rates.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'positional_analysis_mutation_rates.pdf'), bbox_inches='tight')
    print(f"Positional plots saved: {output_dir}/positional_analysis_mutation_rates.png/pdf")
    plt.close()
    
    # Create individual rate distribution plots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    fig.suptitle('Mutation Rate Distributions by Position', fontsize=16, fontweight='bold')
    
    for pos in range(5):
        ax = axes[pos]
        
        for base in ['A', 'T', 'G', 'C']:
            if position_summary[pos][base] is not None:
                rates = position_summary[pos][base]['all_rates']
                ax.hist(rates, alpha=0.6, label=f'{base} (n={len(rates)})', 
                       bins=20, density=True)
        
        ax.set_xlabel('Mutation Rate')
        ax.set_title(f'Position {pos}')
        ax.legend()
        if pos == 0:
            ax.set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mutation_rate_distributions_by_position.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'mutation_rate_distributions_by_position.pdf'), bbox_inches='tight')
    print(f"Distribution plots saved: {output_dir}/mutation_rate_distributions_by_position.png/pdf")
    plt.close()


def write_positional_report_from_rates(position_summary, significance_tests, kmer_rates, output_file):
    """
    Write a detailed text report of the positional analysis from mutation rates.
    """
    print(f"Writing positional analysis report to: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("=== 5mer Positional Base Effects Analysis (Mutation Rates) ===\n\n")
        f.write(f"Total 5mers analyzed: {len(kmer_rates)}\n\n")
        
        # Overall summary
        f.write("=== Summary by Position ===\n\n")
        f.write("Position | Base | Count | Mean_Rate | Median_Rate | Std_Rate | High_Rate | Low_Rate\n")
        f.write("-" * 100 + "\n")
        
        for pos in range(5):
            for base in ['A', 'T', 'G', 'C']:
                if position_summary[pos][base] is not None:
                    stats = position_summary[pos][base]
                    f.write(f"{pos:8d} | {base:4s} | {stats['count']:5d} | "
                           f"{stats['mean_rate']:9.6f} | {stats['median_rate']:11.6f} | "
                           f"{stats['std_rate']:8.6f} | {stats['high_rate_count']:10d} | "
                           f"{stats['low_rate_count']:9d}\n")
                else:
                    f.write(f"{pos:8d} | {base:4s} | {'0':5s} | {'N/A':>9s} | {'N/A':>11s} | "
                           f"{'N/A':>8s} | {'0':>10s} | {'0':>9s}\n")
        
        # Statistical tests
        f.write(f"\n=== Statistical Tests ===\n\n")
        for test_name, result in significance_tests.items():
            f.write(f"{result['description']}:\n")
            f.write(f"  Test: {result['test']}\n")
            f.write(f"  Statistic: {result['statistic']:.4f}\n")
            f.write(f"  P-value: {result['p_value']:.4e}\n")
            f.write(f"  Significant: {result['significant']}\n")
            
            if 'purine_mean' in result:
                f.write(f"  Purine mean: {result['purine_mean']:.6f}\n")
                f.write(f"  Pyrimidine mean: {result['pyrimidine_mean']:.6f}\n")
            f.write("\n")
        
        # Top rates by position
        f.write("=== Top 5 Highest Mutation Rates by Position ===\n\n")
        for pos in range(5):
            f.write(f"Position {pos}:\n")
            
            # Collect all 5mers for this position with their rates
            pos_rates = []
            for kmer, rate_data in kmer_rates.items():
                if len(kmer) == 5:  # Valid 5mer
                    base_at_pos = kmer[pos]
                    pos_rates.append({
                        'kmer': kmer,
                        'base': base_at_pos,
                        'mean_rate': rate_data['mean_rate'],
                        'std_rate': rate_data['std_rate']
                    })
            
            # Sort by mean rate
            pos_rates.sort(key=lambda x: x['mean_rate'], reverse=True)
            
            for i, rate_info in enumerate(pos_rates[:5]):
                f.write(f"  {i+1}. {rate_info['kmer']} (pos {pos} = {rate_info['base']}): "
                       f"rate={rate_info['mean_rate']:.6f} ± {rate_info['std_rate']:.6f}\n")
            f.write("\n")


def get_colorblind_friendly_palette(n_colors, palette_type='qualitative', palette_name='6a'):
    """Get a high-quality color palette from plot_colors.json or matplotlib colormaps."""
    # Handle Viridis palettes (matplotlib colormaps)
    if palette_type == 'viridis' or palette_name in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import numpy as np
        
        # Use the specified colormap or default to plasma
        cmap_name = palette_name if palette_name in ['viridis', 'plasma', 'inferno', 'magma', 'cividis'] else 'plasma'
        cmap = cm.get_cmap(cmap_name)
        
        # For qualitative use (categorical samples), maximize distinction between adjacent colors
        if n_colors == 1:
            colors = [mcolors.rgb2hex(cmap(0.5))]
        elif n_colors == 2:
            colors = [mcolors.rgb2hex(cmap(0.15)), mcolors.rgb2hex(cmap(0.85))]
        else:
            # Interleaved sampling: alternate between start and end of colormap
            start, end = 0.05, 0.95
            
            # Create an interleaved pattern: [0, n-1, 1, n-2, 2, n-3, ...]
            indices = []
            left = 0
            right = n_colors - 1
            from_left = True
            
            while left <= right:
                if from_left:
                    t = left / (n_colors - 1)
                    indices.append(start + (end - start) * t)
                    left += 1
                else:
                    t = right / (n_colors - 1)
                    indices.append(start + (end - start) * t)
                    right -= 1
                from_left = not from_left
            
            colors = [mcolors.rgb2hex(cmap(idx)) for idx in indices]
        
        return colors
    
    # Fallback to hardcoded qualitative colors
    fallback_colors = [
        '#FF1F5B', '#00CD6C', '#009ADE', '#AF58BA', 
        '#FFC61E', '#F28522', '#A0B1BA', '#A6761D',
        '#E31A1C', '#33A02C', '#6A3D9A', '#FF7F00'
    ]
    if n_colors <= len(fallback_colors):
        return fallback_colors[:n_colors]
    else:
        return (fallback_colors * ((n_colors // len(fallback_colors)) + 1))[:n_colors]


def plot_uncollapsed_multiplot(context_counts, genome_counts, output_dir):
    """Create uncollapsed multiplot with both G and C centered mutations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with three panels - larger size for publication
    fig = plt.figure(figsize=(20, 18))
    # Adjust spacing: reduce top margin, increase space between panels
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 2], hspace=0.45, top=0.97, bottom=0.08)
    
    # Panel A: 3mer grouped bar (top) - both G and C centered
    ax_top = fig.add_subplot(gs[0])
    plot_3mer_grouped_bar_uncollapsed(context_counts, genome_counts, ax_top, remove_xaxis=True)
    
    # Panel B: Top 16 5mers grouped bar (middle) - both G and C centered
    ax_middle = fig.add_subplot(gs[1])
    plot_top_5mer_grouped_bar_uncollapsed(context_counts, genome_counts, ax_middle, top_n=16, remove_xaxis=True)
    
    # Panel C: 5mer average signature (bottom) - both G and C centered
    ax_bottom = fig.add_subplot(gs[2])
    plot_5mer_average_signature_uncollapsed(context_counts, genome_counts, ax_bottom)
    
    # Add panel labels aligned with plot titles
    pos_top = ax_top.get_position()
    pos_middle = ax_middle.get_position()
    pos_bottom = ax_bottom.get_position()
    
    # Offset to align with title baseline (titles have pad=20 from top)
    title_offset = 0.02  # Adjust to align with title position
    
    fig.text(0.02, pos_top.y1 - title_offset, 'A', fontsize=32, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_middle.y1 - title_offset, 'B', fontsize=32, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_bottom.y1 - title_offset, 'C', fontsize=32, fontweight='bold', va='top', ha='left')
    
    plt.savefig(os.path.join(output_dir, 'uncollapsed_mutation_rates_multiplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved uncollapsed mutation rates multiplot with G and C centered mutations")


def plot_3mer_grouped_bar_collapsed_aggregate(aggregate_rates, ax, remove_xaxis=False):
    """Create 3mer grouped bar plot using aggregate depth-normalized rates (collapsed)."""
    if not aggregate_rates:
        ax.text(0.5, 0.5, 'No aggregate rates available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract 3mer rates from 5mer aggregate rates (collapsed = C centered only)
    trimer_rates = defaultdict(float)
    for kmer, rate in aggregate_rates.items():
        if len(kmer) == 5 and kmer[2] == 'C':  # C centered 5mers only (collapsed)
            trimer = kmer[1:4]  # Extract 3mer context (positions 1, 2, 3)
            trimer_rates[trimer] += rate  # Sum rates for same 3mer
    
    # Create ordered list of 3mers
    all_trimers = sorted(trimer_rates.keys())
    if not all_trimers:
        ax.text(0.5, 0.5, 'No 3mer data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    rates = [trimer_rates[trimer] for trimer in all_trimers]
    
    # Create bar plot
    x_pos = np.arange(len(all_trimers))
    bars = ax.bar(x_pos, rates, width=0.8, color='orange', alpha=0.7)
    
    if not remove_xaxis:
        ax.set_xlabel('3mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=24, fontweight='bold')
    ax.set_title('3mer Mutation Rate - Collapsed (Aggregate Depth-Normalized)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    if remove_xaxis:
        ax.set_xticklabels(all_trimers, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels(all_trimers, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def plot_top_5mer_grouped_bar_collapsed_aggregate(aggregate_rates, ax, top_n=16, remove_xaxis=False):
    """Create grouped bar plot for top N 5mers using aggregate depth-normalized rates (collapsed)."""
    if not aggregate_rates:
        ax.text(0.5, 0.5, 'No aggregate rates available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Filter to C-centered kmers only (collapsed)
    c_centered_rates = {kmer: rate for kmer, rate in aggregate_rates.items() 
                       if len(kmer) == 5 and kmer[2] == 'C'}
    
    if not c_centered_rates:
        ax.text(0.5, 0.5, 'No C-centered 5mers available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get top N kmers by rate
    top_kmers = [k for k, _ in sorted(c_centered_rates.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    top_rates = [c_centered_rates[kmer] for kmer in top_kmers]
    
    # Create bar plot
    x_pos = np.arange(len(top_kmers))
    bars = ax.bar(x_pos, top_rates, width=0.8, color='orange', alpha=0.7)
    
    if not remove_xaxis:
        ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=24, fontweight='bold')
    ax.set_title(f'Top {top_n} 5mer Mutation Rate - Collapsed (Aggregate Depth-Normalized)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    if remove_xaxis:
        ax.set_xticklabels(top_kmers, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels(top_kmers, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def plot_5mer_average_signature_collapsed_aggregate(aggregate_rates, ax):
    """Create 5mer average signature plot using aggregate depth-normalized rates (collapsed)."""
    if not aggregate_rates:
        ax.text(0.5, 0.5, 'No aggregate rates available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Filter to C-centered kmers only (collapsed)
    c_centered_rates = {kmer: rate for kmer, rate in aggregate_rates.items() 
                       if len(kmer) == 5 and kmer[2] == 'C'}
    
    if not c_centered_rates:
        ax.text(0.5, 0.5, 'No C-centered 5mers available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Sort by mutation rate
    sorted_kmers = sorted(c_centered_rates.items(), key=lambda x: x[1], reverse=True)
    
    # Create plot
    kmers = [k[0] for k in sorted_kmers]
    means = [k[1] for k in sorted_kmers]
    
    bars = ax.bar(range(len(kmers)), means, width=0.95, color='orange', alpha=0.7)
    ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=24, fontweight='bold')
    ax.set_title('Average 5mer Mutation Rate - Collapsed (Aggregate Depth-Normalized)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(range(len(kmers)))
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.tick_params(axis='y', labelsize=18)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Adjust y-axis limits
    if means:
        max_value = max(means)
        min_value = min(means)
        y_range = max_value - min_value
        ax.set_ylim(min_value - 0.1 * abs(min_value), max_value + 0.1 * y_range)
    else:
        ax.set_ylim(-0.1, 0.1)
    
    # Add value labels on top bars (only for top 3 to avoid overlap)
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if i < 3 and abs(mean) > 1e-6:  # Only label top 3 values
            offset = 0.1 * abs(mean) if mean >= 0 else -0.1 * abs(mean)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                   f'{mean:.2e}', ha='center', va='bottom' if mean >= 0 else 'top', 
                   fontsize=20, fontweight='bold')


def plot_collapsed_multiplot(context_counts, genome_counts, output_dir, aggregate_rates=None):
    """Create collapsed multiplot with G mutations reverse complemented to C centered.
    
    Shows individual samples if aggregate_rates is None, otherwise shows aggregate rates.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with three panels - larger size for publication
    fig = plt.figure(figsize=(20, 18))
    # Adjust spacing: reduce top margin, increase space between panels
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 2], hspace=0.45, top=0.97, bottom=0.08)
    
    # Panel A: 3mer grouped bar (top) - G mutations reverse complemented to C
    ax_top = fig.add_subplot(gs[0])
    if aggregate_rates:
        plot_3mer_grouped_bar_collapsed_aggregate(aggregate_rates, ax_top, remove_xaxis=True)
    else:
        plot_3mer_grouped_bar_collapsed(context_counts, genome_counts, ax_top, remove_xaxis=True)
    
    # Panel B: Top 16 5mers grouped bar (middle) - G mutations reverse complemented to C
    ax_middle = fig.add_subplot(gs[1])
    if aggregate_rates:
        plot_top_5mer_grouped_bar_collapsed_aggregate(aggregate_rates, ax_middle, top_n=16, remove_xaxis=True)
    else:
        plot_top_5mer_grouped_bar_collapsed(context_counts, genome_counts, ax_middle, top_n=16, remove_xaxis=True)
    
    # Panel C: 5mer average signature (bottom) - G mutations reverse complemented to C
    ax_bottom = fig.add_subplot(gs[2])
    if aggregate_rates:
        plot_5mer_average_signature_collapsed_aggregate(aggregate_rates, ax_bottom)
    else:
        plot_5mer_average_signature_collapsed(context_counts, genome_counts, ax_bottom)
    
    # Add panel labels aligned with plot titles
    pos_top = ax_top.get_position()
    pos_middle = ax_middle.get_position()
    pos_bottom = ax_bottom.get_position()
    
    # Offset to align with title baseline (titles have pad=20 from top)
    title_offset = 0.02  # Adjust to align with title position
    
    fig.text(0.02, pos_top.y1 - title_offset, 'A', fontsize=32, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_middle.y1 - title_offset, 'B', fontsize=32, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_bottom.y1 - title_offset, 'C', fontsize=32, fontweight='bold', va='top', ha='left')
    
    if aggregate_rates:
        filename = 'collapsed_mutation_rates_multiplot_aggregate.png'
        print("Saved collapsed mutation rates multiplot (aggregate depth-normalized with background subtraction)")
    else:
        filename = 'collapsed_mutation_rates_multiplot_individual.png'
        print("Saved collapsed mutation rates multiplot (individual samples)")
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # If this is the genome-normalized version, we need to save it with a different name
    # This is handled in main() by copying the file


def plot_c_only_multiplot(context_counts, genome_counts, output_dir, aggregate_rates=None):
    """Create C-only multiplot with only C centered mutations.
    
    Args:
        context_counts: dict mapping sample_id -> kmer_counts (for compatibility)
        genome_counts: dict mapping kmer -> genome_count (for compatibility)
        output_dir: output directory
        aggregate_rates: optional dict mapping kmer -> aggregate depth-normalized rate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with three panels - larger size for publication
    fig = plt.figure(figsize=(20, 18))
    # Adjust spacing: reduce top margin, increase space between panels
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 2], hspace=0.45, top=0.97, bottom=0.08)
    
    # Panel A: 3mer grouped bar (top) - C centered only
    ax_top = fig.add_subplot(gs[0])
    plot_3mer_grouped_bar_c_only_aggregate(aggregate_rates, ax_top, remove_xaxis=True)
    
    # Panel B: Top 16 5mers grouped bar (middle) - C centered only
    ax_middle = fig.add_subplot(gs[1])
    plot_top_5mer_grouped_bar_c_only_aggregate(aggregate_rates, ax_middle, top_n=16, remove_xaxis=True)
    
    # Panel C: 5mer average signature (bottom) - C centered only
    ax_bottom = fig.add_subplot(gs[2])
    plot_5mer_average_signature_c_only_aggregate(aggregate_rates, ax_bottom)
    
    # Add panel labels aligned with plot titles
    pos_top = ax_top.get_position()
    pos_middle = ax_middle.get_position()
    pos_bottom = ax_bottom.get_position()
    
    # Offset to align with title baseline (titles have pad=20 from top)
    title_offset = 0.02  # Adjust to align with title position
    
    fig.text(0.02, pos_top.y1 - title_offset, 'A', fontsize=32, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_middle.y1 - title_offset, 'B', fontsize=32, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_bottom.y1 - title_offset, 'C', fontsize=32, fontweight='bold', va='top', ha='left')
    
    plt.savefig(os.path.join(output_dir, 'c_only_mutation_rates_multiplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved C-only mutation rates multiplot with C centered mutations only (aggregate depth-normalized)")


def plot_3mer_grouped_bar_uncollapsed(context_counts, genome_counts, ax, remove_xaxis=False):
    """Create 3mer grouped bar plot for uncollapsed data (both G and C centered)."""
    # Extract 3mer counts from 5mer data, separating C and G centered
    sample_c_3mer_counts = {}
    sample_g_3mer_counts = {}
    for sample, kmer_counts in context_counts.items():
        sample_c_3mer_counts[sample] = {}
        sample_g_3mer_counts[sample] = {}
        for kmer, count in kmer_counts.items():
            if len(kmer) == 5:
                # Extract 3mer context (positions 1, 2, 3)
                trimer = kmer[1:4]
                if kmer[2] == 'C':  # C centered 5mer
                    sample_c_3mer_counts[sample][trimer] = sample_c_3mer_counts[sample].get(trimer, 0) + count
                elif kmer[2] == 'G':  # G centered 5mer
                    sample_g_3mer_counts[sample][trimer] = sample_g_3mer_counts[sample].get(trimer, 0) + count
    
    # Get all EMS samples (including 3d/7d)
    ems_samples = [s for s in sample_c_3mer_counts if 'EMS' in s]
    ems_samples = sort_samples_by_ems_number(ems_samples)
    if not ems_samples:
        ax.text(0.5, 0.5, 'No EMS samples found', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get all C-centered and G-centered 3mers
    all_c_trimers = set()
    all_g_trimers = set()
    for sample in ems_samples:
        all_c_trimers.update(sample_c_3mer_counts[sample].keys())
        all_g_trimers.update(sample_g_3mer_counts[sample].keys())
    
    # Create ordered lists: C-centered on left, G-centered on right
    c_centered_3mers_sorted = sorted(['ACA', 'CCA', 'GCA', 'TCA', 'ACC', 'CCC', 'GCC', 'TCC', 
                                       'ACG', 'CCG', 'GCG', 'TCG', 'ACT', 'CCT', 'GCT', 'TCT'])
    
    # G-centered reverse complements in matching order
    def reverse_complement_3mer(seq):
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join(complement[base] for base in reversed(seq))
    
    g_centered_3mers_sorted = [reverse_complement_3mer(c_tri) for c_tri in c_centered_3mers_sorted]
    
    # Filter to only those present in data
    ordered_trimers = []
    for c_tri in c_centered_3mers_sorted:
        if c_tri in all_c_trimers:
            ordered_trimers.append(('C', c_tri))
    
    for g_tri in g_centered_3mers_sorted:
        if g_tri in all_g_trimers:
            ordered_trimers.append(('G', g_tri))
    
    # Calculate mutation rates for each sample
    genome_c_3mer_counts = {}
    genome_g_3mer_counts = {}
    for kmer, count in genome_counts.items():
        if len(kmer) == 5:
            trimer = kmer[1:4]
            if kmer[2] == 'C':
                genome_c_3mer_counts[trimer] = genome_c_3mer_counts.get(trimer, 0) + count
            elif kmer[2] == 'G':
                genome_g_3mer_counts[trimer] = genome_g_3mer_counts.get(trimer, 0) + count
    
    # Create grouped bar plot
    x_pos = np.arange(len(ordered_trimers))
    width = 0.8 / len(ems_samples)
    
    colors = get_colorblind_friendly_palette(len(ems_samples), COLOR_PALETTE_TYPE, COLOR_PALETTE_NAME)
    
    for i, sample in enumerate(ems_samples):
        sample_rates = []
        for center_type, trimer in ordered_trimers:
            if center_type == 'C':
                sample_count = sample_c_3mer_counts[sample].get(trimer, 0)
                total_sample = sum(sample_c_3mer_counts[sample].values())
            else:  # 'G'
                sample_count = sample_g_3mer_counts[sample].get(trimer, 0)
                total_sample = sum(sample_g_3mer_counts[sample].values())
            
            if total_sample > 0:
                mutation_rate = sample_count / total_sample
            else:
                mutation_rate = 0
            sample_rates.append(mutation_rate)
        
        ax.bar(x_pos + i * width, sample_rates, width, label=clean_sample_label(sample), color=colors[i], alpha=0.9)
    
    # Add vertical line to separate C and G sections
    num_c_trimers = sum(1 for center_type, _ in ordered_trimers if center_type == 'C')
    if num_c_trimers > 0 and num_c_trimers < len(ordered_trimers):
        ax.axvline(x=num_c_trimers - 0.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add section labels
    num_g_trimers = len(ordered_trimers) - num_c_trimers
    if num_c_trimers > 0:
        ax.text(num_c_trimers / 2 - 0.5, ax.get_ylim()[1] * 0.95, 'C>T', 
                ha='center', va='top', fontsize=16, fontweight='bold', color='orange')
    if num_g_trimers > 0:
        ax.text(num_c_trimers + num_g_trimers / 2 - 0.5, ax.get_ylim()[1] * 0.95, 'G>A', 
                ha='center', va='top', fontsize=16, fontweight='bold', color='blue')
    
    if not remove_xaxis:
        ax.set_xlabel('3mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Mutation Rate', fontsize=24, fontweight='bold')
    ax.set_title('3mer Mutation Rate - C>T (left) | G>A (right)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width * (len(ems_samples) - 1) / 2)
    if remove_xaxis:
        ax.set_xticklabels([trimer for _, trimer in ordered_trimers], rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels([trimer for _, trimer in ordered_trimers], rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=13, frameon=True, fancybox=True)


def plot_top_5mer_grouped_bar_uncollapsed(context_counts, genome_counts, ax, top_n=16, remove_xaxis=False):
    """Create grouped bar plot for top N 5mers (uncollapsed: C and G centered separately)."""
    # Get all EMS samples (including 3d/7d) and sort by EMS number
    ems_samples = [s for s in context_counts.keys() if 'EMS' in s]
    ems_samples = sort_samples_by_ems_number(ems_samples)
    if not ems_samples:
        ax.text(0.5, 0.5, 'No EMS samples found', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate average mutation rates to find top kmers
    c_avg_rates = defaultdict(list)
    g_avg_rates = defaultdict(list)
    
    for sample in ems_samples:
        # C centered
        c_centered = {k: v for k, v in context_counts[sample].items() if len(k) == 5 and k[2] == 'C'}
        c_total = sum(c_centered.values())
        for kmer, count in c_centered.items():
            if c_total > 0:
                c_avg_rates[kmer].append(count / c_total)
        
        # G centered
        g_centered = {k: v for k, v in context_counts[sample].items() if len(k) == 5 and k[2] == 'G'}
        g_total = sum(g_centered.values())
        for kmer, count in g_centered.items():
            if g_total > 0:
                g_avg_rates[kmer].append(count / g_total)
    
    # Get top N kmers by average mutation rate
    c_means = {k: np.mean(v) for k, v in c_avg_rates.items() if v}
    g_means = {k: np.mean(v) for k, v in g_avg_rates.items() if v}
    
    c_top = sorted(c_means.items(), key=lambda x: x[1], reverse=True)[:top_n//2]
    g_top = sorted(g_means.items(), key=lambda x: x[1], reverse=True)[:top_n//2]
    
    c_top_kmers = [k for k, _ in c_top]
    g_top_kmers = [k for k, _ in g_top]
    
    # Create grouped bar plot data
    all_top_kmers = c_top_kmers + g_top_kmers
    x_pos = np.arange(len(all_top_kmers))
    width = 0.8 / len(ems_samples)
    
    colors = get_colorblind_friendly_palette(len(ems_samples), COLOR_PALETTE_TYPE, COLOR_PALETTE_NAME)
    
    for i, sample in enumerate(ems_samples):
        sample_rates = []
        
        # C centered rates
        c_centered = {k: v for k, v in context_counts[sample].items() if len(k) == 5 and k[2] == 'C'}
        c_total = sum(c_centered.values())
        
        for kmer in c_top_kmers:
            if c_total > 0:
                sample_rates.append(c_centered.get(kmer, 0) / c_total)
            else:
                sample_rates.append(0)
        
        # G centered rates
        g_centered = {k: v for k, v in context_counts[sample].items() if len(k) == 5 and k[2] == 'G'}
        g_total = sum(g_centered.values())
        
        for kmer in g_top_kmers:
            if g_total > 0:
                sample_rates.append(g_centered.get(kmer, 0) / g_total)
            else:
                sample_rates.append(0)
        
        ax.bar(x_pos + i * width, sample_rates, width, label=clean_sample_label(sample), color=colors[i], alpha=0.9)
    
    # Add vertical line to separate C and G
    if c_top_kmers and g_top_kmers:
        ax.axvline(x=len(c_top_kmers) - 0.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    if not remove_xaxis:
        ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Mutation Rate', fontsize=24, fontweight='bold')
    ax.set_title(f'Top {top_n} 5mer Mutation Rate (C>T | G>A)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width * (len(ems_samples) - 1) / 2)
    if remove_xaxis:
        ax.set_xticklabels(all_top_kmers, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels(all_top_kmers, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=13, frameon=True, fancybox=True)


def plot_5mer_average_signature_uncollapsed(context_counts, genome_counts, ax):
    """Create 5mer average signature plot for uncollapsed data (G and C centered side-by-side)."""
    # Process G and C centered mutations separately
    c_kmers = set()
    g_kmers = set()
    
    # Collect C and G centered kmers separately
    for sample, sample_counts in context_counts.items():
        for kmer in sample_counts.keys():
            if len(kmer) == 5 and kmer[2] == 'C':
                c_kmers.add(kmer)
            elif len(kmer) == 5 and kmer[2] == 'G':
                g_kmers.add(kmer)
    
    # Calculate mutation rates for C centered
    c_avg_rates = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
        c_centered = {k: v for k, v in kmer_counts.items() if len(k) == 5 and k[2] == 'C'}
        c_total = sum(c_centered.values())
        
        for kmer in c_kmers:
            if c_total > 0:
                c_avg_rates[kmer].append(c_centered.get(kmer, 0) / c_total)
    
    # Calculate mutation rates for G centered
    g_avg_rates = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
        g_centered = {k: v for k, v in kmer_counts.items() if len(k) == 5 and k[2] == 'G'}
        g_total = sum(g_centered.values())
        
        for kmer in g_kmers:
            if g_total > 0:
                g_avg_rates[kmer].append(g_centered.get(kmer, 0) / g_total)
    
    # Calculate means and stds for both
    c_mean_rates = {}
    c_std_rates = {}
    for kmer, rates in c_avg_rates.items():
        c_mean_rates[kmer] = np.mean(rates)
        c_std_rates[kmer] = np.std(rates)
    
    g_mean_rates = {}
    g_std_rates = {}
    for kmer, rates in g_avg_rates.items():
        g_mean_rates[kmer] = np.mean(rates)
        g_std_rates[kmer] = np.std(rates)
    
    # Sort by mutation rate
    c_sorted = sorted(c_mean_rates.items(), key=lambda x: x[1], reverse=True)
    g_sorted = sorted(g_mean_rates.items(), key=lambda x: x[1], reverse=True)
    
    # Create side-by-side plot with clear separation
    c_kmers_list = [k[0] for k in c_sorted]
    g_kmers_list = [k[0] for k in g_sorted]
    c_means = [k[1] for k in c_sorted]
    g_means = [k[1] for k in g_sorted]
    c_stds = [c_std_rates[k[0]] for k in c_sorted]
    g_stds = [g_std_rates[k[0]] for k in g_sorted]
    
    # Position bars: C centered on left, G centered on right with gap
    max_c = len(c_kmers_list)
    max_g = len(g_kmers_list)
    gap = max(max_c, max_g) // 4  # Gap between sections
    
    x_c = np.arange(max_c)
    x_g = np.arange(max_g) + max_c + gap
    
    # Create plot
    bars_c = ax.bar(x_c, c_means, width=0.8, color='orange', alpha=0.7, label='C>T')
    bars_g = ax.bar(x_g, g_means, width=0.8, color='blue', alpha=0.7, label='G>A')
    
    ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Mutation Rate', fontsize=24, fontweight='bold')
    ax.set_title('Average 5mer Mutation Rate - G and C centered (side-by-side)', fontsize=26, fontweight='bold', pad=20)
    
    # Add section labels with top 3 kmers info
    c_top3_text = "C>T Mutations\n"
    for i in range(min(3, len(c_sorted))):
        kmer, rate = c_sorted[i]
        c_top3_text += f"{kmer}: {rate:.4f}\n"
    
    ax.text(max_c/2, ax.get_ylim()[1] * 0.92, c_top3_text.strip(), 
            ha='center', va='top', fontsize=22, fontweight='bold', color='orange',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='white', alpha=0.9, 
                     edgecolor='orange', linewidth=2))
    
    # G>A section label with top 3 kmers
    g_top3_text = "G>A Mutations\n"
    for i in range(min(3, len(g_sorted))):
        kmer, rate = g_sorted[i]
        g_top3_text += f"{kmer}: {rate:.4f}\n"
    
    ax.text(max_c + gap + max_g/2, ax.get_ylim()[1] * 0.92, g_top3_text.strip(), 
            ha='center', va='top', fontsize=22, fontweight='bold', color='blue',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='white', alpha=0.9,
                     edgecolor='blue', linewidth=2))
    
    # Add vertical line to separate sections
    ax.axvline(x=max_c + gap/2, color='black', linestyle='-', alpha=0.3, linewidth=2)
    
    ax.set_xticks([])  # Remove x-axis labels
    ax.tick_params(axis='y', labelsize=18)
    
    # Adjust y-axis limits
    all_means = c_means + g_means
    if all_means:
        max_value = max(all_means)
        ax.set_ylim(0, max_value * 1.1)
    else:
        ax.set_ylim(0, 1.0)
    
    ax.legend(fontsize=14, frameon=True, fancybox=True)


def plot_3mer_grouped_bar_collapsed(context_counts, genome_counts, ax, remove_xaxis=False):
    """Create 3mer grouped bar plot for collapsed data (G mutations reverse complemented to C)."""
    # Extract 3mer counts from 5mer data, reverse complement G centered to C
    sample_3mer_counts = {}
    for sample, kmer_counts in context_counts.items():
        sample_3mer_counts[sample] = {}
        for kmer, count in kmer_counts.items():
            if len(kmer) == 5:
                if kmer[2] == 'C':  # C centered - use as is
                    trimer = kmer[1:4]
                    sample_3mer_counts[sample][trimer] = sample_3mer_counts[sample].get(trimer, 0) + count
                elif kmer[2] == 'G':  # G centered - reverse complement to C
                    rc_kmer = reverse_complement(kmer)
                    if rc_kmer[2] == 'C':  # Should be C centered after RC
                        trimer = rc_kmer[1:4]
                        sample_3mer_counts[sample][trimer] = sample_3mer_counts[sample].get(trimer, 0) + count
    
    # Get all EMS samples (including 3d/7d)
    ems_samples = [s for s in sample_3mer_counts if 'EMS' in s]
    ems_samples = sort_samples_by_ems_number(ems_samples)
    if not ems_samples:
        ax.text(0.5, 0.5, 'No EMS samples found', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get all 3mer contexts
    all_trimers = set()
    for sample_counts in sample_3mer_counts.values():
        all_trimers.update(sample_counts.keys())
    all_trimers = sorted(all_trimers)
    
    # Create grouped bar plot
    x_pos = np.arange(len(all_trimers))
    width = 0.8 / len(ems_samples)
    
    colors = get_colorblind_friendly_palette(len(ems_samples), COLOR_PALETTE_TYPE, COLOR_PALETTE_NAME)
    
    for i, sample in enumerate(ems_samples):
        sample_rates = []
        total_sample = sum(sample_3mer_counts[sample].values())
        for trimer in all_trimers:
            sample_count = sample_3mer_counts[sample].get(trimer, 0)
            if total_sample > 0:
                mutation_rate = sample_count / total_sample
            else:
                mutation_rate = 0
            sample_rates.append(mutation_rate)
        
        ax.bar(x_pos + i * width, sample_rates, width, label=clean_sample_label(sample), color=colors[i], alpha=0.9)
    
    if not remove_xaxis:
        ax.set_xlabel('3mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Mutation Rate', fontsize=24, fontweight='bold')
    ax.set_title('3mer Mutation Rate - Collapsed', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width * (len(ems_samples) - 1) / 2)
    if remove_xaxis:
        ax.set_xticklabels(all_trimers, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels(all_trimers, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=13, frameon=True, fancybox=True)


def plot_top_5mer_grouped_bar_collapsed(context_counts, genome_counts, ax, top_n=16, remove_xaxis=False):
    """Create grouped bar plot for top N 5mers (collapsed: G mutations reverse complemented to C)."""
    # Get EMS samples (excluding 3d/7d)
    ems_samples = [s for s in context_counts.keys() if 'EMS' in s and '3d' not in s and '7d' not in s]
    ems_samples = sort_samples_by_ems_number(ems_samples)
    if not ems_samples:
        ax.text(0.5, 0.5, 'No EMS samples found', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate average mutation rates to find top kmers
    avg_rates = defaultdict(list)
    
    for sample in ems_samples:
        # Process kmers: C centered as is, G centered reverse complemented to C
        processed_kmers = {}
        for kmer, count in context_counts[sample].items():
            if len(kmer) == 5:
                if kmer[2] == 'C':  # C centered - use as is
                    processed_kmers[kmer] = count
                elif kmer[2] == 'G':  # G centered - reverse complement to C
                    rc_kmer = reverse_complement(kmer)
                    if rc_kmer[2] == 'C':
                        processed_kmers[rc_kmer] = processed_kmers.get(rc_kmer, 0) + count
        
        total_sample = sum(processed_kmers.values())
        for kmer, count in processed_kmers.items():
            if total_sample > 0:
                avg_rates[kmer].append(count / total_sample)
    
    # Get top N kmers by average mutation rate
    means = {k: np.mean(v) for k, v in avg_rates.items() if v}
    top_kmers = [k for k, _ in sorted(means.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    
    # Create grouped bar plot data
    x_pos = np.arange(len(top_kmers))
    width = 0.8 / len(ems_samples)
    
    colors = get_colorblind_friendly_palette(len(ems_samples), COLOR_PALETTE_TYPE, COLOR_PALETTE_NAME)
    
    for i, sample in enumerate(ems_samples):
        # Process kmers for this sample
        processed_kmers = {}
        for kmer, count in context_counts[sample].items():
            if len(kmer) == 5:
                if kmer[2] == 'C':
                    processed_kmers[kmer] = count
                elif kmer[2] == 'G':
                    rc_kmer = reverse_complement(kmer)
                    if rc_kmer[2] == 'C':
                        processed_kmers[rc_kmer] = processed_kmers.get(rc_kmer, 0) + count
        
        total_sample = sum(processed_kmers.values())
        sample_rates = []
        for kmer in top_kmers:
            if total_sample > 0:
                sample_rates.append(processed_kmers.get(kmer, 0) / total_sample)
            else:
                sample_rates.append(0)
        
        ax.bar(x_pos + i * width, sample_rates, width, label=clean_sample_label(sample), color=colors[i], alpha=0.9)
    
    if not remove_xaxis:
        ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Mutation Rate', fontsize=24, fontweight='bold')
    ax.set_title(f'Top {top_n} 5mer Mutation Rate (Collapsed)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width * (len(ems_samples) - 1) / 2)
    if remove_xaxis:
        ax.set_xticklabels(top_kmers, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels(top_kmers, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=13, frameon=True, fancybox=True)


def plot_5mer_average_signature_collapsed(context_counts, genome_counts, ax):
    """Create 5mer average signature plot for collapsed data (G mutations reverse complemented to C)."""
    # Collect all C-centered kmers (including reverse complemented G mutations)
    all_kmers = set()
    for sample, sample_counts in context_counts.items():
        for kmer in sample_counts.keys():
            if len(kmer) == 5:
                if kmer[2] == 'C':  # C centered - use as is
                    all_kmers.add(kmer)
                elif kmer[2] == 'G':  # G centered - reverse complement to C
                    rc_kmer = reverse_complement(kmer)
                    if rc_kmer[2] == 'C':  # Should be C centered after RC
                        all_kmers.add(rc_kmer)
    
    # Average mutation rates across samples
    avg_rates = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
            
        # Process kmers: C centered as is, G centered reverse complemented to C
        processed_kmers = {}
        for kmer, count in kmer_counts.items():
            if len(kmer) == 5:
                if kmer[2] == 'C':  # C centered - use as is
                    processed_kmers[kmer] = count
                elif kmer[2] == 'G':  # G centered - reverse complement to C
                    rc_kmer = reverse_complement(kmer)
                    if rc_kmer[2] == 'C':  # Should be C centered after RC
                        processed_kmers[rc_kmer] = processed_kmers.get(rc_kmer, 0) + count
        
        total_sample = sum(processed_kmers.values())
        for kmer in all_kmers:
            if total_sample > 0:
                avg_rates[kmer].append(processed_kmers.get(kmer, 0) / total_sample)
    
    # Calculate mean and std
    mean_rates = {}
    std_rates = {}
    for kmer, rates in avg_rates.items():
        mean_rates[kmer] = np.mean(rates)
        std_rates[kmer] = np.std(rates)
    
    # Sort by mean mutation rate
    sorted_kmers = sorted(mean_rates.items(), key=lambda x: x[1], reverse=True)
    
    # Create plot
    kmers = [k[0] for k in sorted_kmers]
    means = [k[1] for k in sorted_kmers]
    stds = [std_rates[k[0]] for k in sorted_kmers]
    
    bars = ax.bar(range(len(kmers)), means, width=0.95,
                  color='orange', alpha=0.7)
    ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Mutation Rate', fontsize=24, fontweight='bold')
    ax.set_title('Average 5mer Mutation Rate - Collapsed', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(range(len(kmers)))
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.tick_params(axis='y', labelsize=18)
    
    # Adjust y-axis limits
    if means:
        max_value = max(means)
        ax.set_ylim(0, max_value * 1.1)
    else:
        ax.set_ylim(0, 1.0)
    
    # Add value labels on top bars (only for top 3 to avoid overlap)
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if i < 3 and mean > 0.01:  # Only label top 3 highly mutated values
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{mean:.4f}', ha='center', va='bottom', fontsize=25, fontweight='bold')


def plot_3mer_grouped_bar_c_only(context_counts, genome_counts, ax, remove_xaxis=False):
    """Create 3mer grouped bar plot for C-only data (C centered only)."""
    # Extract 3mer counts from 5mer data for C centered only
    sample_3mer_counts = {}
    for sample, kmer_counts in context_counts.items():
        sample_3mer_counts[sample] = {}
        for kmer, count in kmer_counts.items():
            if len(kmer) == 5 and kmer[2] == 'C':  # C centered 5mers only
                # Extract 3mer context (positions 1, 2, 3)
                trimer = kmer[1:4]
                sample_3mer_counts[sample][trimer] = sample_3mer_counts[sample].get(trimer, 0) + count
    
    # Get all EMS samples (including 3d/7d)
    ems_samples = [s for s in sample_3mer_counts if 'EMS' in s]
    ems_samples = sort_samples_by_ems_number(ems_samples)
    if not ems_samples:
        ax.text(0.5, 0.5, 'No EMS samples found', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get all 3mer contexts
    all_trimers = set()
    for sample_counts in sample_3mer_counts.values():
        all_trimers.update(sample_counts.keys())
    all_trimers = sorted(all_trimers)
    
    # Create grouped bar plot
    x_pos = np.arange(len(all_trimers))
    width = 0.8 / len(ems_samples)
    
    colors = get_colorblind_friendly_palette(len(ems_samples), COLOR_PALETTE_TYPE, COLOR_PALETTE_NAME)
    
    for i, sample in enumerate(ems_samples):
        sample_rates = []
        total_sample = sum(sample_3mer_counts[sample].values())
        for trimer in all_trimers:
            sample_count = sample_3mer_counts[sample].get(trimer, 0)
            if total_sample > 0:
                mutation_rate = sample_count / total_sample
            else:
                mutation_rate = 0
            sample_rates.append(mutation_rate)
        
        ax.bar(x_pos + i * width, sample_rates, width, label=clean_sample_label(sample), color=colors[i], alpha=0.9)
    
    if not remove_xaxis:
        ax.set_xlabel('3mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Mutation Rate', fontsize=24, fontweight='bold')
    ax.set_title('3mer Mutation Rate - C centered only', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width * (len(ems_samples) - 1) / 2)
    if remove_xaxis:
        ax.set_xticklabels(all_trimers, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels(all_trimers, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=13, frameon=True, fancybox=True)


def plot_top_5mer_grouped_bar_c_only(context_counts, genome_counts, ax, top_n=16, remove_xaxis=False):
    """Create grouped bar plot for top N 5mers (C-only: only C centered mutations)."""
    # Get EMS samples (excluding 3d/7d)
    ems_samples = [s for s in context_counts.keys() if 'EMS' in s and '3d' not in s and '7d' not in s]
    ems_samples = sort_samples_by_ems_number(ems_samples)
    if not ems_samples:
        ax.text(0.5, 0.5, 'No EMS samples found', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate average mutation rates to find top kmers
    avg_rates = defaultdict(list)
    genome_c_centered = filter_centered_kmers(genome_counts, 'C')
    
    for sample in ems_samples:
        sample_c_centered = filter_centered_kmers(context_counts[sample], 'C')
        total_sample = sum(sample_c_centered.values())
        for kmer, count in sample_c_centered.items():
            if total_sample > 0:
                avg_rates[kmer].append(count / total_sample)
    
    # Get top N kmers by average mutation rate
    means = {k: np.mean(v) for k, v in avg_rates.items() if v}
    top_kmers = [k for k, _ in sorted(means.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    
    # Create grouped bar plot data
    x_pos = np.arange(len(top_kmers))
    width = 0.8 / len(ems_samples)
    
    colors = get_colorblind_friendly_palette(len(ems_samples), COLOR_PALETTE_TYPE, COLOR_PALETTE_NAME)
    
    for i, sample in enumerate(ems_samples):
        sample_c_centered = filter_centered_kmers(context_counts[sample], 'C')
        total_sample = sum(sample_c_centered.values())
        
        sample_rates = []
        for kmer in top_kmers:
            if total_sample > 0:
                sample_rates.append(sample_c_centered.get(kmer, 0) / total_sample)
            else:
                sample_rates.append(0)
        
        ax.bar(x_pos + i * width, sample_rates, width, label=clean_sample_label(sample), color=colors[i], alpha=0.9)
    
    if not remove_xaxis:
        ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Mutation Rate', fontsize=24, fontweight='bold')
    ax.set_title(f'Top {top_n} 5mer Mutation Rate (C-only)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width * (len(ems_samples) - 1) / 2)
    if remove_xaxis:
        ax.set_xticklabels(top_kmers, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels(top_kmers, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=13, frameon=True, fancybox=True)


def plot_3mer_grouped_bar_c_only_aggregate(aggregate_rates, ax, remove_xaxis=False):
    """Create 3mer grouped bar plot using aggregate depth-normalized rates (C centered only)."""
    if not aggregate_rates:
        ax.text(0.5, 0.5, 'No aggregate rates available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract 3mer rates from 5mer aggregate rates
    trimer_rates = defaultdict(float)
    for kmer, rate in aggregate_rates.items():
        if len(kmer) == 5 and kmer[2] == 'C':  # C centered 5mers only
            trimer = kmer[1:4]  # Extract 3mer context (positions 1, 2, 3)
            trimer_rates[trimer] += rate  # Sum rates for same 3mer
    
    # Create ordered list of 3mers
    all_trimers = sorted(trimer_rates.keys())
    if not all_trimers:
        ax.text(0.5, 0.5, 'No 3mer data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    rates = [trimer_rates[trimer] for trimer in all_trimers]
    
    # Create bar plot
    x_pos = np.arange(len(all_trimers))
    bars = ax.bar(x_pos, rates, width=0.8, color='orange', alpha=0.7)
    
    if not remove_xaxis:
        ax.set_xlabel('3mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=24, fontweight='bold')
    ax.set_title('3mer Mutation Rate - C centered (Aggregate Depth-Normalized)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    if remove_xaxis:
        ax.set_xticklabels(all_trimers, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels(all_trimers, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def plot_top_5mer_grouped_bar_c_only_aggregate(aggregate_rates, ax, top_n=16, remove_xaxis=False):
    """Create grouped bar plot for top N 5mers using aggregate depth-normalized rates (C-only)."""
    if not aggregate_rates:
        ax.text(0.5, 0.5, 'No aggregate rates available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Filter to C-centered kmers only
    c_centered_rates = {kmer: rate for kmer, rate in aggregate_rates.items() 
                       if len(kmer) == 5 and kmer[2] == 'C'}
    
    if not c_centered_rates:
        ax.text(0.5, 0.5, 'No C-centered 5mers available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get top N kmers by rate
    top_kmers = [k for k, _ in sorted(c_centered_rates.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    top_rates = [c_centered_rates[kmer] for kmer in top_kmers]
    
    # Create bar plot
    x_pos = np.arange(len(top_kmers))
    bars = ax.bar(x_pos, top_rates, width=0.8, color='orange', alpha=0.7)
    
    if not remove_xaxis:
        ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=24, fontweight='bold')
    ax.set_title(f'Top {top_n} 5mer Mutation Rate (C-only, Aggregate Depth-Normalized)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    if remove_xaxis:
        ax.set_xticklabels(top_kmers, rotation=45, ha='right', fontsize=16)
        ax.set_xlabel('')  # Remove x-axis title only
    else:
        ax.set_xticklabels(top_kmers, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def plot_5mer_average_signature_c_only_aggregate(aggregate_rates, ax):
    """Create 5mer average signature plot using aggregate depth-normalized rates (C centered only)."""
    if not aggregate_rates:
        ax.text(0.5, 0.5, 'No aggregate rates available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Filter to C-centered kmers only
    c_centered_rates = {kmer: rate for kmer, rate in aggregate_rates.items() 
                       if len(kmer) == 5 and kmer[2] == 'C'}
    
    if not c_centered_rates:
        ax.text(0.5, 0.5, 'No C-centered 5mers available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Sort by mutation rate
    sorted_kmers = sorted(c_centered_rates.items(), key=lambda x: x[1], reverse=True)
    
    # Create plot
    kmers = [k[0] for k in sorted_kmers]
    means = [k[1] for k in sorted_kmers]
    
    bars = ax.bar(range(len(kmers)), means, width=0.95, color='orange', alpha=0.7)
    ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=24, fontweight='bold')
    ax.set_title('Average 5mer Mutation Rate - C centered (Aggregate Depth-Normalized)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(range(len(kmers)))
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.tick_params(axis='y', labelsize=18)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Adjust y-axis limits
    if means:
        max_value = max(means)
        min_value = min(means)
        y_range = max_value - min_value
        ax.set_ylim(min_value - 0.1 * abs(min_value), max_value + 0.1 * y_range)
    else:
        ax.set_ylim(-0.1, 0.1)
    
    # Add value labels on top bars (only for top 3 to avoid overlap)
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if i < 3 and abs(mean) > 1e-6:  # Only label top 3 values
            offset = 0.1 * abs(mean) if mean >= 0 else -0.1 * abs(mean)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                   f'{mean:.2e}', ha='center', va='bottom' if mean >= 0 else 'top', 
                   fontsize=20, fontweight='bold')


def plot_5mer_average_signature_c_only(context_counts, genome_counts, ax):
    """Create 5mer average signature plot for C-only data (C centered only)."""
    # Filter to C-centered kmers only
    genome_c_centered = filter_centered_kmers(genome_counts, 'C')
    
    # Collect all C-centered kmers across samples (excluding 3d/7d)
    all_kmers = set()
    for sample, sample_counts in context_counts.items():
        sample_c_centered = filter_centered_kmers(sample_counts, 'C')
        all_kmers.update(sample_c_centered.keys())
    
    # Average mutation rates across samples
    avg_rates = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
            
        sample_c_centered = filter_centered_kmers(kmer_counts, 'C')
        total_sample = sum(sample_c_centered.values())
        
        for kmer in all_kmers:
            if total_sample > 0:
                avg_rates[kmer].append(sample_c_centered.get(kmer, 0) / total_sample)
    
    # Calculate mean and std
    mean_rates = {}
    std_rates = {}
    for kmer, rates in avg_rates.items():
        mean_rates[kmer] = np.mean(rates)
        std_rates[kmer] = np.std(rates)
    
    # Sort by mean mutation rate
    sorted_kmers = sorted(mean_rates.items(), key=lambda x: x[1], reverse=True)
    
    # Create plot
    kmers = [k[0] for k in sorted_kmers]
    means = [k[1] for k in sorted_kmers]
    stds = [std_rates[k[0]] for k in sorted_kmers]
    
    bars = ax.bar(range(len(kmers)), means, width=0.95,
                  color='orange', alpha=0.7)
    ax.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax.set_ylabel('Mutation Rate', fontsize=24, fontweight='bold')
    ax.set_title('Average 5mer Mutation Rate - C centered only', fontsize=26, fontweight='bold', pad=20)
    ax.set_xticks(range(len(kmers)))
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.tick_params(axis='y', labelsize=18)
    
    # Adjust y-axis limits
    if means:
        max_value = max(means)
        ax.set_ylim(0, max_value * 1.1)
    else:
        ax.set_ylim(0, 1.0)
    
    # Add value labels on top bars (only for top 3 to avoid overlap)
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if i < 3 and mean > 0.01:  # Only label top 3 highly mutated values
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{mean:.4f}', ha='center', va='bottom', fontsize=25, fontweight='bold')


def plot_mutation_rate_distribution(context_counts, genome_counts, output_dir):
    """Plot distribution of mutation rates to identify patterns."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to C-centered kmers only
    genome_c_centered = filter_centered_kmers(genome_counts, 'C')
    
    # Get EMS samples (excluding 3d/7d)
    ems_samples = [s for s in context_counts.keys() if 'EMS' in s and '3d' not in s and '7d' not in s]
    
    # Collect all mutation rates
    all_rates = []
    for sample in ems_samples:
        sample_c_centered = filter_centered_kmers(context_counts[sample], 'C')
        total_sample = sum(sample_c_centered.values())
        for kmer, count in sample_c_centered.items():
            if total_sample > 0:
                all_rates.append(count / total_sample)
    
    # Create distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(all_rates, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Mutation Rate')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of 5mer Mutation Rates')
    ax1.set_yscale('log')
    
    # Box plot
    ax2.boxplot(all_rates, vert=True)
    ax2.set_ylabel('Mutation Rate')
    ax2.set_title('5mer Mutation Rates Box Plot')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5mer_mutation_rate_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    mean_rate = np.mean(all_rates)
    median_rate = np.median(all_rates)
    std_rate = np.std(all_rates)
    q95_rate = np.percentile(all_rates, 95)
    
    print(f"Mutation rate statistics:")
    print(f"  Mean: {mean_rate:.6f}")
    print(f"  Median: {median_rate:.6f}")
    print(f"  Std: {std_rate:.6f}")
    print(f"  95th percentile: {q95_rate:.6f}")


def plot_mutation_rate_heatmap(context_counts, genome_counts, output_dir, 
                               top_n=20, data_type='c_only'):
    """Create heatmap of top mutation rates across samples."""
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    if data_type == 'uncollapsed':
        # For uncollapsed data, create separate heatmaps for C and G centered
        # C centered heatmap
        c_kmers = set()
        for sample, sample_counts in context_counts.items():
            for kmer in sample_counts.keys():
                if len(kmer) == 5 and kmer[2] == 'C':
                    c_kmers.add(kmer)
        
        # Get all EMS samples (including 3d/7d)
        ems_samples = [s for s in context_counts.keys() if 'EMS' in s]
        ems_samples = sort_samples_by_ems_number(ems_samples)
        
        # C centered heatmap
        c_heatmap_data = []
        for sample in ems_samples:
            c_centered = {k: v for k, v in context_counts[sample].items() if len(k) == 5 and k[2] == 'C'}
            c_total = sum(c_centered.values())
            
            # Get top C centered kmers
            c_sorted = sorted(c_centered.items(), key=lambda x: x[1]/c_total if c_total > 0 else 0, reverse=True)
            c_top_kmers = [k[0] for k in c_sorted[:top_n]]
            row = [c_centered.get(kmer, 0) / c_total if c_total > 0 else 0 for kmer in c_top_kmers]
            c_heatmap_data.append(row)
        
        # Create C centered heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        c_heatmap_df = pd.DataFrame(c_heatmap_data, index=ems_samples, 
                                   columns=[k[0] for k in sorted(c_centered.items(), 
                                                               key=lambda x: x[1]/c_total if c_total > 0 else 0, 
                                                               reverse=True)[:top_n]])
        
        sns.heatmap(c_heatmap_df, annot=True, fmt='.4f', cmap='Reds', 
                    cbar_kws={'label': 'Mutation Rate'}, ax=ax)
        ax.set_title(f'Top {top_n} C>T Mutation Rate Heatmap (Uncollapsed)')
        ax.set_xlabel('5mer Context')
        ax.set_ylabel('Sample')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mutation_rate_heatmap_uncollapsed_c.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved uncollapsed mutation rate heatmap with {top_n} 5mers")
        
    elif data_type == 'collapsed':
        # For collapsed data, reverse complement G mutations to C
        all_kmers = set()
        for sample, sample_counts in context_counts.items():
            for kmer in sample_counts.keys():
                if len(kmer) == 5:
                    if kmer[2] == 'C':  # C centered - use as is
                        all_kmers.add(kmer)
                    elif kmer[2] == 'G':  # G centered - reverse complement to C
                        rc_kmer = reverse_complement(kmer)
                        if rc_kmer[2] == 'C':  # Should be C centered after RC
                            all_kmers.add(rc_kmer)
        
        # Get all EMS samples (including 3d/7d)
        ems_samples = [s for s in context_counts.keys() if 'EMS' in s]
        ems_samples = sort_samples_by_ems_number(ems_samples)
        
        # Create heatmap data
        heatmap_data = []
        for sample in ems_samples:
            # Process kmers: C centered as is, G centered reverse complemented to C
            processed_kmers = {}
            for kmer, count in context_counts[sample].items():
                if len(kmer) == 5:
                    if kmer[2] == 'C':  # C centered - use as is
                        processed_kmers[kmer] = count
                    elif kmer[2] == 'G':  # G centered - reverse complement to C
                        rc_kmer = reverse_complement(kmer)
                        if rc_kmer[2] == 'C':  # Should be C centered after RC
                            processed_kmers[rc_kmer] = processed_kmers.get(rc_kmer, 0) + count
            
            total_sample = sum(processed_kmers.values())
            
            # Get top kmers
            sorted_kmers = sorted(processed_kmers.items(), 
                                key=lambda x: x[1]/total_sample if total_sample > 0 else 0, 
                                reverse=True)
            top_kmers = [k[0] for k in sorted_kmers[:top_n]]
            row = [processed_kmers.get(kmer, 0) / total_sample if total_sample > 0 else 0 for kmer in top_kmers]
            heatmap_data.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        heatmap_df = pd.DataFrame(heatmap_data, index=ems_samples, 
                                 columns=[k[0] for k in sorted(processed_kmers.items(), 
                                                             key=lambda x: x[1]/total_sample if total_sample > 0 else 0, 
                                                             reverse=True)[:top_n]])
        
        sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='Reds', 
                    cbar_kws={'label': 'Mutation Rate'}, ax=ax)
        ax.set_title(f'Top {top_n} Mutation Rate Heatmap (Collapsed)')
        ax.set_xlabel('5mer Context')
        ax.set_ylabel('Sample')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mutation_rate_heatmap_collapsed.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved collapsed mutation rate heatmap with {top_n} 5mers")
        
    else:  # c_only
        # Get all EMS samples (including 3d/7d)
        ems_samples = [s for s in context_counts.keys() if 'EMS' in s]
        ems_samples = sort_samples_by_ems_number(ems_samples)
        
        # Calculate average mutation rates to find top kmers
        avg_rates = defaultdict(list)
        genome_c_centered = filter_centered_kmers(genome_counts, 'C')
        
        for sample in ems_samples:
            sample_c_centered = filter_centered_kmers(context_counts[sample], 'C')
            total_sample = sum(sample_c_centered.values())
            for kmer, count in sample_c_centered.items():
                if total_sample > 0:
                    avg_rates[kmer].append(count / total_sample)
        
        # Get top N 5mers by average mutation rate
        means = {k: np.mean(v) for k, v in avg_rates.items() if v}
        top_kmers = [k for k, _ in sorted(means.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        
        # Create heatmap data
        heatmap_data = []
        for sample in ems_samples:
            sample_c_centered = filter_centered_kmers(context_counts[sample], 'C')
            total_sample = sum(sample_c_centered.values())
            
            row = [sample_c_centered.get(kmer, 0) / total_sample if total_sample > 0 else 0 for kmer in top_kmers]
            heatmap_data.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        heatmap_df = pd.DataFrame(heatmap_data, index=ems_samples, columns=top_kmers)
        
        sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='Reds', 
                    cbar_kws={'label': 'Mutation Rate'}, ax=ax)
        ax.set_title(f'Top {len(top_kmers)} Mutation Rate Heatmap (C-only)')
        ax.set_xlabel('5mer Context')
        ax.set_ylabel('Sample')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mutation_rate_heatmap_c_only.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved C-only mutation rate heatmap with {len(top_kmers)} 5mers")


def create_sequence_logo_from_5mers(context_counts, genome_counts, output_dir, 
                                     top_n=3, get_highest=True):
    """
    Create a simple sequence logo from the top N or bottom N 5mers per sample (collapsed data only).
    
    For each sample, identifies the top N or bottom N 5mers by mutation rate, then creates
    a position weight matrix (PWM) from all these sequences and generates a sequence logo.
    
    Args:
        context_counts: Dictionary of sample -> kmer counts
        genome_counts: Dictionary of genome kmer counts
        output_dir: Output directory for plots
        top_n: Number of 5mers to use per sample (default: 3)
        get_highest: If True, get highest rate kmers; if False, get lowest rate kmers
    """
    if not LOGOMAKER_AVAILABLE:
        print("Skipping sequence logo generation (logomaker not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all EMS samples (including 3d/7d)
    ems_samples = [s for s in context_counts.keys() if 'EMS' in s]
    ems_samples = sort_samples_by_ems_number(ems_samples)
    
    if not ems_samples:
        print("No EMS samples found for sequence logo generation")
        return
    
    # Collect top N or bottom N 5mers from each sample (collapsed mode)
    all_selected_5mers = []
    sample_selected_5mers = {}
    
    rate_type = "highest" if get_highest else "lowest"
    print(f"Collecting {rate_type} {top_n} 5mers from each of {len(ems_samples)} samples (collapsed)...")
    
    for sample in ems_samples:
        # Process kmers: C centered as is, G centered reverse complemented to C
        processed_kmers = {}
        for kmer, count in context_counts[sample].items():
            if len(kmer) == 5:
                if kmer[2] == 'C':  # C centered - use as is
                    processed_kmers[kmer] = processed_kmers.get(kmer, 0) + count
                elif kmer[2] == 'G':  # G centered - reverse complement to C
                    rc_kmer = reverse_complement(kmer)
                    if rc_kmer[2] == 'C':
                        processed_kmers[rc_kmer] = processed_kmers.get(rc_kmer, 0) + count
        
        # Calculate normalized mutation rates (enrichment ratios)
        # Normalize by genome frequency to get true enrichment
        total_sample = sum(processed_kmers.values())
        total_genome = sum(genome_counts.values())
        sample_rates = {}
        for kmer, count in processed_kmers.items():
            if kmer in genome_counts and genome_counts[kmer] > 0 and total_sample > 0 and total_genome > 0:
                # Calculate enrichment ratio: observed / expected
                genome_freq = genome_counts[kmer] / total_genome
                expected_count = genome_freq * total_sample
                if expected_count > 0:
                    sample_rates[kmer] = count / expected_count
                else:
                    sample_rates[kmer] = 0
            else:
                # If kmer not in genome or no mutations, set rate to 0
                sample_rates[kmer] = 0
        
        # Get top N or bottom N 5mers for this sample
        sorted_kmers = sorted(sample_rates.items(), key=lambda x: x[1], reverse=get_highest)
        selected_kmers = [kmer for kmer, rate in sorted_kmers[:top_n]]
        
        sample_selected_5mers[sample] = selected_kmers
        all_selected_5mers.extend(selected_kmers)
    
    if not all_selected_5mers:
        print("No 5mers found for sequence logo generation")
        return
    
    # Total number of sequences
    num_sequences = len(all_selected_5mers)
    
    # Create position weight matrix (PWM)
    # Count bases at each position across all sequences
    pwm_counts = {pos: {'A': 0, 'T': 0, 'G': 0, 'C': 0} for pos in range(5)}
    
    for kmer in all_selected_5mers:
        if len(kmer) == 5 and all(base in 'ATGC' for base in kmer):
            for pos, base in enumerate(kmer):
                if base in pwm_counts[pos]:
                    pwm_counts[pos][base] += 1
    
    # Convert counts to frequencies (each position sums to 1.0)
    bases = ['A', 'T', 'G', 'C']
    pwm_freqs = {}
    for pos in range(5):
        total_count = sum(pwm_counts[pos].values())
        if total_count > 0:
            pwm_freqs[pos] = {base: count / total_count 
                             for base, count in pwm_counts[pos].items()}
        else:
            pwm_freqs[pos] = {base: 0.25 for base in bases}
    
    # Calculate information content (bits) for each position (for reporting)
    # IC = sum(p * log2(p / bg)) where bg is background frequency (0.25 for uniform)
    bg_freq = 0.25  # Uniform background
    pwm_ic = {}
    for pos in range(5):
        ic_total = sum(pwm_freqs[pos][base] * (np.log2(pwm_freqs[pos][base] / bg_freq) 
                                               if pwm_freqs[pos][base] > 0 else 0) 
                       for base in bases)
        pwm_ic[pos] = ic_total
    
    # Create DataFrame for logomaker (positions x bases)
    logo_df = pd.DataFrame(
        [[pwm_freqs[pos][base] for base in bases] for pos in range(5)],
        columns=bases,
        index=range(5)
    )
    
    # Create sequence logo
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create logo - logomaker will calculate information content from probabilities
    logo = logomaker.Logo(logo_df, ax=ax, color_scheme='classic', 
                          font_name='Arial', show_spines=True, 
                          vpad=0.05, width=0.8)
    
    # Get the actual y-axis limits from the logo and set them properly
    y_min, y_max = ax.get_ylim()
    # Set y-axis to show full range (0 to max height of letters)
    ax.set_ylim(0, y_max)
    
    ax.set_xlabel('Position in 5mer', fontsize=16, fontweight='bold')
    ax.set_ylabel('Bits', fontsize=16, fontweight='bold')
    title_text = f'Sequence Logo from {rate_type.capitalize()} {top_n} 5mers per Sample (Collapsed)'
    ax.set_title(title_text, fontsize=18, fontweight='bold', pad=15)
    
    # Add text annotation with sample info
    info_text = f"Based on {rate_type} {top_n} 5mers from {len(ems_samples)} samples\n"
    info_text += f"Total sequences: {num_sequences}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Create output filename based on highest/lowest
    suffix = "highest" if get_highest else "lowest"
    output_file = os.path.join(output_dir, f'sequence_logo_{suffix}{top_n}_collapsed.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Sequence logo saved: {output_file}")
    
    # Also create a detailed text file with all selected 5mers per sample
    detail_file = os.path.join(output_dir, f'sequence_logo_{suffix}{top_n}_collapsed_details.txt')
    with open(detail_file, 'w') as f:
        f.write(f"{rate_type.capitalize()} {top_n} 5mers per Sample for Sequence Logo (collapsed)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total sequences used: {num_sequences}\n")
        f.write(f"Number of samples: {len(ems_samples)}\n\n")
        
        f.write("Position Weight Matrix (Frequency):\n")
        f.write("-" * 80 + "\n")
        f.write("Position |     A     |     T     |     G     |     C     | IC (bits)\n")
        f.write("-" * 80 + "\n")
        for pos in range(5):
            f.write(f"   {pos}    | {pwm_freqs[pos]['A']:8.4f}  | {pwm_freqs[pos]['T']:8.4f}  | "
                   f"{pwm_freqs[pos]['G']:8.4f}  | {pwm_freqs[pos]['C']:8.4f}  | {pwm_ic[pos]:8.4f}\n")
        
        f.write(f"\n\n{rate_type.capitalize()} 5mers by Sample:\n")
        f.write("-" * 80 + "\n")
        for sample in ems_samples:
            f.write(f"\n{sample}:\n")
            for i, kmer in enumerate(sample_selected_5mers[sample], 1):
                f.write(f"  {i}. {kmer}\n")
    
    print(f"Details saved: {detail_file}")


def create_sequence_logo_from_aggregate_rates(aggregate_rates, output_dir, top_n=3, get_highest=True):
    """
    Create a sequence logo from aggregate depth-normalized rates.
    
    Args:
        aggregate_rates: dict mapping kmer -> aggregate depth-normalized rate
        output_dir: output directory
        top_n: number of top 5mers to use (default: 3)
        get_highest: if True, get highest rate kmers; if False, get lowest rate kmers
    """
    if not LOGOMAKER_AVAILABLE:
        print("Skipping sequence logo generation (logomaker not available)")
        return
    
    if not aggregate_rates:
        print("No aggregate rates available for sequence logo generation")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to C-centered kmers only (collapsed)
    c_centered_rates = {kmer: rate for kmer, rate in aggregate_rates.items() 
                       if len(kmer) == 5 and kmer[2] == 'C'}
    
    if not c_centered_rates:
        print("No C-centered 5mers available for sequence logo generation")
        return
    
    # Get top N or bottom N kmers by rate
    rate_type = "highest" if get_highest else "lowest"
    sorted_kmers = sorted(c_centered_rates.items(), key=lambda x: x[1], reverse=get_highest)
    selected_kmers = [kmer for kmer, rate in sorted_kmers[:top_n]]
    
    if not selected_kmers:
        print("No 5mers found for sequence logo generation")
        return
    
    num_sequences = len(selected_kmers)
    
    # Create position weight matrix (PWM)
    pwm_counts = {pos: {'A': 0, 'T': 0, 'G': 0, 'C': 0} for pos in range(5)}
    
    for kmer in selected_kmers:
        if len(kmer) == 5 and all(base in 'ATGC' for base in kmer):
            for pos, base in enumerate(kmer):
                if base in pwm_counts[pos]:
                    pwm_counts[pos][base] += 1
    
    # Convert counts to frequencies
    bases = ['A', 'T', 'G', 'C']
    pwm_freqs = {}
    for pos in range(5):
        total_count = sum(pwm_counts[pos].values())
        if total_count > 0:
            pwm_freqs[pos] = {base: count / total_count 
                             for base, count in pwm_counts[pos].items()}
        else:
            pwm_freqs[pos] = {base: 0.25 for base in bases}
    
    # Calculate information content
    bg_freq = 0.25
    pwm_ic = {}
    for pos in range(5):
        ic_total = sum(pwm_freqs[pos][base] * (np.log2(pwm_freqs[pos][base] / bg_freq) 
                                               if pwm_freqs[pos][base] > 0 else 0) 
                       for base in bases)
        pwm_ic[pos] = ic_total
    
    # Create DataFrame for logomaker
    logo_df = pd.DataFrame(
        [[pwm_freqs[pos][base] for base in bases] for pos in range(5)],
        columns=bases,
        index=range(5)
    )
    
    # Create sequence logo
    fig, ax = plt.subplots(figsize=(12, 4))
    
    logo = logomaker.Logo(logo_df, ax=ax, color_scheme='classic', 
                          font_name='Arial', show_spines=True, 
                          vpad=0.05, width=0.8)
    
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max)
    
    ax.set_xlabel('Position in 5mer', fontsize=16, fontweight='bold')
    ax.set_ylabel('Bits', fontsize=16, fontweight='bold')
    title_text = f'Sequence Logo from {rate_type.capitalize()} {top_n} 5mers (Aggregate Depth-Normalized)'
    ax.set_title(title_text, fontsize=18, fontweight='bold', pad=15)
    
    # Add text annotation
    info_text = f"Based on {rate_type} {top_n} 5mers from aggregate rates\n"
    info_text += f"Total sequences: {num_sequences}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    suffix = "highest" if get_highest else "lowest"
    output_file = os.path.join(output_dir, f'sequence_logo_{suffix}{top_n}_aggregate.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Sequence logo saved: {output_file}")
    
    # Create detail file
    detail_file = os.path.join(output_dir, f'sequence_logo_{suffix}{top_n}_aggregate_details.txt')
    with open(detail_file, 'w') as f:
        f.write(f"{rate_type.capitalize()} {top_n} 5mers for Sequence Logo (Aggregate)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total sequences used: {num_sequences}\n\n")
        
        f.write("Position Weight Matrix (Frequency):\n")
        f.write("-" * 80 + "\n")
        f.write("Position |     A     |     T     |     G     |     C     | IC (bits)\n")
        f.write("-" * 80 + "\n")
        for pos in range(5):
            f.write(f"   {pos}    | {pwm_freqs[pos]['A']:8.4f}  | {pwm_freqs[pos]['T']:8.4f}  | "
                   f"{pwm_freqs[pos]['G']:8.4f}  | {pwm_freqs[pos]['C']:8.4f}  | {pwm_ic[pos]:8.4f}\n")
        
        f.write(f"\n\n{rate_type.capitalize()} 5mers:\n")
        f.write("-" * 80 + "\n")
        for i, (kmer, rate) in enumerate(sorted_kmers[:top_n], 1):
            f.write(f"  {i}. {kmer} (rate: {rate:.2e})\n")
    
    print(f"Details saved: {detail_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot collapsed 5mer mutation rate analysis with individual samples and aggregate depth-normalized rates. Includes sequence logos for both. Applies exclusion mask to exclude sites appearing in multiple controls."
    )
    parser.add_argument("--kmers-dir", required=True,
                        help="Directory containing *_5mer_contexts.json files")
    parser.add_argument("--genome-fasta", required=True,
                        help="Path to genome FASTA file (supports .gz files)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for plots")
    parser.add_argument("--counts-dir", type=str, required=True,
                        help="Directory containing .counts files for depth normalization. Required for aggregate analysis.")
    parser.add_argument("--exclusion-mask", type=str, default=None,
                        help="Optional: Path to exclusion mask TSV file (chrom\tpos format). If not provided, will build from controls (sites appearing in >1 control).")
    parser.add_argument("--min-alt", type=int, default=1,
                        help="Minimum alt allele count required for a site to be included in all analysis. Used for both exclusion mask building and depth filtering (default: 1)")
    parser.add_argument("--gene-only", action="store_true",
                        help="Use only gene mutations instead of total mutations")
    parser.add_argument("--sequence-logo-top-n", type=int, default=3,
                        help="Number of top 5mers to use for sequence logo (default: 3)")
    
    args = parser.parse_args()
    
    print("Loading 5mer context counts...")
    context_counts = load_5mer_contexts(args.kmers_dir, use_gene_only=args.gene_only, no_strand_collapse=True)
    
    if not context_counts:
        print("No 5mer context data found!")
        return
    
    print("Generating genome 5mer counts...")
    genome_counts = generate_genome_5mer_counts_separate(args.genome_fasta)
    # Combine C and G centered for compatibility with existing functions
    combined_genome_counts = {}
    combined_genome_counts.update(genome_counts['c_centered'])
    combined_genome_counts.update(genome_counts['g_centered'])
    print(f"Generated {len(genome_counts['c_centered'])} C-centered and {len(genome_counts['g_centered'])} G-centered genome 5mers")
    
    # Load or build exclusion mask
    print("\n=== EXCLUSION MASK ===")
    exclusion_mask = None
    if args.exclusion_mask:
        print(f"Loading exclusion mask from {args.exclusion_mask}...")
        exclusion_mask = load_exclusion_mask_tsv(args.exclusion_mask)
        if exclusion_mask:
            print(f"  Loaded {len(exclusion_mask)} sites to exclude")
        else:
            print("  Warning: No sites found in exclusion mask file")
    else:
        print("Building exclusion mask from control files...")
        exclusion_mask = build_exclusion_mask_from_controls(args.counts_dir, min_alt=args.min_alt)
        if exclusion_mask:
            print(f"  Built exclusion mask: {len(exclusion_mask)} sites to exclude")
        else:
            print("  Warning: No exclusion mask built (no control files found or no sites to exclude)")
    
    # Load depth information and calculate aggregate rates
    print("\n=== DEPTH NORMALIZATION ===")
    print(f"Loading depth information from {args.counts_dir}...")
    sample_depths = load_sample_depths(args.counts_dir, exclusion_mask=exclusion_mask, min_alt=args.min_alt)
    
    # Debug: Check sample name matching
    print(f"\nSample name matching check:")
    print(f"  Samples in context_counts: {sorted(context_counts.keys())[:5]}...")
    print(f"  Samples with depth info: {sorted(sample_depths.keys())[:5]}...")
    matching_samples = set(context_counts.keys()) & set(sample_depths.keys())
    print(f"  Matching samples: {len(matching_samples)}")
    if len(matching_samples) < len(context_counts.keys()):
        missing = set(context_counts.keys()) - set(sample_depths.keys())
        print(f"  Warning: {len(missing)} samples in context_counts have no depth info: {list(missing)[:5]}...")
    
    aggregate_rates = None
    aggregate_genome_normalized_rates = None
    if sample_depths:
        print("\nCalculating aggregate depth-normalized rates with background subtraction...")
        aggregate_rates = calculate_aggregate_depth_normalized_rates(
            context_counts, sample_depths, use_collapsed=True
        )
        if aggregate_rates:
            print(f"Successfully calculated aggregate rates for {len(aggregate_rates)} kmers")
            print(f"  Sample rates: min={min(aggregate_rates.values()):.2e}, max={max(aggregate_rates.values()):.2e}")
        else:
            print("Warning: Failed to calculate aggregate rates (returned empty dict)")
        
        print("\nCalculating aggregate depth and genome-frequency-normalized rates with background subtraction...")
        aggregate_genome_normalized_rates = calculate_aggregate_depth_and_genome_normalized_rates(
            context_counts, sample_depths, combined_genome_counts, use_collapsed=True
        )
        if aggregate_genome_normalized_rates:
            print(f"Successfully calculated aggregate genome-normalized rates for {len(aggregate_genome_normalized_rates)} kmers")
            print(f"  Sample rates: min={min(aggregate_genome_normalized_rates.values()):.2e}, max={max(aggregate_genome_normalized_rates.values()):.2e}")
        else:
            print("Warning: Failed to calculate aggregate genome-normalized rates (returned empty dict)")
    else:
        print("Warning: No depth information found (sample_depths is empty)")
    
    print("\n=== COLLAPSED ANALYSIS ===")
    
    # Always create individual, aggregate, and aggregate genome-normalized versions
    
    # 1. Individual samples collapsed multiplot
    print("Creating collapsed mutation rate multiplot (individual samples)...")
    plot_collapsed_multiplot(context_counts, combined_genome_counts, args.output_dir, aggregate_rates=None)
    print("  ✓ Individual samples multiplot saved")
    
    # 2. Aggregate collapsed multiplot (depth-normalized only)
    print("\nCreating collapsed mutation rate multiplot (aggregate depth-normalized)...")
    if aggregate_rates and len(aggregate_rates) > 0:
        print(f"  Aggregate rates available: {len(aggregate_rates)} kmers")
        plot_collapsed_multiplot(context_counts, combined_genome_counts, args.output_dir, aggregate_rates=aggregate_rates)
        print("  ✓ Aggregate multiplot saved")
    else:
        print("  ✗ ERROR: Cannot create aggregate multiplot")
        if not aggregate_rates:
            print("    - aggregate_rates is None (check depth loading and sample name matching above)")
        elif len(aggregate_rates) == 0:
            print("    - aggregate_rates is empty dict (no kmers found)")
        else:
            print(f"    - Unknown issue (aggregate_rates has {len(aggregate_rates)} items)")
    
    # 3. Aggregate genome-normalized collapsed multiplot (depth + genome-frequency normalized)
    print("\nCreating collapsed mutation rate multiplot (aggregate depth and genome-frequency-normalized)...")
    if aggregate_genome_normalized_rates and len(aggregate_genome_normalized_rates) > 0:
        print(f"  Aggregate genome-normalized rates available: {len(aggregate_genome_normalized_rates)} kmers")
        plot_collapsed_multiplot(context_counts, combined_genome_counts, args.output_dir, aggregate_rates=aggregate_genome_normalized_rates)
        # Save with different filename to distinguish
        import shutil
        src_file = os.path.join(args.output_dir, 'collapsed_mutation_rates_multiplot_aggregate.png')
        dst_file = os.path.join(args.output_dir, 'collapsed_mutation_rates_multiplot_aggregate_genome_normalized.png')
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"  ✓ Aggregate genome-normalized multiplot saved")
    else:
        print("  ✗ ERROR: Cannot create aggregate genome-normalized multiplot")
        if not aggregate_genome_normalized_rates:
            print("    - aggregate_genome_normalized_rates is None")
        elif len(aggregate_genome_normalized_rates) == 0:
            print("    - aggregate_genome_normalized_rates is empty dict")
        else:
            print(f"    - Unknown issue (aggregate_genome_normalized_rates has {len(aggregate_genome_normalized_rates)} items)")
    
    # Sequence logo generation
    if LOGOMAKER_AVAILABLE:
        print("\n=== SEQUENCE LOGO GENERATION ===")
        
        # Individual samples sequence logos
        print("Creating sequence logos from individual samples...")
        print("  Creating highest rate 5mers logo...")
        create_sequence_logo_from_5mers(context_counts, combined_genome_counts, args.output_dir,
                                        top_n=args.sequence_logo_top_n, get_highest=True)
        print("  Creating lowest rate 5mers logo...")
        create_sequence_logo_from_5mers(context_counts, combined_genome_counts, args.output_dir,
                                        top_n=5, get_highest=False)
        print("  ✓ Individual samples sequence logos saved")
        
        # Aggregate sequence logos (depth-normalized only)
        print("\nCreating sequence logos from aggregate rates (depth-normalized)...")
        if aggregate_rates and len(aggregate_rates) > 0:
            print("  Creating highest rate 5mers logo...")
            create_sequence_logo_from_aggregate_rates(aggregate_rates, args.output_dir,
                                                     top_n=args.sequence_logo_top_n, get_highest=True)
            print("  Creating lowest rate 5mers logo...")
            create_sequence_logo_from_aggregate_rates(aggregate_rates, args.output_dir,
                                                     top_n=5, get_highest=False)
            print("  ✓ Aggregate sequence logos saved")
        else:
            print("  ✗ ERROR: Cannot create aggregate sequence logos")
            if not aggregate_rates:
                print("    - aggregate_rates is None (check depth loading and sample name matching above)")
            elif len(aggregate_rates) == 0:
                print("    - aggregate_rates is empty dict (no kmers found)")
            else:
                print(f"    - Unknown issue (aggregate_rates has {len(aggregate_rates)} items)")
        
        # Aggregate genome-normalized sequence logos (depth + genome-frequency normalized)
        print("\nCreating sequence logos from aggregate rates (depth and genome-frequency-normalized)...")
        if aggregate_genome_normalized_rates and len(aggregate_genome_normalized_rates) > 0:
            print("  Creating highest rate 5mers logo...")
            create_sequence_logo_from_aggregate_rates(aggregate_genome_normalized_rates, args.output_dir,
                                                     top_n=args.sequence_logo_top_n, get_highest=True)
            # Rename output files to distinguish
            import shutil
            src_highest = os.path.join(args.output_dir, f'sequence_logo_highest{args.sequence_logo_top_n}_aggregate.png')
            dst_highest = os.path.join(args.output_dir, f'sequence_logo_highest{args.sequence_logo_top_n}_aggregate_genome_normalized.png')
            if os.path.exists(src_highest):
                shutil.copy(src_highest, dst_highest)
                shutil.copy(src_highest.replace('.png', '.pdf'), dst_highest.replace('.png', '.pdf'))
                shutil.copy(src_highest.replace('.png', '_details.txt'), dst_highest.replace('.png', '_details.txt'))
            
            print("  Creating lowest rate 5mers logo...")
            create_sequence_logo_from_aggregate_rates(aggregate_genome_normalized_rates, args.output_dir,
                                                     top_n=5, get_highest=False)
            # Rename output files to distinguish
            src_lowest = os.path.join(args.output_dir, 'sequence_logo_lowest5_aggregate.png')
            dst_lowest = os.path.join(args.output_dir, 'sequence_logo_lowest5_aggregate_genome_normalized.png')
            if os.path.exists(src_lowest):
                shutil.copy(src_lowest, dst_lowest)
                shutil.copy(src_lowest.replace('.png', '.pdf'), dst_lowest.replace('.png', '.pdf'))
                shutil.copy(src_lowest.replace('.png', '_details.txt'), dst_lowest.replace('.png', '_details.txt'))
            
            print("  ✓ Aggregate genome-normalized sequence logos saved")
        else:
            print("  ✗ ERROR: Cannot create aggregate genome-normalized sequence logos")
            if not aggregate_genome_normalized_rates:
                print("    - aggregate_genome_normalized_rates is None")
            elif len(aggregate_genome_normalized_rates) == 0:
                print("    - aggregate_genome_normalized_rates is empty dict")
            else:
                print(f"    - Unknown issue (aggregate_genome_normalized_rates has {len(aggregate_genome_normalized_rates)} items)")
    else:
        print("\nSkipping sequence logo generation (logomaker not available)")
    
    print(f"\n=== OUTPUT SUMMARY ===")
    print(f"All plots saved to {args.output_dir}")
    print("\nIndividual samples analysis:")
    print("  - collapsed_mutation_rates_multiplot_individual.png")
    print(f"  - sequence_logo_highest{args.sequence_logo_top_n}_collapsed.png")
    print("  - sequence_logo_lowest5_collapsed.png")
    
    if aggregate_rates and len(aggregate_rates) > 0:
        print("\nAggregate depth-normalized analysis:")
        print("  - collapsed_mutation_rates_multiplot_aggregate.png")
        print(f"  - sequence_logo_highest{args.sequence_logo_top_n}_aggregate.png")
        print("  - sequence_logo_lowest5_aggregate.png")
    else:
        print("\n✗ Aggregate depth-normalized analysis NOT generated (see errors above)")
    
    if aggregate_genome_normalized_rates and len(aggregate_genome_normalized_rates) > 0:
        print("\nAggregate depth and genome-frequency-normalized analysis:")
        print("  - collapsed_mutation_rates_multiplot_aggregate_genome_normalized.png")
        print(f"  - sequence_logo_highest{args.sequence_logo_top_n}_aggregate_genome_normalized.png")
        print("  - sequence_logo_lowest5_aggregate_genome_normalized.png")
    else:
        print("\n✗ Aggregate genome-frequency-normalized analysis NOT generated (see errors above)")


if __name__ == "__main__":
    main()
