#!/usr/bin/env python3
"""
Plot 5mer mutation rate analysis for strand-collapsed (C-centered only) mutations.

This script creates various plots for analyzing 5mer context mutation rates in EMS mutations,
focusing on normalized mutation rates rather than enrichment ratios.
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
    """Extract sample ID from filename."""
    basename = os.path.basename(filename)
    # Remove _5mer_contexts.json extension
    sample_id = basename.replace('_5mer_contexts.json', '')
    return sample_id


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


def plot_collapsed_multiplot(context_counts, genome_counts, output_dir):
    """Create collapsed multiplot with G mutations reverse complemented to C centered."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with three panels - larger size for publication
    fig = plt.figure(figsize=(20, 18))
    # Adjust spacing: reduce top margin, increase space between panels
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 2], hspace=0.45, top=0.97, bottom=0.08)
    
    # Panel A: 3mer grouped bar (top) - G mutations reverse complemented to C
    ax_top = fig.add_subplot(gs[0])
    plot_3mer_grouped_bar_collapsed(context_counts, genome_counts, ax_top, remove_xaxis=True)
    
    # Panel B: Top 16 5mers grouped bar (middle) - G mutations reverse complemented to C
    ax_middle = fig.add_subplot(gs[1])
    plot_top_5mer_grouped_bar_collapsed(context_counts, genome_counts, ax_middle, top_n=16, remove_xaxis=True)
    
    # Panel C: 5mer average signature (bottom) - G mutations reverse complemented to C
    ax_bottom = fig.add_subplot(gs[2])
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
    
    plt.savefig(os.path.join(output_dir, 'collapsed_mutation_rates_multiplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved collapsed mutation rates multiplot with G mutations reverse complemented to C centered")


def plot_c_only_multiplot(context_counts, genome_counts, output_dir):
    """Create C-only multiplot with only C centered mutations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with three panels - larger size for publication
    fig = plt.figure(figsize=(20, 18))
    # Adjust spacing: reduce top margin, increase space between panels
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 2], hspace=0.45, top=0.97, bottom=0.08)
    
    # Panel A: 3mer grouped bar (top) - C centered only
    ax_top = fig.add_subplot(gs[0])
    plot_3mer_grouped_bar_c_only(context_counts, genome_counts, ax_top, remove_xaxis=True)
    
    # Panel B: Top 16 5mers grouped bar (middle) - C centered only
    ax_middle = fig.add_subplot(gs[1])
    plot_top_5mer_grouped_bar_c_only(context_counts, genome_counts, ax_middle, top_n=16, remove_xaxis=True)
    
    # Panel C: 5mer average signature (bottom) - C centered only
    ax_bottom = fig.add_subplot(gs[2])
    plot_5mer_average_signature_c_only(context_counts, genome_counts, ax_bottom)
    
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
    
    print("Saved C-only mutation rates multiplot with C centered mutations only")


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
    
    # Get EMS samples (excluding 3d/7d)
    ems_samples = [s for s in sample_c_3mer_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
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
    # Get EMS samples (excluding 3d/7d) and sort by EMS number
    ems_samples = [s for s in context_counts.keys() if 'EMS' in s and '3d' not in s and '7d' not in s]
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
        if '3d' in sample or '7d' in sample:
            continue
        for kmer in sample_counts.keys():
            if len(kmer) == 5 and kmer[2] == 'C':
                c_kmers.add(kmer)
            elif len(kmer) == 5 and kmer[2] == 'G':
                g_kmers.add(kmer)
    
    # Calculate mutation rates for C centered
    c_avg_rates = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
        if '3d' in sample or '7d' in sample:
            continue
        c_centered = {k: v for k, v in kmer_counts.items() if len(k) == 5 and k[2] == 'C'}
        c_total = sum(c_centered.values())
        
        for kmer in c_kmers:
            if c_total > 0:
                c_avg_rates[kmer].append(c_centered.get(kmer, 0) / c_total)
    
    # Calculate mutation rates for G centered
    g_avg_rates = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
        if '3d' in sample or '7d' in sample:
            continue
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
    
    # Get EMS samples (excluding 3d/7d)
    ems_samples = [s for s in sample_3mer_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
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
        if '3d' in sample or '7d' in sample:
            continue
        for kmer in sample_counts.keys():
            if len(kmer) == 5:
                if kmer[2] == 'C':  # C centered - use as is
                    all_kmers.add(kmer)
                elif kmer[2] == 'G':  # G centered - reverse complement to C
                    rc_kmer = reverse_complement(kmer)
                    if rc_kmer[2] == 'C':  # Should be C centered after RC
                        all_kmers.add(rc_kmer)
    
    # Average mutation rates across samples (excluding 3d/7d samples)
    avg_rates = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
        # Skip 3d/7d samples
        if '3d' in sample or '7d' in sample:
            continue
            
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
    
    # Get EMS samples (excluding 3d/7d)
    ems_samples = [s for s in sample_3mer_counts if 'EMS' in s and '3d' not in s and '7d' not in s]
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


def plot_5mer_average_signature_c_only(context_counts, genome_counts, ax):
    """Create 5mer average signature plot for C-only data (C centered only)."""
    # Filter to C-centered kmers only
    genome_c_centered = filter_centered_kmers(genome_counts, 'C')
    
    # Collect all C-centered kmers across samples (excluding 3d/7d)
    all_kmers = set()
    for sample, sample_counts in context_counts.items():
        if '3d' in sample or '7d' in sample:
            continue
        sample_c_centered = filter_centered_kmers(sample_counts, 'C')
        all_kmers.update(sample_c_centered.keys())
    
    # Average mutation rates across samples (excluding 3d/7d samples)
    avg_rates = defaultdict(list)
    for sample, kmer_counts in context_counts.items():
        # Skip 3d/7d samples
        if '3d' in sample or '7d' in sample:
            continue
            
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
            if '3d' in sample or '7d' in sample:
                continue
            for kmer in sample_counts.keys():
                if len(kmer) == 5 and kmer[2] == 'C':
                    c_kmers.add(kmer)
        
        # Get EMS samples (excluding 3d/7d)
        ems_samples = [s for s in context_counts.keys() if 'EMS' in s and '3d' not in s and '7d' not in s]
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
            if '3d' in sample or '7d' in sample:
                continue
            for kmer in sample_counts.keys():
                if len(kmer) == 5:
                    if kmer[2] == 'C':  # C centered - use as is
                        all_kmers.add(kmer)
                    elif kmer[2] == 'G':  # G centered - reverse complement to C
                        rc_kmer = reverse_complement(kmer)
                        if rc_kmer[2] == 'C':  # Should be C centered after RC
                            all_kmers.add(rc_kmer)
        
        # Get EMS samples (excluding 3d/7d)
        ems_samples = [s for s in context_counts.keys() if 'EMS' in s and '3d' not in s and '7d' not in s]
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
        # Get EMS samples (excluding 3d/7d)
        ems_samples = [s for s in context_counts.keys() if 'EMS' in s and '3d' not in s and '7d' not in s]
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot 5mer mutation rate analysis with three different multiplot types: uncollapsed, collapsed, and C-only"
    )
    parser.add_argument("--kmers-dir", required=True,
                        help="Directory containing *_5mer_contexts.json files")
    parser.add_argument("--genome-fasta", required=True,
                        help="Path to genome FASTA file (supports .gz files)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for plots")
    parser.add_argument("--gene-only", action="store_true",
                        help="Use only gene mutations instead of total mutations")
    
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
    
    print("Creating three different mutation rate multiplot files...")
    
    # 1. Uncollapsed multiplot (G and C centered mutations)
    print("Creating uncollapsed mutation rate multiplot (G and C centered mutations)...")
    plot_uncollapsed_multiplot(context_counts, combined_genome_counts, args.output_dir)
    
    # 2. Collapsed multiplot (G mutations reverse complemented to C centered)
    print("Creating collapsed mutation rate multiplot (G mutations reverse complemented to C centered)...")
    plot_collapsed_multiplot(context_counts, combined_genome_counts, args.output_dir)
    
    # 3. C-only multiplot (C mutations only)
    print("Creating C-only mutation rate multiplot (C mutations only)...")
    plot_c_only_multiplot(context_counts, combined_genome_counts, args.output_dir)
    
    # Create analysis subfolders
    uncollapsed_analysis_dir = os.path.join(args.output_dir, 'analysis_uncollapsed')
    collapsed_analysis_dir = os.path.join(args.output_dir, 'analysis_collapsed')
    
    os.makedirs(uncollapsed_analysis_dir, exist_ok=True)
    os.makedirs(collapsed_analysis_dir, exist_ok=True)
    
    # Additional statistical analyses
    print("\nPerforming additional statistical analyses...")
    
    # Uncollapsed analysis (raw G and C centered data)
    print("\n=== UNCOLLAPSED ANALYSIS ===")
    print("Creating uncollapsed mutation rate heatmap...")
    plot_mutation_rate_heatmap(context_counts, combined_genome_counts, uncollapsed_analysis_dir, 
                               data_type='uncollapsed')
    
    # Collapsed analysis (G mutations reverse complemented to C)
    print("\n=== COLLAPSED ANALYSIS ===")
    print("Creating collapsed mutation rate heatmap...")
    plot_mutation_rate_heatmap(context_counts, combined_genome_counts, collapsed_analysis_dir, 
                               data_type='collapsed')
    
    # General analyses (using C-only data for compatibility with existing functions)
    print("\n=== GENERAL ANALYSES ===")
    print("Plotting mutation rate distribution...")
    plot_mutation_rate_distribution(context_counts, combined_genome_counts, args.output_dir)
    
    print("Creating C-only mutation rate heatmap...")
    plot_mutation_rate_heatmap(context_counts, combined_genome_counts, args.output_dir, 
                               data_type='c_only')
    
    print(f"\nAll plots and analyses saved to {args.output_dir}")
    print(f"Uncollapsed analysis saved to {uncollapsed_analysis_dir}")
    print(f"Collapsed analysis saved to {collapsed_analysis_dir}")


if __name__ == "__main__":
    main()
