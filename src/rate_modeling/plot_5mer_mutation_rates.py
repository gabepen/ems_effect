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
from matplotlib.ticker import LogLocator, SymmetricalLogLocator
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

def reverse_complement(seq):
    """Get reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def predict_5mer_rates_from_model(model_dir, counts_dir, genome_fasta, exclusion_mask=None):
    """
    Use fitted 5mer model to predict rates at site level, then aggregate by 5mer.
    This matches how observed rates are calculated - aggregate counts, then calculate rates.
    
    Args:
        model_dir: Path to directory containing fitted models (e.g., output_dir/fitted_models)
        counts_dir: Directory containing .counts files
        genome_fasta: Path to genome FASTA file
        exclusion_mask: Optional exclusion mask file path
    
    Returns:
        dict mapping kmer -> {'rate': float} (aggregated from site-level predictions)
        or None if model cannot be loaded
    """
    try:
        import sys
        import os
        
        # Add current directory to path to import modules
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import required modules
        try:
            from rate_modeling.load_saved_models import load_saved_models, prepare_data_for_model
            from rate_modeling.sequence_bias_modeling_sitelevel import load_site_level_data
        except ImportError:
            from load_saved_models import load_saved_models, prepare_data_for_model
            from sequence_bias_modeling_sitelevel import load_site_level_data
        
        # Load model
        models, metadata = load_saved_models(model_dir)
        if '5mer' not in models:
            print(f"Warning: 5mer model not found in {model_dir}")
            return None
        
        model = models['5mer']
        
        # Load site-level data (same as model was trained on)
        print(f"Loading site-level data from {counts_dir}...")
        df = load_site_level_data(counts_dir, genome_fasta, exclusion_mask)
        if df is None or len(df) == 0:
            print("Warning: No site-level data loaded")
            return None
        
        print(f"  Loaded {len(df):,} sites")
        
        # Prepare features for 5mer model
        df['log_depth'] = np.log(df['depth'])
        X, feature_cols = prepare_data_for_model(df, '5mer', metadata)
        
        # Add constant term (same as model was fit with)
        import statsmodels.api as sm
        X_design = sm.add_constant(X, has_constant='add')
        offset = df['log_depth'].values
        
        # Predict for all sites: control (treatment=0) and treated (treatment=1)
        print("Predicting rates at site level...")
        X_control = X_design.copy()
        X_control['treatment'] = 0
        pred_control = model.predict(X_control, offset=offset)
        
        X_treated = X_design.copy()
        X_treated['treatment'] = 1
        pred_treated = model.predict(X_treated, offset=offset)
        
        # Convert predictions to rates (predictions are expected counts, divide by depth)
        # Ensure Series alignment
        if not isinstance(pred_control, pd.Series):
            pred_control = pd.Series(pred_control, index=df.index)
        if not isinstance(pred_treated, pd.Series):
            pred_treated = pd.Series(pred_treated, index=df.index)
        
        rate_control = pred_control / df['depth']
        rate_treated = pred_treated / df['depth']
        rate_diff = rate_treated - rate_control
        
        # Aggregate by 5mer (C-centered only, matching observed rates)
        # Sum rates per 5mer, then divide by number of sites (mean rate)
        kmer_rate_sums = defaultdict(float)
        kmer_site_counts = defaultdict(int)
        
        for idx in df.index:
            kmer = df.loc[idx, 'kmer5']
            if pd.isna(kmer) or len(kmer) != 5 or kmer[2] != 'C':
                continue
            kmer_rate_sums[kmer] += rate_diff.loc[idx]
            kmer_site_counts[kmer] += 1
        
        # Calculate mean rate per 5mer
        model_rates = {}
        for kmer in kmer_rate_sums:
            if kmer_site_counts[kmer] > 0:
                model_rates[kmer] = {
                    'rate': kmer_rate_sums[kmer] / kmer_site_counts[kmer]
                }
        
        print(f"Predicted rates for {len(model_rates)} 5mers")
        if model_rates:
            rate_vals = [r['rate'] for r in model_rates.values()]
            print(f"  Rate range: {min(rate_vals):.2e} to {max(rate_vals):.2e}")
        
        return model_rates
        
    except Exception as e:
        print(f"Warning: Could not predict rates from 5mer model: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_5mer_enrichment_rates(aggregate_rates, output_dir, plot_prefix='observed'):
    """
    Create 5mer enrichment rate plots from aggregate rates.
    
    Args:
        aggregate_rates: dict mapping kmer -> rate (depth-normalized, treated - control)
        output_dir: output directory
        plot_prefix: prefix for output filenames (default: 'observed')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not aggregate_rates or len(aggregate_rates) == 0:
        print(f"ERROR: aggregate_rates is empty!")
        return
    
    # Filter to C-centered kmers only
    c_centered_rates = {kmer: rate for kmer, rate in aggregate_rates.items() 
                       if len(kmer) == 5 and kmer[2] == 'C'}
    
    if not c_centered_rates:
        print(f"ERROR: No C-centered 5mers found!")
        return
    
    # Sort by rate (descending)
    sorted_kmers = sorted(c_centered_rates.items(), key=lambda x: x[1], reverse=True)
    kmers = [k[0] for k in sorted_kmers]
    rates = [k[1] for k in sorted_kmers]
    
    # Create multiplot with three panels
    fig = plt.figure(figsize=(20, 18))
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 2], hspace=0.45, top=0.97, bottom=0.08)
    
    # Panel A: 3mer grouped bar
    ax_top = fig.add_subplot(gs[0])
    trimer_rates = defaultdict(float)
    for kmer, rate in c_centered_rates.items():
        if len(kmer) == 5 and kmer[2] == 'C':
            trimer = kmer[1:4]
            trimer_rates[trimer] += rate
    
    all_trimers = sorted(trimer_rates.keys())
    trimer_rate_vals = [trimer_rates[t] for t in all_trimers]
    
    x_pos = np.arange(len(all_trimers))
    ax_top.bar(x_pos, trimer_rate_vals, width=0.8, color='orange', alpha=0.7)
    ax_top.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=24, fontweight='bold')
    ax_top.set_title('3mer Mutation Rate', fontsize=26, fontweight='bold', pad=20)
    ax_top.set_xticks(x_pos)
    ax_top.set_xticklabels(all_trimers, rotation=45, ha='right', fontsize=16)
    ax_top.tick_params(axis='y', labelsize=18)
    ax_top.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Panel B: Top 16 5mers
    ax_middle = fig.add_subplot(gs[1])
    top_n = min(16, len(kmers))
    top_kmers = kmers[:top_n]
    top_rates = rates[:top_n]
    
    x_pos = np.arange(len(top_kmers))
    ax_middle.bar(x_pos, top_rates, width=0.8, color='orange', alpha=0.7)
    ax_middle.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=24, fontweight='bold')
    ax_middle.set_title(f'Top {top_n} 5mer Mutation Rate', fontsize=26, fontweight='bold', pad=20)
    ax_middle.set_xticks(x_pos)
    ax_middle.set_xticklabels(top_kmers, rotation=45, ha='right', fontsize=16)
    ax_middle.tick_params(axis='y', labelsize=18)
    ax_middle.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Panel C: All 5mers sorted
    ax_bottom = fig.add_subplot(gs[2])
    ax_bottom.bar(range(len(kmers)), rates, width=0.95, color='orange', alpha=0.7)
    ax_bottom.set_xlabel('5mer Context', fontsize=24, fontweight='bold')
    ax_bottom.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=24, fontweight='bold')
    ax_bottom.set_title('Average 5mer Mutation Rate', fontsize=26, fontweight='bold', pad=20)
    ax_bottom.set_xticks(range(len(kmers)))
    ax_bottom.set_xticklabels([])
    ax_bottom.tick_params(axis='y', labelsize=18)
    ax_bottom.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Adjust y-axis limits
    if rates:
        max_value = max(rates)
        min_value = min(rates)
        y_range = max_value - min_value
        ax_bottom.set_ylim(min_value - 0.1 * abs(min_value), max_value + 0.1 * y_range)
    
    # Add panel labels
    pos_top = ax_top.get_position()
    pos_middle = ax_middle.get_position()
    pos_bottom = ax_bottom.get_position()
    title_offset = 0.02
    
    fig.text(0.02, pos_top.y1 - title_offset, 'A', fontsize=32, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_middle.y1 - title_offset, 'B', fontsize=32, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_bottom.y1 - title_offset, 'C', fontsize=32, fontweight='bold', va='top', ha='left')
    
    output_file = os.path.join(output_dir, f'{plot_prefix}_5mer_enrichment_rates.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {plot_prefix} 5mer enrichment rate plot: {output_file}")
    
    # Write TSV output for 5mer rates
    tsv_file = os.path.join(output_dir, f'{plot_prefix}_5mer_enrichment_rates.tsv')
    df_5mer = pd.DataFrame({
        '5mer': kmers,
        'rate': rates
    })
    df_5mer.to_csv(tsv_file, sep='\t', index=False, float_format='%.6e')
    print(f"Saved 5mer rates TSV: {tsv_file}")
    
    # Write TSV output for 3mer rates
    tsv_3mer_file = os.path.join(output_dir, f'{plot_prefix}_3mer_enrichment_rates.tsv')
    df_3mer = pd.DataFrame({
        '3mer': all_trimers,
        'rate': trimer_rate_vals
    })
    df_3mer = df_3mer.sort_values('rate', ascending=False)
    df_3mer.to_csv(tsv_3mer_file, sep='\t', index=False, float_format='%.6e')
    print(f"Saved 3mer rates TSV: {tsv_3mer_file}")


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
   
    
def extract_dinucleotide_rates(aggregate_rates):
    """
    Extract dinucleotide enrichment rates from 5mer rates.
    
    For a 5mer like "ACGTG" (positions -2, -1, 0, +1, +2):
    - Upstream dinucleotide: AC (positions -2, -1)
    - Downstream dinucleotide: TG (positions +1, +2)
    
    Args:
        aggregate_rates: dict mapping 5mer -> rate
    
    Returns:
        tuple: (upstream_dinuc_rates, downstream_dinuc_rates)
        Each is a dict mapping dinucleotide -> weighted average rate
    """
    from collections import defaultdict
    
    upstream_counts = defaultdict(float)
    upstream_weights = defaultdict(float)
    downstream_counts = defaultdict(float)
    downstream_weights = defaultdict(float)
    
    # Filter to C-centered 5mers only
    c_centered_rates = {kmer: rate for kmer, rate in aggregate_rates.items() 
                       if len(kmer) == 5 and kmer[2] == 'C'}
    
    for kmer, rate in c_centered_rates.items():
        if len(kmer) != 5 or kmer[2] != 'C':
            continue
        
        # Extract dinucleotides
        upstream_dinuc = kmer[0:2]  # positions -2, -1
        downstream_dinuc = kmer[3:5]  # positions +1, +2
        
        # Weight by absolute rate to give more weight to highly enriched/depleted kmers
        weight = abs(rate)
        
        # Accumulate weighted rates
        upstream_counts[upstream_dinuc] += rate * weight
        upstream_weights[upstream_dinuc] += weight
        
        downstream_counts[downstream_dinuc] += rate * weight
        downstream_weights[downstream_dinuc] += weight
    
    # Calculate weighted averages
    upstream_rates = {}
    for dinuc in upstream_counts:
        if upstream_weights[dinuc] > 0:
            upstream_rates[dinuc] = upstream_counts[dinuc] / upstream_weights[dinuc]
        else:
            upstream_rates[dinuc] = 0.0
    
    downstream_rates = {}
    for dinuc in downstream_counts:
        if downstream_weights[dinuc] > 0:
            downstream_rates[dinuc] = downstream_counts[dinuc] / downstream_weights[dinuc]
        else:
            downstream_rates[dinuc] = 0.0
    
    return upstream_rates, downstream_rates


def extract_position_specific_rates(aggregate_rates):
    """
    Extract position-specific base enrichment rates from 5mer rates.
    
    For a 5mer like "ACGTG" (positions -2, -1, 0, +1, +2):
    - Position -2: A
    - Position -1: C
    - Position 0: G (mutated C, not included)
    - Position +1: T
    - Position +2: G
    
    Args:
        aggregate_rates: dict mapping 5mer -> rate
    
    Returns:
        dict mapping position -> dict mapping base -> weighted average rate
        Positions: -2, -1, +1, +2
    """
    from collections import defaultdict
    
    position_rates = {
        -2: defaultdict(float),
        -1: defaultdict(float),
        +1: defaultdict(float),
        +2: defaultdict(float)
    }
    position_weights = {
        -2: defaultdict(float),
        -1: defaultdict(float),
        +1: defaultdict(float),
        +2: defaultdict(float)
    }
    
    # Filter to C-centered 5mers only
    c_centered_rates = {kmer: rate for kmer, rate in aggregate_rates.items() 
                       if len(kmer) == 5 and kmer[2] == 'C'}
    
    for kmer, rate in c_centered_rates.items():
        if len(kmer) != 5 or kmer[2] != 'C':
            continue
        
        # Extract bases at each position
        # Position mapping: 0=-2, 1=-1, 3=+1, 4=+2
        bases = {
            -2: kmer[0],
            -1: kmer[1],
            +1: kmer[3],
            +2: kmer[4]
        }
        
        # Weight by absolute rate
        weight = abs(rate)
        
        for pos, base in bases.items():
            position_rates[pos][base] += rate * weight
            position_weights[pos][base] += weight
    
    # Calculate weighted averages
    result = {}
    for pos in [-2, -1, +1, +2]:
        result[pos] = {}
        for base in position_rates[pos]:
            if position_weights[pos][base] > 0:
                result[pos][base] = position_rates[pos][base] / position_weights[pos][base]
            else:
                result[pos][base] = 0.0
    
    return result


def plot_5mer_enrichment_rates_single_panel(aggregate_rates, output_dir, plot_prefix='observed', fig_width=20, fig_height=6):
    """
    Create a multi-panel plot of 5mer enrichment rates with sequence logo-style heatmap.
    
    Panels:
    1. 5mer enrichment rates (main plot with table)
    2. Sequence logo-style heatmap showing nucleotide mutation rates at positions -2, -1, +1, +2
    
    Args:
        aggregate_rates: dict mapping kmer -> rate (depth-normalized, treated - control)
        output_dir: output directory
        plot_prefix: prefix for output filenames (default: 'observed')
        fig_width: figure width in inches (default: 20)
        fig_height: figure height in inches (default: 6)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not aggregate_rates or len(aggregate_rates) == 0:
        print(f"ERROR: aggregate_rates is empty!")
        return
    
    # Filter to C-centered kmers only
    c_centered_rates = {kmer: rate for kmer, rate in aggregate_rates.items() 
                       if len(kmer) == 5 and kmer[2] == 'C'}
    
    if not c_centered_rates:
        print(f"ERROR: No C-centered 5mers found!")
        return
    
    # Extract position-specific rates and dinucleotide rates first
    position_rates = extract_position_specific_rates(aggregate_rates)
    upstream_dinuc_rates, downstream_dinuc_rates = extract_dinucleotide_rates(aggregate_rates)
    
    # Calculate median mutation rate for enrichment calculation
    all_rate_values = list(c_centered_rates.values())
    median_rate = np.median(all_rate_values)
    
    # Calculate enrichment (deviation from median) for all rates
    enrichments = {kmer: rate - median_rate for kmer, rate in c_centered_rates.items()}
    
    # Sort by rate in ascending order (lowest to highest, left to right)
    sorted_kmers = sorted(c_centered_rates.items(), key=lambda x: x[1], reverse=False)
    kmers = [k[0] for k in sorted_kmers]
    rates = [k[1] for k in sorted_kmers]
    enrichments_list = [enrichments[kmer] for kmer in kmers]
    
    # Get top 3 enriched (highest rates) and top 3 depleted (lowest rates)
    top_3_enriched = sorted(c_centered_rates.items(), key=lambda x: x[1], reverse=True)[:3]
    top_3_depleted = sorted(c_centered_rates.items(), key=lambda x: x[1], reverse=False)[:3]
    
    # Determine global enrichment range for consistent color scale across all panels
    all_enrichment_values = list(enrichments.values())
    # Also include enrichments from position-specific and dinucleotide rates
    for pos in position_rates.values():
        for rate in pos.values():
            all_enrichment_values.append(rate - median_rate)
    for rate in upstream_dinuc_rates.values():
        all_enrichment_values.append(rate - median_rate)
    for rate in downstream_dinuc_rates.values():
        all_enrichment_values.append(rate - median_rate)
    
    # Use actual maximum to ensure colorbar covers full data range
    if all_enrichment_values:
        abs_enrichments = [abs(e) for e in all_enrichment_values]
        max_enrichment = np.max(abs_enrichments)
        # Ensure we have a reasonable minimum range
        if max_enrichment < 1e-8:
            max_enrichment = 1e-6
    else:
        max_enrichment = 1e-6
    
    # Extend the colorbar range beyond the data extremes to use more of the color spectrum
    # This improves contrast by compressing the data range within the colormap
    # Extend by 30% to give better color distribution
    extended_max = max_enrichment * 1.3
    vmin_enrichment = -extended_max
    vmax_enrichment = extended_max
    
    # Create blue-to-orange diverging colormap
    import matplotlib.colors as mcolors
    # Blue (low) to white (median) to orange (high)
    # Removed colors near center to compress white region and improve contrast
    # This spreads data across more of the blue/orange spectrum
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_orange_diverging',
        ['#2166AC', '#4393C3', '#92C5DE', '#F7F7F7', '#F4A582', '#FF8C42', '#FF6B00']
    )
    
    # Use linear normalization with extended range for better contrast
    # The extended range compresses the data within the colormap, using more of the spectrum
    norm = mcolors.Normalize(vmin=vmin_enrichment, vmax=vmax_enrichment)
    
    # Create multi-panel figure with GridSpec
    # Layout: 3 rows, 1 column
    # Row 1: 5mer plot (larger)
    # Row 2: Sequence logo-style heatmap (nucleotides)
    # Row 3: Dinucleotide pair heatmap
    fig = plt.figure(figsize=(fig_width, fig_height * 3.5))
    gs = GridSpec(3, 1, height_ratios=[1.5, 1, 1.2], hspace=0.35, top=0.97, bottom=0.08, right=0.85)
    
    # Panel 1: 5mer enrichment rates
    ax1 = fig.add_subplot(gs[0])
    
    # Create bar plot with colors based on enrichment
    bars = ax1.bar(range(len(kmers)), rates, width=0.95)
    # Color bars based on enrichment (using PowerNorm for better contrast)
    for i, (bar, enrichment) in enumerate(zip(bars, enrichments_list)):
        bar.set_facecolor(cmap(norm(enrichment)))
        bar.set_edgecolor('none')
    ax1.set_xlabel('5mer Context', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Depth-Normalized Rate\n(EMS - Control)', fontsize=20, fontweight='bold')
    ax1.set_title('5mer Mutation Rate', fontsize=22, fontweight='bold', pad=15)
    ax1.set_xticks(range(len(kmers)))
    ax1.set_xticklabels([])  # Remove x-axis labels
    ax1.tick_params(axis='y', labelsize=16)
    
    # Add table with top 3 enriched and top 3 depleted 5mers
    if top_3_enriched or top_3_depleted:
        table_data = []
        
        # Add enriched section header and data
        if top_3_enriched:
            table_data.append(['Top 3 Enriched', ''])
            for i, (kmer, rate) in enumerate(top_3_enriched, 1):
                # Format rate with consistent width for better alignment
                rate_str = f"{rate:>10.2e}"
                table_data.append([f"  {i}. {kmer}", rate_str])
        
        # Add separator row
        if top_3_enriched and top_3_depleted:
            table_data.append(['', ''])  # Empty row for spacing
        
        # Add depleted section header and data
        if top_3_depleted:
            table_data.append(['Top 3 Depleted', ''])
            for i, (kmer, rate) in enumerate(top_3_depleted, 1):
                # Format rate with consistent width for better alignment
                rate_str = f"{rate:>10.2e}"
                table_data.append([f"  {i}. {kmer}", rate_str])
        
        # Create table with better positioning and sizing
        table = ax1.table(cellText=table_data,
                        colLabels=['5mer', 'Rate'],
                        cellLoc='left',
                        loc='upper left',
                        bbox=[0.02, 0.50, 0.35, 0.45],
                        colWidths=[0.20, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.0)
        
        # Style the table with improved appearance
        for i in range(len(table_data) + 1):  # +1 for header row
            for j in range(2):
                cell = table[(i, j)]
                
                if i == 0:  # Header row
                    cell.set_facecolor('white')  # Remove blue background
                    cell.set_text_props(weight='bold', color='black', size=14)
                    cell.set_edgecolor('#1A1A1A')
                    cell.set_linewidth(1.0)
                    if j == 1:  # Rate column
                        cell.get_text().set_horizontalalignment('right')
                else:
                    row_data = table_data[i-1]
                    if row_data[0] in ['Top 3 Enriched', 'Top 3 Depleted']:
                        # Use blue and orange colors matching the colormap
                        if row_data[0] == 'Top 3 Enriched':
                            cell.set_facecolor('#FF8C42')  # Orange for enriched (above median)
                            cell.set_text_props(weight='bold', color='black', size=13)
                        else:
                            cell.set_facecolor('#92C5DE')  # Blue for depleted (below median)
                            cell.set_text_props(weight='bold', color='black', size=13)
                        cell.set_edgecolor('#1A1A1A')
                        cell.set_linewidth(0.8)
                    elif row_data[0] == '':  # Empty separator row
                        cell.set_facecolor('#F0F0F0')
                        cell.set_text_props(size=8)
                        cell.set_edgecolor('#E0E0E0')
                        cell.set_linewidth(0.3)
                    else:  # Data row
                        if (i - 1) % 2 == 0:
                            cell.set_facecolor('#F8F9FA')
                        else:
                            cell.set_facecolor('white')
                        cell.set_text_props(size=12, family='monospace')
                        cell.set_edgecolor('#D0D0D0')
                        cell.set_linewidth(0.5)
                        if j == 1:  # Rate column
                            cell.get_text().set_horizontalalignment('right')
    
    # Set log scale for y-axis
    if rates:
        min_rate = min(rates)
        max_rate = max(rates)
        if min_rate > 0:
            ax1.set_yscale('log')
            # Set ylim first to ensure proper range
            ax1.set_ylim(bottom=min_rate * 0.9, top=max_rate * 1.1)
            # Manually generate ticks based on data range to ensure multiple ticks
            from math import floor, ceil, log10
            log_min = floor(log10(min_rate))
            log_max = ceil(log10(max_rate))
            # Generate ticks at 1, 2, 5 times powers of 10
            ticks = []
            for exp in range(log_min - 1, log_max + 2):
                for mult in [1, 2, 5]:
                    tick_val = mult * (10 ** exp)
                    if min_rate * 0.8 <= tick_val <= max_rate * 1.2:
                        ticks.append(tick_val)
            # Remove duplicates and sort
            ticks = sorted(list(set(ticks)))
            if len(ticks) > 0:
                ax1.set_yticks(ticks)
                # Format labels properly for log scale
                from matplotlib.ticker import FuncFormatter
                def log_formatter(x, pos):
                    if x >= 1:
                        return f'{x:.0f}'
                    elif x >= 0.01:
                        return f'{x:.2f}'
                    else:
                        return f'{x:.2e}'
                ax1.yaxis.set_major_formatter(FuncFormatter(log_formatter))
                print(f"DEBUG: Set {len(ticks)} y-axis ticks for log scale")
            else:
                # Fallback to LogLocator if manual generation fails
                locator = LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=15)
                ax1.yaxis.set_major_locator(locator)
                print("DEBUG: Using LogLocator fallback")
            ax1.tick_params(axis='y', labelsize=16, which='major')
            # Ensure minor ticks are off to avoid clutter
            ax1.tick_params(axis='y', which='minor', length=0)
        else:
            max_abs_rate = max(abs(r) for r in rates)
            if max_abs_rate > 0:
                linthresh = max(1e-8, max_abs_rate * 0.01)
                ax1.set_yscale('symlog', linthresh=linthresh)
                # Set ylim first
                ax1.set_ylim(bottom=-max_abs_rate * 1.1, top=max_abs_rate * 1.1)
                # Use SymmetricalLogLocator for symlog scale
                locator = SymmetricalLogLocator(base=10, linthresh=linthresh, subs=[1.0, 2.0, 5.0])
                ax1.yaxis.set_major_locator(locator)
                ax1.tick_params(axis='y', labelsize=16)
            else:
                ax1.set_yscale('log')
                ax1.set_ylim(bottom=1e-6, top=1e-4)
                locator = LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=15)
                ax1.yaxis.set_major_locator(locator)
                ax1.tick_params(axis='y', labelsize=16)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Panel 2: Sequence logo-style heatmap showing nucleotide enrichment
    ax2 = fig.add_subplot(gs[1])
    if position_rates:
        positions = [-2, -1, +1, +2]
        bases = ['A', 'T', 'G', 'C']
        
        # Prepare data matrix for heatmap: rows = positions, columns = bases
        # Convert rates to enrichments (deviation from median)
        heatmap_data = []
        for pos in positions:
            row = []
            for base in bases:
                rate = position_rates.get(pos, {}).get(base, 0.0)
                enrichment = rate - median_rate
                row.append(enrichment)
            heatmap_data.append(row)
        
        # Create arrays for upstream and downstream separately
        upstream_array = np.array([heatmap_data[0], heatmap_data[1]])  # -2, -1
        downstream_array = np.array([heatmap_data[2], heatmap_data[3]])  # +1, +2
        
        # Calculate gap size: 1/3 of a row height
        gap_size = 1.0 / 3.0
        
        # Create coordinate arrays for pcolormesh
        # Upstream: rows 0-1, columns 0-3
        # Gap: row 2 (1/3 size), columns 0-3
        # Downstream: rows 2+gap to 3+gap, columns 0-3
        
        # X coordinates (same for all): column edges
        x_edges = np.arange(len(bases) + 1) - 0.5
        
        # Y coordinates: row edges with gap
        # Upstream rows: 0 to 2
        # Gap: 2 to 2+gap_size (1/3 size)
        # Downstream rows: 2+gap_size to 4+gap_size
        y_edges = np.array([0, 1, 2, 2+gap_size, 3+gap_size, 4+gap_size])
        
        # Combine data with gap row
        combined_data = np.vstack([
            upstream_array,
            np.full((1, len(bases)), np.nan),  # Gap row
            downstream_array
        ])
        
        # Use pcolormesh for precise control over cell sizes
        X, Y = np.meshgrid(x_edges, y_edges)
        im = ax2.pcolormesh(X, Y, combined_data, cmap=cmap, norm=norm, 
                           shading='flat', edgecolors='none')
        
        # Set ticks and labels
        # Y-axis: positions at center of each data row (skip gap)
        y_positions = [0.5, 1.5, 2.5+gap_size, 3.5+gap_size]
        y_labels = [f'{p:+d}' for p in positions]
        ax2.set_xticks(np.arange(len(bases)))
        ax2.set_yticks(y_positions)
        ax2.set_xticklabels(bases, fontsize=18, fontweight='bold')
        ax2.set_yticklabels(y_labels, fontsize=18, fontweight='bold')
        
        # Set axis limits
        ax2.set_xlim(-0.5, len(bases)-0.5)
        ax2.set_ylim(4+gap_size, 0)
        
        # Remove text annotations from heatmap (values not shown)
        
        # Labels and title
        ax2.set_xlabel('Nucleotide', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Position Relative to Mutation', fontsize=18, fontweight='bold')
        ax2.set_title('Nucleotide Enrichment at Flanking Positions', fontsize=20, fontweight='bold', pad=15)
    
    # Panel 3: Dinucleotide pair heatmap
    ax3 = fig.add_subplot(gs[2])
    # Extract dinucleotide rates
    upstream_dinuc_rates, downstream_dinuc_rates = extract_dinucleotide_rates(aggregate_rates)
    
    if upstream_dinuc_rates or downstream_dinuc_rates:
        bases = ['A', 'T', 'G', 'C']
        
        # Create two 4x4 heatmaps side by side (upstream and downstream)
        # Prepare data matrices: rows = first base, columns = second base
        # Convert rates to enrichments (deviation from median)
        upstream_data = np.zeros((4, 4))
        downstream_data = np.zeros((4, 4))
        
        for i, base1 in enumerate(bases):
            for j, base2 in enumerate(bases):
                dinuc = base1 + base2
                upstream_enrichment = upstream_dinuc_rates.get(dinuc, 0.0) - median_rate
                downstream_enrichment = downstream_dinuc_rates.get(dinuc, 0.0) - median_rate
                upstream_data[i, j] = upstream_enrichment
                downstream_data[i, j] = downstream_enrichment
        
        # Calculate gap size: smaller gap (about 1/5 of a column width)
        gap_size = 1.0 / 5.0
        
        # Create coordinate arrays for pcolormesh
        # Upstream: columns 0-3, rows 0-3
        # Gap: column 4 (1/3 size), rows 0-3
        # Downstream: columns 4+gap to 7+gap, rows 0-3
        
        # X coordinates: column edges with gap
        # Upstream columns: 0 to 4 (each column is 1 unit wide)
        # Gap: 4 to 4+gap_size (1/3 size)
        # Downstream columns: 4+gap_size to 8+gap_size (each column is 1 unit wide)
        # Subtract 0.5 to center boxes at integer positions
        x_edges = np.array([0, 1, 2, 3, 4, 4+gap_size, 5+gap_size, 6+gap_size, 7+gap_size, 8+gap_size]) - 0.5
        # Box centers are at midpoints: 0, 1, 2, 3, 4+gap_size, 5+gap_size, 6+gap_size, 7+gap_size
        
        # Y coordinates: row edges
        y_edges = np.arange(5) - 0.5
        
        # Combine data with gap column
        gap_column = np.full((4, 1), np.nan)
        combined_data = np.hstack([upstream_data, gap_column, downstream_data])
        
        # Use pcolormesh for precise control over cell sizes
        X, Y = np.meshgrid(x_edges, y_edges)
        im = ax3.pcolormesh(X, Y, combined_data, cmap=cmap, norm=norm, 
                           shading='flat', edgecolors='none')
        
        # Set ticks and labels
        # X-axis: two sets of 4 ticks (one for upstream, one for downstream)
        # Calculate box centers directly from x_edges to ensure perfect alignment
        # Upstream boxes: edges at indices 0-4, centers are midpoints
        x_ticks_upstream = [(x_edges[i] + x_edges[i+1]) / 2 for i in range(4)]
        # Downstream boxes: edges at indices 5-9, centers are midpoints
        x_ticks_downstream = [(x_edges[i] + x_edges[i+1]) / 2 for i in range(5, 9)]
        x_ticks = x_ticks_upstream + x_ticks_downstream
        x_labels = bases + bases
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(x_labels, fontsize=16, fontweight='bold')
        
        # Y-axis: bases (first base of dinucleotide)
        # Position ticks at center of each row
        # With y_edges = [-0.5, 0.5, 1.5, 2.5, 3.5], row centers are at [0, 1, 2, 3]
        y_tick_positions = [0, 1, 2, 3]  # Centers of each row
        ax3.set_yticks(y_tick_positions)
        ax3.set_yticklabels(bases, fontsize=18, fontweight='bold')
        
        # Set axis limits
        ax3.set_xlim(-0.5, 8+gap_size-0.5)
        ax3.set_ylim(3.5, -0.5)
        
        # Labels (no title)
        ax3.set_xlabel('Second Base of Dinucleotide', fontsize=18, fontweight='bold')
        ax3.set_ylabel('First Base of Dinucleotide', fontsize=18, fontweight='bold')
        
        # Add section labels as plain text (no boxes)
        # Adjust positions to account for gap: upstream center is at 1.5, downstream center is at 6.5
        ax3.text(1.5, -0.15, 'Upstream (-2, -1)', ha='center', va='top', 
                fontsize=18, fontweight='bold')
        ax3.text(6.5, -0.15, 'Downstream (+1, +2)', ha='center', va='top', 
                fontsize=18, fontweight='bold')
    
    # Add shared colorbar for all panels (positioned on the right side, 70% of figure height)
    # Calculate 70% of figure height: bottom=0.08, top=0.97, so height=0.89, 70% = 0.623
    # Center it vertically: bottom = 0.08 + (0.89 - 0.623) / 2 = 0.08 + 0.1335 = 0.2135
    cbar_height = 0.89 * 0.7  # 70% of available height
    cbar_bottom = 0.08 + (0.89 - cbar_height) / 2  # Center vertically
    cbar_ax = fig.add_axes([0.87, cbar_bottom, 0.02, cbar_height])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Enrichment from Median\nMutation Rate', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    # Add panel labels
    # Get panel positions to align labels properly
    pos_top = ax1.get_position()
    pos_middle = ax2.get_position()
    pos_bottom = ax3.get_position()
    
    fig.text(0.02, pos_top.y1 - 0.02, 'A', fontsize=24, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_middle.y1 - 0.02, 'B', fontsize=24, fontweight='bold', va='top', ha='left')
    fig.text(0.02, pos_bottom.y1 - 0.02, 'C', fontsize=24, fontweight='bold', va='top', ha='left')
    
    output_file = os.path.join(output_dir, f'{plot_prefix}_5mer_enrichment_rates_single_panel.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {plot_prefix} 5mer enrichment rate multi-panel plot: {output_file}")


def load_rates_from_tsv(tsv_file):
    """
    Load 5mer rates from TSV file.
    
    Args:
        tsv_file: Path to TSV file with columns '5mer' and 'rate'
    
    Returns:
        dict mapping kmer -> rate, or None if file doesn't exist or can't be loaded
    """
    if not os.path.exists(tsv_file):
        print(f"Warning: TSV file not found: {tsv_file}")
        return None
    
    try:
        df = pd.read_csv(tsv_file, sep='\t')
        if '5mer' not in df.columns or 'rate' not in df.columns:
            print(f"Warning: TSV file missing required columns (5mer, rate): {tsv_file}")
            return None
        
        rates = dict(zip(df['5mer'], df['rate']))
        print(f"Loaded {len(rates)} 5mer rates from {tsv_file}")
        return rates
    except Exception as e:
        print(f"Error loading TSV file {tsv_file}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot 5mer enrichment rates from uncollapsed kmer data and model predictions."
    )
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for plots")
    parser.add_argument("--regenerate-from-tsv", action='store_true',
                        help="Regenerate plots from existing TSV files without redoing analysis")
    parser.add_argument("--kmers-dir",
                        help="Directory containing *_variants_5mer_contexts.json files (required unless --regenerate-from-tsv)")
    parser.add_argument("--genome-fasta",
                        help="Path to genome FASTA file (supports .gz files) (required unless --regenerate-from-tsv)")
    parser.add_argument("--counts-dir", type=str,
                        help="Directory containing .counts files for depth normalization (required unless --regenerate-from-tsv)")
    parser.add_argument("--exclusion-mask", type=str, default=None,
                        help="Optional: Path to exclusion mask TSV file (chrom\tpos format). If not provided, will build from controls.")
    parser.add_argument("--min-alt", type=int, default=1,
                        help="Minimum alt allele count required for a site to be included (default: 1)")
    parser.add_argument("--model-dir", type=str,
                        help="Path to directory containing fitted 5mer model (e.g., output_dir/fitted_models). Must contain 5mer_model.pkl. (required unless --regenerate-from-tsv)")
    
    args = parser.parse_args()
    
    # If regenerating from TSV, skip analysis and just load TSV files
    if args.regenerate_from_tsv:
        print("=== REGENERATING PLOTS FROM TSV FILES ===")
        
        # Look for TSV files in output directory
        observed_tsv = os.path.join(args.output_dir, 'observed_5mer_enrichment_rates.tsv')
        model_tsv = os.path.join(args.output_dir, 'model_5mer_enrichment_rates.tsv')
        
        # Load observed rates if available
        observed_rates = None
        if os.path.exists(observed_tsv):
            observed_rates = load_rates_from_tsv(observed_tsv)
            if observed_rates:
                print("\nCreating observed 5mer enrichment rate plots...")
                plot_5mer_enrichment_rates(observed_rates, args.output_dir, plot_prefix='observed')
                plot_5mer_enrichment_rates_single_panel(observed_rates, args.output_dir, plot_prefix='observed')
        else:
            print(f"Warning: Observed rates TSV not found: {observed_tsv}")
        
        # Load model rates if available
        model_rates = None
        if os.path.exists(model_tsv):
            model_rates = load_rates_from_tsv(model_tsv)
            if model_rates:
                print("\nCreating model-based 5mer enrichment rate plots...")
                plot_5mer_enrichment_rates(model_rates, args.output_dir, plot_prefix='model')
                plot_5mer_enrichment_rates_single_panel(model_rates, args.output_dir, plot_prefix='model')
        else:
            print(f"Warning: Model rates TSV not found: {model_tsv}")
        
        if not observed_rates and not model_rates:
            print("ERROR: No TSV files found to regenerate plots from!")
            return
        
        print(f"\n=== OUTPUT SUMMARY ===")
        print(f"Plots regenerated from TSV files in {args.output_dir}")
        return
    
    # Normal analysis mode - check required arguments
    if not args.kmers_dir:
        parser.error("--kmers-dir is required (unless --regenerate-from-tsv)")
    if not args.genome_fasta:
        parser.error("--genome-fasta is required (unless --regenerate-from-tsv)")
    if not args.counts_dir:
        parser.error("--counts-dir is required (unless --regenerate-from-tsv)")
    if not args.model_dir:
        parser.error("--model-dir is required (unless --regenerate-from-tsv)")
    
    # Load uncollapsed 5mer context counts
    print("Loading uncollapsed 5mer context counts...")
    context_counts = load_5mer_contexts(args.kmers_dir, use_gene_only=False, no_strand_collapse=True)
    
    if not context_counts:
        print("No 5mer context data found!")
        return
    
    # Load or build exclusion mask
    print("\n=== EXCLUSION MASK ===")
    exclusion_mask_set = None
    exclusion_mask_file = args.exclusion_mask
    if args.exclusion_mask:
        print(f"Loading exclusion mask from {args.exclusion_mask}...")
        exclusion_mask_set = load_exclusion_mask_tsv(args.exclusion_mask)
        if exclusion_mask_set:
            print(f"  Loaded {len(exclusion_mask_set)} sites to exclude")
    else:
        print("Building exclusion mask from control files...")
        exclusion_mask_set = build_exclusion_mask_from_controls(args.counts_dir, min_alt=args.min_alt)
        if exclusion_mask_set:
            print(f"  Built exclusion mask: {len(exclusion_mask_set)} sites to exclude")
    
    # Load depth information and calculate aggregate rates from uncollapsed data
    print("\n=== OBSERVED RATES ===")
    print(f"Loading depth information from {args.counts_dir}...")
    sample_depths = load_sample_depths(args.counts_dir, exclusion_mask=exclusion_mask_set, min_alt=args.min_alt)
    
    if not sample_depths:
        print("ERROR: No depth information found!")
        return
    
    print("Calculating aggregate depth-normalized rates with background subtraction...")
    aggregate_rates = calculate_aggregate_depth_normalized_rates(
        context_counts, sample_depths, use_collapsed=True
    )
    
    if not aggregate_rates:
        print("ERROR: Failed to calculate aggregate rates!")
        return
    
    print(f"Calculated rates for {len(aggregate_rates)} kmers")
    print(f"  Rate range: {min(aggregate_rates.values()):.2e} to {max(aggregate_rates.values()):.2e}")
    
    # Create plot for observed rates
    print("\nCreating observed 5mer enrichment rate plot...")
    plot_5mer_enrichment_rates(aggregate_rates, args.output_dir, plot_prefix='observed')
    
    # Create single panel plot for observed rates
    print("\nCreating observed 5mer enrichment rate single panel plot...")
    plot_5mer_enrichment_rates_single_panel(aggregate_rates, args.output_dir, plot_prefix='observed')
    
    # Predict model-based rates
    print("\n=== MODEL-BASED RATES ===")
    print(f"Predicting rates from fitted 5mer model at {args.model_dir}...")
    model_rates = predict_5mer_rates_from_model(
        args.model_dir, 
        args.counts_dir, 
        args.genome_fasta, 
        exclusion_mask=exclusion_mask_file
    )
    
    if not model_rates:
        print("ERROR: Could not predict model-based rates!")
        return
    
    # Convert model_rates dict to simple rate dict for plotting
    model_rates_simple = {kmer: data['rate'] if isinstance(data, dict) else data 
                         for kmer, data in model_rates.items()}
    
    print(f"Predicted rates for {len(model_rates_simple)} 5mers")
    print(f"  Rate range: {min(model_rates_simple.values()):.2e} to {max(model_rates_simple.values()):.2e}")
    
    # Create plot for model rates
    print("\nCreating model-based 5mer enrichment rate plot...")
    plot_5mer_enrichment_rates(model_rates_simple, args.output_dir, plot_prefix='model')
    
    # Create single panel plot for model rates
    print("\nCreating model-based 5mer enrichment rate single panel plot...")
    plot_5mer_enrichment_rates_single_panel(model_rates_simple, args.output_dir, plot_prefix='model')
    
    print(f"\n=== OUTPUT SUMMARY ===")
    print(f"All plots saved to {args.output_dir}")
    print("  - observed_5mer_enrichment_rates.png")
    print("  - observed_5mer_enrichment_rates_single_panel.png")
    print("  - model_5mer_enrichment_rates.png")
    print("  - model_5mer_enrichment_rates_single_panel.png")


if __name__ == "__main__":
    main()
