#!/usr/bin/env python3
"""
Plot mutation spectra from count TSV files.
Matches plot_spectra.py logic exactly.

Usage:
  python plot_ems_spectra.py /path/to/counts_dir -o output.png --min-alt 3 --min-depth 10
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu


def list_tsv_files(input_dir: str, pattern: str) -> List[str]:
    pattern_path = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(pattern_path))
    return [f for f in files if os.path.isfile(f)]


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


def perform_mann_whitney_tests(df, NT_samples_sorted, other_samples_sorted, substitution_order):
    """Perform Mann-Whitney U tests comparing NT controls vs treated samples for each substitution type.
    
    Returns dict mapping substitution -> p_value
    """
    p_values = {}
    
    for substitution in substitution_order:
        # Get data for this substitution
        sub_data = df[df['substitution'] == substitution]
        
        if sub_data.empty:
            p_values[substitution] = 1.0
            continue
            
        # Separate NT controls and treated samples
        nt_values = []
        treated_values = []
        
        for _, row in sub_data.iterrows():
            sample = row['sample']
            proportion = row['proportion']
            
            if sample in NT_samples_sorted:
                nt_values.append(proportion)
            elif sample in other_samples_sorted:
                treated_values.append(proportion)
        
        # Perform Mann-Whitney U test
        if len(nt_values) >= 2 and len(treated_values) >= 2:
            try:
                statistic, p_value = mannwhitneyu(treated_values, nt_values, alternative='two-sided')
                p_values[substitution] = p_value
            except ValueError:
                p_values[substitution] = 1.0
        else:
            p_values[substitution] = 1.0
    
    return p_values


def detect_columns(path: str) -> Tuple[bool, Dict[str, int]]:
    """Detect columns - supports wide count format."""
    with open(path, 'r') as fh:
        first = fh.readline().rstrip('\n')
    if not first:
        return False, {}
    parts = first.split('\t')
    lower = [p.strip().lower() for p in parts]
    col_map: Dict[str, int] = {}

    name_aliases = {
        'chrom': ['chrom', 'chr', 'contig', 'scaffold'],
        'pos': ['pos', 'position'],
        'ref': ['ref', 'reference', 'ref_base'],
        'depth': ['depth', 'dp', 'cov', 'coverage'],
        'ref_count': ['ref_count', 'refreads', 'ref_reads', 'ref_n', 'refcnt'],
        'A_count': ['a_count', 'a_reads', 'a_n', 'a'],
        'C_count': ['c_count', 'c_reads', 'c_n', 'c'],
        'G_count': ['g_count', 'g_reads', 'g_n', 'g'],
        'T_count': ['t_count', 't_reads', 't_n', 't'],
    }
    for key, aliases in name_aliases.items():
        for a in aliases:
            if a in lower:
                col_map[key] = lower.index(a)
                break

    # If headered
    if 'chrom' in col_map and 'pos' in col_map and 'ref' in col_map:
        return True, col_map

    # Headerless wide fallback
    if len(parts) >= 9:
        fallback = {
            'chrom': 0, 'pos': 1, 'ref': 2, 'ref_count': 3,
            'A_count': 4, 'C_count': 5, 'G_count': 6, 'T_count': 7, 'depth': 8
        }
        return False, fallback

    return False, col_map


def process_tsv_file(tsv_path: str, min_alt: int, min_depth: int, min_rate: float, 
                     depth_correction: str, excluded_positions: set = None) -> Dict[str, float]:
    """
    Process a single TSV file to compute mutation rates.
    Matches plot_spectra.py logic exactly.
    
    depth_correction: 'filter' (default) or 'normalize'
    
    Returns dict of substitution -> rate
    """
    has_header, col_map = detect_columns(tsv_path)
    
    # Need at least chrom, pos, ref
    for req in ['chrom', 'pos', 'ref']:
        if req not in col_map:
            return {}

    # Must be wide format
    if not all(k in col_map for k in ['A_count', 'C_count', 'G_count', 'T_count', 'depth']):
        return {}

    # Track depth-weighted mutant sites per type
    mutant_depth_by_type: Dict[str, float] = defaultdict(float)
    # Track total depth per reference base  
    total_depth_by_ref: Dict[str, float] = defaultdict(float)
    # Track which sites we've counted (prevent double-counting)
    seen_site_for_type: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    seen_site_for_ref: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    with open(tsv_path, 'r') as fh:
        if has_header:
            next(fh, None)
        for line in fh:
            if not line or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')

            chrom = parts[col_map['chrom']] if col_map['chrom'] < len(parts) else None
            pos = parts[col_map['pos']] if col_map['pos'] < len(parts) else None
            ref = parts[col_map['ref']] if col_map['ref'] < len(parts) else None
            if chrom is None or pos is None or ref is None:
                continue
            site_key = (chrom, pos)
            
            # Skip excluded positions
            if excluded_positions and site_key in excluded_positions:
                continue

            # Get depth
            depth_val = None
            if 'depth' in col_map and col_map['depth'] < len(parts):
                try:
                    depth_val = float(parts[col_map['depth']])
                except ValueError:
                    depth_val = None
            if depth_val is None or depth_val < min_depth:
                continue

            # Add to denominator (once per site per ref)
            if site_key not in seen_site_for_ref[ref]:
                total_depth_by_ref[ref] += depth_val
                seen_site_for_ref[ref].add(site_key)

            # Get base counts
            counts = {}
            for base in ['A', 'C', 'G', 'T']:
                key = f'{base}_count'
                idx = col_map.get(key)
                if idx is not None and idx < len(parts):
                    try:
                        counts[base] = float(parts[idx])
                    except ValueError:
                        counts[base] = 0.0
                else:
                    counts[base] = 0.0

            # Check each substitution
            for alt in ['A', 'C', 'G', 'T']:
                if alt == ref:
                    continue
                    
                alt_count = counts.get(alt, 0.0)
                sub_name = f"{ref}>{alt}"
                
                # Apply filters based on depth_correction mode (matching plot_spectra.py):
                is_mutant = False
                if depth_correction == 'normalize':
                    # For normalize mode: require BOTH min_alt count AND min_rate threshold
                    count_pass = alt_count >= min_alt
                    rate_pass = (alt_count / depth_val) >= min_rate if depth_val > 0 else False
                    is_mutant = count_pass and rate_pass
                else:
                    # For filter mode (default): only check min_alt
                    is_mutant = alt_count >= min_alt
                
                if is_mutant:
                    # Add depth to numerator (once per site per type)
                    if site_key not in seen_site_for_type[sub_name]:
                        mutant_depth_by_type[sub_name] += depth_val
                        seen_site_for_type[sub_name].add(site_key)

    # Calculate rates: (depth at mutant sites) / (total depth at ref sites)
    # All 12 substitution types
    all_subs = ['A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T', 
                'G>A', 'G>C', 'G>T', 'T>A', 'T>C', 'T>G']
    
    rates = {}
    for sub in all_subs:
        ref = sub.split('>')[0]
        denom = total_depth_by_ref.get(ref, 0.0)
        num = mutant_depth_by_type.get(sub, 0.0)
        rates[sub] = (num / denom) if denom > 0 else 0.0

    return rates


def filter_positions_by_nt_controls(tsv_dir: str, pattern: str, max_nt_proportion: float, 
                                     min_alt: int, min_depth: int, min_rate: float, depth_correction: str):
    """
    Build exclusion set of positions where >max_nt_proportion of NT controls have variants.
    Matches plot_spectra.py logic - applies filters based on depth_correction mode.
    """
    files = list_tsv_files(tsv_dir, pattern)
    
    # Identify NT control files
    nt_files = [f for f in files if 'NT' in os.path.basename(f)]
    
    if not nt_files:
        print("Warning: No NT control samples found. Skipping position filtering.")
        return set()
    
    print(f"Found {len(nt_files)} NT control samples for position filtering")
    
    # FIRST: Identify all positions that exist across ALL files (not just NT) with sufficient depth
    # This matches plot_spectra.py which builds df_combined from all samples first
    all_positions: Set[Tuple[str, str]] = set()
    for f in files:
        has_header, col_map = detect_columns(f)
        if not all(k in col_map for k in ['chrom', 'pos', 'depth']):
            continue
        with open(f, 'r') as fh:
            if has_header:
                next(fh, None)
            for line in fh:
                if not line or line.startswith('#'):
                    continue
                parts = line.rstrip('\n').split('\t')
                chrom = parts[col_map['chrom']] if col_map['chrom'] < len(parts) else None
                pos = parts[col_map['pos']] if col_map['pos'] < len(parts) else None
                if not all([chrom, pos]):
                    continue
                # Check depth filter
                depth_val = None
                if 'depth' in col_map and col_map['depth'] < len(parts):
                    try:
                        depth_val = float(parts[col_map['depth']])
                    except ValueError:
                        pass
                if depth_val is not None and depth_val >= min_depth:
                    all_positions.add((chrom, pos))
    
    print(f"Found {len(all_positions)} unique positions across all samples with depth >= {min_depth}")
    
    # SECOND: For these positions, check NT samples for variants
    position_variant_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    position_sample_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    
    for nt_file in nt_files:
        has_header, col_map = detect_columns(nt_file)
        
        if not all(k in col_map for k in ['chrom', 'pos', 'ref', 'A_count', 'C_count', 'G_count', 'T_count', 'depth']):
            continue
            
        with open(nt_file, 'r') as fh:
            if has_header:
                next(fh, None)
            for line in fh:
                if not line or line.startswith('#'):
                    continue
                parts = line.rstrip('\n').split('\t')
                
                chrom = parts[col_map['chrom']] if col_map['chrom'] < len(parts) else None
                pos = parts[col_map['pos']] if col_map['pos'] < len(parts) else None
                ref = parts[col_map['ref']] if col_map['ref'] < len(parts) else None
                
                if not all([chrom, pos, ref]):
                    continue
                
                site_key = (chrom, pos)
                
                # ONLY evaluate positions that exist in all_positions (matching plot_spectra.py)
                if site_key not in all_positions:
                    continue
                
                # Get depth
                depth_val = None
                if 'depth' in col_map and col_map['depth'] < len(parts):
                    try:
                        depth_val = float(parts[col_map['depth']])
                    except ValueError:
                        pass
                
                # Skip if depth too low
                if depth_val is None or depth_val < min_depth:
                    continue
                    
                position_sample_counts[site_key] += 1
                
                # Check if ANY alt base passes filters (based on depth_correction mode)
                # This matches plot_spectra.py's logic where it first processes sites with filters,
                # then checks if any substitution column > 0
                has_variant = False
                for base in ['A', 'C', 'G', 'T']:
                    if base == ref:
                        continue
                    key = f'{base}_count'
                    idx = col_map.get(key)
                    if idx is not None and idx < len(parts):
                        try:
                            count = float(parts[idx])
                            
                            # Apply filters based on depth_correction mode
                            is_mutant = False
                            if depth_correction == 'normalize':
                                # For normalize: check both min_alt and min_rate
                                count_pass = count >= min_alt
                                rate_pass = (count / depth_val) >= min_rate if depth_val > 0 else False
                                is_mutant = count_pass and rate_pass
                            else:
                                # For filter mode (default): only check min_alt
                                is_mutant = count >= min_alt
                            
                            if is_mutant:
                                has_variant = True
                                break
                        except ValueError:
                            pass
                
                if has_variant:
                    position_variant_counts[site_key] += 1
    
    # Identify positions to exclude
    excluded_positions = set()
    for site_key, variant_count in position_variant_counts.items():
        total_count = position_sample_counts[site_key]
        if total_count > 0:
            proportion = variant_count / total_count
            if proportion > max_nt_proportion:
                excluded_positions.add(site_key)
    
    print(f"Found {len(excluded_positions)} positions exceeding {max_nt_proportion*100}% NT variant threshold")
    return excluded_positions


def plot_spectra(sample_to_rates: Dict[str, Dict[str, float]], output_path: str, 
                 filtering_stats: dict = None, depth_correction: str = 'filter', ax=None, title=None):
    """
    Plot mutation spectra.
    Matches plot_spectra.py style exactly.
    
    Args:
        sample_to_rates: Dict mapping sample names to substitution rates
        output_path: Output file path (ignored if ax is provided)
        filtering_stats: Dict with filtering statistics
        depth_correction: Depth correction method
        ax: Optional matplotlib axis to plot on (for subplots)
        title: Optional title for the plot
    """
    if not sample_to_rates:
        print("Warning: No data to plot.")
        return
    
    # Convert to dataframe format
    import pandas as pd
    
    rows = []
    for sample, rates in sample_to_rates.items():
        for sub, rate in rates.items():
            rows.append({
                'sample': sample,
                'substitution': sub,
                'proportion': rate
            })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("Warning: No data to plot.")
        return
    
    # Sort samples by C>T mutation rates within NT and non-NT groups
    all_samples = df["sample"].unique()
    
    # Get C>T rates for sorting
    ct_rates = df[df["substitution"] == "C>T"].set_index("sample")["proportion"].to_dict()
    
    # Split and sort by C>T rates (ascending order)
    NT_samples = [s for s in all_samples if "NT" in s]
    other_samples = [s for s in all_samples if "NT" not in s]
    
    NT_samples_sorted = sorted(NT_samples, key=lambda x: ct_rates.get(x, 0))
    other_samples_sorted = sorted(other_samples, key=lambda x: ct_rates.get(x, 0))
    
    sample_order = NT_samples_sorted + other_samples_sorted
    
    print(f"Sample order by C>T rates:")
    print(f"  NT samples: {[f'{s} ({ct_rates.get(s, 0):.2e})' for s in NT_samples_sorted]}")
    print(f"  Other samples: {[f'{s} ({ct_rates.get(s, 0):.2e})' for s in other_samples_sorted]}")
    
    # Create color scheme: single grey for NT controls, orange-to-red gradient for treated samples
    NT_GREY = '#808080'  # Single grey color for all NT controls
    
    colors = []
    if NT_samples_sorted:
        # Use single grey shade for all NT controls
        colors.extend([NT_GREY] * len(NT_samples_sorted))
    if other_samples_sorted:
        # Use orange to red gradient for treated samples
        # Create custom orange-to-red colormap
        n_treated = len(other_samples_sorted)
        if n_treated == 1:
            colors.append('#FF6B35')  # Single orange-red color
        else:
            # Create gradient from light orange to dark red
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'orange_red', ['#FFB366', '#FF8C42', '#FF6B35', '#E63946', '#C1121F']
            )
            treated_colors = [mcolors.rgb2hex(cmap(i / (n_treated - 1))) for i in range(n_treated)]
            colors.extend(treated_colors)
    
    color_dict = dict(zip(sample_order, colors))
    
    # Apply clean sample labels for display, ensuring uniqueness
    sample_label_map = {}
    seen_labels = {}
    for s in sample_order:
        base_label = clean_sample_label(s)
        # If this label was already used, append a suffix
        if base_label in seen_labels:
            seen_labels[base_label] += 1
            unique_label = f"{base_label}_{seen_labels[base_label]}"
        else:
            seen_labels[base_label] = 0
            unique_label = base_label
        sample_label_map[s] = unique_label
    
    # Keep individual sample display labels for plotting (each NT gets its own bar)
    # But create a separate legend mapping for grouping NT controls
    df["sample_display"] = df["sample"].map(sample_label_map)
    
    # Create legend mapping: individual samples -> legend groups
    legend_mapping = {}
    for s in sample_order:
        if s in NT_samples_sorted:
            legend_mapping[sample_label_map[s]] = "NT controls"
        else:
            legend_mapping[sample_label_map[s]] = sample_label_map[s]
    
    df["legend_group"] = df["sample_display"].map(legend_mapping)
    
    # Create display label order (individual samples, not groups)
    display_order = [sample_label_map[s] for s in sample_order]
    
    # Color dict uses individual sample labels
    color_dict_display = {sample_label_map[s]: color_dict[s] for s in sample_order}
    
    # Create custom substitution order with complementary pairs adjacent (matching plot_spectra.py)
    substitution_order = ["A>C", "T>G", "A>G", "T>C", "A>T", "T>A", 
                         "C>A", "G>T", "C>G", "G>C", "C>T", "G>A"]
    
    # Sort by custom substitution order, then by sample display order
    df["sample_display"] = pd.Categorical(df["sample_display"], categories=display_order, ordered=True)
    df["substitution"] = pd.Categorical(df["substitution"], categories=substitution_order, ordered=True)
    df = df.sort_values(["substitution", "sample_display"])
    
    # Perform Mann-Whitney U tests
    p_values = perform_mann_whitney_tests(df, NT_samples_sorted, other_samples_sorted, substitution_order)
    
    # Create figure if ax not provided
    if ax is None:
        plt.figure(figsize=(14, 6))
        ax = plt.gca()
        save_figure = True
    else:
        save_figure = False
   
            
    sns.barplot(
        data=df,
        x="substitution",
        y="proportion",
        hue="sample_display",
        palette=color_dict_display,
        ax=ax
    )
    
    # Build ylabel matching plot_spectra.py
    ylabel = "Proportion of sites with mutant alleles"
    
            
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_yscale('log')  # Log scale for y-axis
    ax.set_ylim(bottom=1e-4)  # Set minimum y value to 10^-4
    
    # Adjust upper y-limit to accommodate p-value boxes
    current_ylim = ax.get_ylim()
    ax.set_ylim(top=current_ylim[1] * 1.3)  # Increase upper limit by 30%
    
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.set_xlabel('')  # Remove x-axis label
    
    # Create custom legend that groups NT controls
    handles, labels = ax.get_legend_handles_labels()
    
    # Group NT controls under single legend entry
    legend_handles = []
    legend_labels = []
    seen_nt = False
    
    for handle, label in zip(handles, labels):
        if label in [sample_label_map[s] for s in NT_samples_sorted]:
            if not seen_nt:
                # Add single NT controls entry
                legend_handles.append(handle)
                legend_labels.append("NT controls")
                seen_nt = True
        else:
            # Add individual treated sample entries
            legend_handles.append(handle)
            legend_labels.append(label)
    
    ax.legend(legend_handles, legend_labels, fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Add p-value annotations below the title but above the bars
    y_max = ax.get_ylim()[1]
    y_pos = y_max * 0.72  # Position lower to avoid overlap with bars
    
    for i, substitution in enumerate(substitution_order):
        p_val = p_values.get(substitution, 1.0)
        
        # Format p-value
        if p_val < 0.001:
            p_text = "p < 0.001"
        elif p_val < 0.01:
            p_text = f"p = {p_val:.3f}"
        elif p_val < 0.05:
            p_text = f"p = {p_val:.3f}"
        else:
            p_text = f"p = {p_val:.3f}"
        
        # Color code the p-values
        if p_val < 0.001:
            color = 'red'
            weight = 'bold'
        elif p_val < 0.01:
            color = 'darkorange'
            weight = 'bold'
        elif p_val < 0.05:
            color = 'orange'
            weight = 'normal'
        else:
            color = 'gray'
            weight = 'normal'
        
        ax.text(i, y_pos, p_text, ha='center', va='bottom', 
                fontsize=10, color=color, fontweight=weight,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Save figure if standalone plot
    if save_figure:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {output_path}")


def plot_spectra_comparison(sample_to_rates1: Dict[str, Dict[str, float]], 
                           sample_to_rates2: Dict[str, Dict[str, float]],
                           output_path: str,
                           filtering_stats1: dict = None,
                           filtering_stats2: dict = None,
                           depth_correction: str = 'filter',
                           title1: str = "Dataset 1",
                           title2: str = "Dataset 2"):
    """
    Plot two mutation spectra one on top of the other for comparison.
    
    Args:
        sample_to_rates1: First dataset rates
        sample_to_rates2: Second dataset rates
        output_path: Output file path
        filtering_stats1: Filtering stats for first dataset
        filtering_stats2: Filtering stats for second dataset
        depth_correction: Depth correction method
        title1: Title for first subplot
        title2: Title for second subplot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot first dataset
    plot_spectra(sample_to_rates1, output_path, filtering_stats1, depth_correction, ax=ax2, title=title1)
    
    # Plot second dataset
    plot_spectra(sample_to_rates2, output_path, filtering_stats2, depth_correction, ax=ax1, title=title2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description='Plot mutation spectra from count TSV files (matches plot_spectra.py logic)'
    )
    ap.add_argument('tsv_dir', nargs='+', help='Directory(ies) containing count TSV files (1 or 2 for comparison mode)')
    ap.add_argument('-o', '--out', required=True, help='Output plot file (e.g., spectra.png)')
    ap.add_argument('--pattern', default='*.counts', help='File pattern (default: *.counts)')
    ap.add_argument('--min-alt', type=int, default=1, help='Minimum alt reads (default: 1)')
    ap.add_argument('--min-depth', type=int, default=10, help='Minimum depth (default: 10)')
    ap.add_argument('--depth-correction', choices=['none', 'filter', 'normalize'], default='filter',
                    help='Depth correction method: none=no correction, filter=exclude low depth sites, normalize=depth-adjusted rates (default: filter)')
    ap.add_argument('--min-rate', type=float, default=0.01, 
                    help='Minimum mutation rate for normalize mode (default: 0.01 = 1%%)')
    ap.add_argument('--skip-nt-filter', action='store_true', 
                    help='Skip NT control position filtering')
    ap.add_argument('--max-nt-proportion', type=float, default=0.15,
                    help='Max proportion of NT controls with variants at position (default: 0.15)')
    ap.add_argument('--title1', default=None, help='Title for first plot/dataset')
    ap.add_argument('--title2', default=None, help='Title for second plot/dataset (comparison mode only)')
    args = ap.parse_args()
    
    # Check if comparison mode
    comparison_mode = len(args.tsv_dir) == 2
    
    if len(args.tsv_dir) > 2:
        print(f"Error: Expected 1 or 2 directories, got {len(args.tsv_dir)}")
        return 1
    
    # Process first dataset
    tsv_dir1 = args.tsv_dir[0]
    files = list_tsv_files(tsv_dir1, args.pattern)
    
    if not files:
        print(f"Error: No files matching '{args.pattern}' found in {tsv_dir1}")
        return 1
    
    print(f"=== Dataset 1 ===")
    print(f"Found {len(files)} files to process")
    print(f"Parameters: min_alt={args.min_alt}, min_depth={args.min_depth}, depth_correction={args.depth_correction}")
    if args.depth_correction == 'normalize':
        print(f"  min_rate={args.min_rate} (used in normalize mode)")
    
    # Build exclusion mask from NT controls
    excluded_positions1 = set()
    filtering_stats1 = {}
    if not args.skip_nt_filter:
        print("\nApplying NT control position filtering...")
        excluded_positions1 = filter_positions_by_nt_controls(
            tsv_dir1, args.pattern, args.max_nt_proportion, 
            args.min_alt, args.min_depth, args.min_rate, args.depth_correction
        )
        filtering_stats1 = {
            'positions_filtered': len(excluded_positions1),
            'max_nt_proportion': args.max_nt_proportion
        }
    else:
        print("\nSkipping NT control position filtering")
    
    # Process each file
    print("\nProcessing files...")
    sample_to_rates1 = {}
    for f in files:
        sample = os.path.splitext(os.path.basename(f))[0]
        print(f"  Processing {sample}...")
        rates = process_tsv_file(f, args.min_alt, args.min_depth, args.min_rate, 
                                args.depth_correction, excluded_positions1)
        if rates:
            sample_to_rates1[sample] = rates
    
    if not sample_to_rates1:
        print("Error: No valid data found in dataset 1.")
        return 1
    
    print(f"\nSuccessfully processed {len(sample_to_rates1)} samples")
    
    # Process second dataset if in comparison mode
    sample_to_rates2 = None
    filtering_stats2 = {}
    if comparison_mode:
        tsv_dir2 = args.tsv_dir[1]
        print(f"\n=== Dataset 2 ===")
        files2 = list_tsv_files(tsv_dir2, args.pattern)
        
        if not files2:
            print(f"Error: No files matching '{args.pattern}' found in {tsv_dir2}")
            return 1
        
        print(f"Found {len(files2)} files to process")
        
        # Build exclusion mask from NT controls
        excluded_positions2 = set()
        if not args.skip_nt_filter:
            print("\nApplying NT control position filtering...")
            excluded_positions2 = filter_positions_by_nt_controls(
                tsv_dir2, args.pattern, args.max_nt_proportion, 
                args.min_alt, args.min_depth, args.min_rate, args.depth_correction
            )
            filtering_stats2 = {
                'positions_filtered': len(excluded_positions2),
                'max_nt_proportion': args.max_nt_proportion
            }
        else:
            print("\nSkipping NT control position filtering")
        
        # Process each file
        print("\nProcessing files...")
        sample_to_rates2 = {}
        for f in files2:
            sample = os.path.splitext(os.path.basename(f))[0]
            print(f"  Processing {sample}...")
            rates = process_tsv_file(f, args.min_alt, args.min_depth, args.min_rate, 
                                    args.depth_correction, excluded_positions2)
            if rates:
                sample_to_rates2[sample] = rates
        
        if not sample_to_rates2:
            print("Error: No valid data found in dataset 2.")
            return 1
        
        print(f"\nSuccessfully processed {len(sample_to_rates2)} samples")
    
    # Save rates to TSV
    tsv_output = args.out.replace('.png', '.tsv').replace('.pdf', '.tsv')
    with open(tsv_output, 'w') as f:
        # Header
        all_subs = ['A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T', 
                    'G>A', 'G>C', 'G>T', 'T>A', 'T>C', 'T>G']
        f.write('sample\t' + '\t'.join(all_subs) + '\n')
        
        # Data from dataset 1
        for sample in sorted(sample_to_rates1.keys()):
            rates = sample_to_rates1[sample]
            f.write(sample)
            for sub in all_subs:
                f.write(f'\t{rates.get(sub, 0.0):.10g}')
            f.write('\n')
        
        # Data from dataset 2 if comparison mode
        if comparison_mode and sample_to_rates2:
            for sample in sorted(sample_to_rates2.keys()):
                rates = sample_to_rates2[sample]
                f.write(sample)
                for sub in all_subs:
                    f.write(f'\t{rates.get(sub, 0.0):.10g}')
                f.write('\n')
    
    print(f"Rates saved: {tsv_output}")
    
    # Plot
    if comparison_mode:
        title1 = args.title1 if args.title1 else f"Dataset 1 ({os.path.basename(tsv_dir1)})"
        title2 = args.title2 if args.title2 else f"Dataset 2 ({os.path.basename(args.tsv_dir[1])})"
        plot_spectra_comparison(sample_to_rates1, sample_to_rates2, args.out, 
                               filtering_stats1, filtering_stats2, args.depth_correction,
                               title1, title2)
    else:
        title = args.title1 if args.title1 else None
        plot_spectra(sample_to_rates1, args.out, filtering_stats1, args.depth_correction, title=title)
    
    # Print summary
    print("\n=== Summary ===")
    print("\nDataset 1:")
    for sample in sorted(sample_to_rates1.keys()):
        total_rate = sum(sample_to_rates1[sample].values())
        ct_rate = sample_to_rates1[sample].get('C>T', 0.0)
        print(f"  {sample}: total_rate={total_rate:.6f}, C>T={ct_rate:.6f}")
    
    if comparison_mode and sample_to_rates2:
        print("\nDataset 2:")
        for sample in sorted(sample_to_rates2.keys()):
            total_rate = sum(sample_to_rates2[sample].values())
            ct_rate = sample_to_rates2[sample].get('C>T', 0.0)
            print(f"  {sample}: total_rate={total_rate:.6f}, C>T={ct_rate:.6f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
