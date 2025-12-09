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


def load_rates_from_tsv(tsv_path: str) -> Dict[str, Dict[str, float]]:
    """Load mutation rates from a previously saved TSV file."""
    sample_to_rates = {}
    with open(tsv_path, 'r') as f:
        header = f.readline().strip().split('\t')
        # First column is 'sample', rest are substitution types
        substitutions = header[1:]
        
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            sample = parts[0]
            rates = {}
            for i, sub in enumerate(substitutions):
                try:
                    rates[sub] = float(parts[i + 1])
                except (IndexError, ValueError):
                    rates[sub] = 0.0
            sample_to_rates[sample] = rates
    
    return sample_to_rates


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
                     depth_correction: str, excluded_positions: set = None, 
                     rate_model: str = 'depth-weighted') -> Dict[str, float]:
    """
    Process a single TSV file to compute mutation rates.
    Matches plot_spectra.py logic exactly.
    
    depth_correction: 'filter' (default) or 'normalize'
    rate_model: 
        - 'depth-weighted' (default): (depth at mutant sites) / (total depth at ref sites)
        - 'finite-sites': (alt allele counts) / (total depth at ref sites) - matches estimate_rates.py
    
    Returns dict of substitution -> rate
    """
    has_header, col_map = detect_columns(tsv_path)
    
    # All 12 substitution types - initialize early so we always return this structure
    all_subs = ['A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T', 
                'G>A', 'G>C', 'G>T', 'T>A', 'T>C', 'T>G']
    
    # Need at least chrom, pos, ref
    for req in ['chrom', 'pos', 'ref']:
        if req not in col_map:
            # Return dict with all zeros for invalid file format
            return {sub: 0.0 for sub in all_subs}

    # Must be wide format
    if not all(k in col_map for k in ['A_count', 'C_count', 'G_count', 'T_count', 'depth']):
        # Return dict with all zeros for invalid file format
        return {sub: 0.0 for sub in all_subs}

    # Track depth-weighted mutant sites per type (used by depth-weighted model)
    mutant_depth_by_type: Dict[str, float] = defaultdict(float)
    # Track alt allele counts per type (used by finite-sites model)
    alt_allele_counts_by_type: Dict[str, float] = defaultdict(float)
    # Track total depth per reference base  
    total_depth_by_ref: Dict[str, float] = defaultdict(float)
    # Track which sites we've counted (prevent double-counting)
    seen_site_for_type: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    seen_site_for_ref: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    
    # Diagnostic counters
    sites_processed = 0
    sites_passed_depth = 0
    sites_excluded = 0

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
            sites_processed += 1
            
            # Skip excluded positions
            if excluded_positions and site_key in excluded_positions:
                sites_excluded += 1
                continue

            # Get depth
            depth_val = None
            if 'depth' in col_map and col_map['depth'] < len(parts):
                try:
                    depth_val = float(parts[col_map['depth']])
                except ValueError:
                    depth_val = None
            if depth_val is None:
                continue  # Skip sites with no depth value
            
            # Get base counts (needed for both models)
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

            # For finite-sites model: count alt allele counts (matching estimate_rates.py exactly)
            # - NO min_depth filter (estimate_rates.py doesn't apply one)
            # - Count alt allele counts for all sites where alt_count >= min_alt
            # - Add depth to denominator for ALL sites (not just those passing min_depth)
            if rate_model == 'finite-sites':
                # Add depth to denominator for ALL sites (matching estimate_rates.py line 621)
                if site_key not in seen_site_for_ref[ref]:
                    total_depth_by_ref[ref] += depth_val
                    seen_site_for_ref[ref].add(site_key)
                
                # Count alt allele counts (matching estimate_rates.py line 633)
                # estimate_rates.py only counts if ems_count >= min_alt
                for alt in ['A', 'C', 'G', 'T']:
                    if alt == ref:
                        continue
                    alt_count = counts.get(alt, 0.0)
                    sub_name = f"{ref}>{alt}"
                    
                    # Apply filters based on depth_correction mode
                    is_mutant = False
                    if depth_correction == 'normalize':
                        # For normalize mode: require min_alt count AND min_rate threshold (if min_rate > 0)
                        count_pass = alt_count >= min_alt
                        if min_rate > 0:
                            rate_pass = (alt_count / depth_val) >= min_rate if depth_val > 0 else False
                            is_mutant = count_pass and rate_pass
                        else:
                            # If min_rate is 0, only check min_alt
                            is_mutant = count_pass
                    else:
                        # For filter mode (default): only check min_alt (matching estimate_rates.py line 631)
                        is_mutant = alt_count >= min_alt
                    
                    if is_mutant:
                        # Count alt allele counts (not depth) - matching estimate_rates.py
                        alt_allele_counts_by_type[sub_name] += alt_count
            else:
                # Depth-weighted model: apply min_depth filter
                if depth_val < min_depth:
                    continue
                sites_passed_depth += 1
                if site_key not in seen_site_for_ref[ref]:
                    total_depth_by_ref[ref] += depth_val
                    seen_site_for_ref[ref].add(site_key)
                
                # Check each substitution for depth-weighted model
                for alt in ['A', 'C', 'G', 'T']:
                    if alt == ref:
                        continue
                    alt_count = counts.get(alt, 0.0)
                    sub_name = f"{ref}>{alt}"
                    
                    # Apply filters based on depth_correction mode
                    is_mutant = False
                    if depth_correction == 'normalize':
                        # For normalize mode: require min_alt count AND min_rate threshold (if min_rate > 0)
                        count_pass = alt_count >= min_alt
                        if min_rate > 0:
                            rate_pass = (alt_count / depth_val) >= min_rate if depth_val > 0 else False
                            is_mutant = count_pass and rate_pass
                        else:
                            # If min_rate is 0, only check min_alt
                            is_mutant = count_pass
                    else:
                        # For filter mode (default): only check min_alt
                        is_mutant = alt_count >= min_alt
                    
                    if is_mutant:
                        # Add depth to numerator (once per site per type)
                        if site_key not in seen_site_for_type[sub_name]:
                            mutant_depth_by_type[sub_name] += depth_val
                            seen_site_for_type[sub_name].add(site_key)

    # Calculate rates based on model
    # all_subs already defined at top of function
    rates = {}
    if rate_model == 'finite-sites':
        # Finite-sites model: (alt allele counts) / (total depth at ref sites)
        # Matching estimate_rates.py finite_sites_mutations calculation exactly:
        # - NO min_depth filter (estimate_rates.py doesn't apply one)
        # - Count alt allele counts for all sites where alt_count >= min_alt
        # - Rate = alt_allele_counts / total_depth_at_ref_sites
        total_alt_alleles = sum(alt_allele_counts_by_type.values())
        total_depth_all = sum(total_depth_by_ref.values())
        print(f"  Finite-sites diagnostics: sites_processed={sites_processed}, sites_excluded={sites_excluded}")
        print(f"    (NO min_depth filter applied - matching estimate_rates.py)")
        if total_depth_all == 0:
            print(f"  WARNING: total_depth_all is 0 for {os.path.basename(tsv_path)} in finite-sites mode")
            print(f"    This means no sites found or all were excluded")
        else:
            if total_alt_alleles == 0:
                print(f"  WARNING: total_alt_alleles is 0 for {os.path.basename(tsv_path)} in finite-sites mode (total_depth={total_depth_all:.0f})")
                print(f"    This means no mutations passed the min_alt filter (min_alt={min_alt})")
        for sub in all_subs:
            ref = sub.split('>')[0]
            num = alt_allele_counts_by_type.get(sub, 0.0)  # Use alt allele counts
            denom = total_depth_by_ref.get(ref, 0.0)  # Use depth from ALL sites (no min_depth filter)
            rates[sub] = (num / denom) if denom > 0 else 0.0
    else:
        # Depth-weighted model: (depth at mutant sites) / (total depth at ref sites)
        for sub in all_subs:
            ref = sub.split('>')[0]
            denom = total_depth_by_ref.get(ref, 0.0)
            num = mutant_depth_by_type.get(sub, 0.0)
            rates[sub] = (num / denom) if denom > 0 else 0.0

    # Always return a dict with all substitution types, even if all zeros
    # This ensures the sample is included in output
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
                                # For normalize: check min_alt and min_rate (if min_rate > 0)
                                count_pass = count >= min_alt
                                if min_rate > 0:
                                    rate_pass = (count / depth_val) >= min_rate if depth_val > 0 else False
                                    is_mutant = count_pass and rate_pass
                                else:
                                    # If min_rate is 0, only check min_alt
                                    is_mutant = count_pass
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
                 filtering_stats: dict = None, depth_correction: str = 'filter', ax=None, title=None,
                 color_dict: Dict[str, str] = None, suppress_legend: bool = False):
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
        color_dict: Optional pre-computed color mapping (if None, computed internally)
        suppress_legend: If True, don't create a legend (for shared legends in comparison plots)
    
    Returns:
        Dict mapping sample names to colors that were used
    
    Returns:
        If return_legend_info is True, returns (legend_handles, legend_labels, sample_label_map, NT_samples_sorted)
        Otherwise returns None
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
    
    # Diagnostic output
    print(f"\nPlotting data summary:")
    print(f"  Total rows in dataframe: {len(df)}")
    print(f"  Unique samples: {df['sample'].nunique()}")
    print(f"  Unique substitutions: {df['substitution'].nunique()}")
    print(f"  Non-zero proportions: {(df['proportion'] > 0).sum()} / {len(df)}")
    if len(df) > 0:
        print(f"  Proportion range: [{df['proportion'].min():.2e}, {df['proportion'].max():.2e}]")
        print(f"  Sample rate totals:")
        for sample in df['sample'].unique():
            sample_total = df[df['sample'] == sample]['proportion'].sum()
            sample_nonzero = (df[df['sample'] == sample]['proportion'] > 0).sum()
            print(f"    {sample}: total={sample_total:.2e}, non-zero={sample_nonzero}/12")
    
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
    
    # Use provided color_dict if available, otherwise create one
    if color_dict is None:
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
            print(f"    Creating gradient for {n_treated} treated samples")
            if n_treated == 1:
                colors.append('#FF6B35')  # Single orange-red color
                print(f"      Single treated sample, using #FF6B35")
            else:
                # Create gradient from light orange to dark red
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'orange_red', ['#FFB366', '#FF8C42', '#FF6B35', '#E63946', '#C1121F']
                )
                treated_colors = [mcolors.rgb2hex(cmap(i / (n_treated - 1))) for i in range(n_treated)]
                print(f"      Created {len(treated_colors)} gradient colors: {treated_colors}")
                colors.extend(treated_colors)
        
        color_dict = dict(zip(sample_order, colors))
    else:
        # Use provided color_dict, but ensure all samples have colors (use default if missing)
        NT_GREY = '#808080'
        for sample in sample_order:
            if sample not in color_dict:
                # Try cleaned label too for cross-panel compatibility
                cleaned = clean_sample_label(sample)
                if cleaned in color_dict:
                    color_dict[sample] = color_dict[cleaned]
                else:
                    # Assign grey for NT, default orange for others
                    if sample in NT_samples_sorted:
                        color_dict[sample] = NT_GREY
                    else:
                        color_dict[sample] = '#FF6B35'
    
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
    # Handle missing samples in provided color_dict
    NT_GREY = '#808080'
    color_dict_display = {}
    for s in sample_order:
        display_label = sample_label_map[s]
        # Try to find color: first by original sample name, then by cleaned label
        if s in color_dict:
            color_dict_display[display_label] = color_dict[s]
        elif display_label in color_dict:
            # Try cleaned label
            color_dict_display[display_label] = color_dict[display_label]
        else:
            # Default color if not in provided color_dict
            if s in NT_samples_sorted:
                color_dict_display[display_label] = NT_GREY
            else:
                color_dict_display[display_label] = '#FF6B35'
    
    # Convert to list in display_order - seaborn applies colors in order
    palette_list = [color_dict_display[label] for label in display_order]
    
    # Create custom substitution order with complementary pairs adjacent (matching plot_spectra.py)
    substitution_order = ["A>C", "T>G", "A>G", "T>C", "A>T", "T>A", 
                         "C>A", "G>T", "C>G", "G>C", "C>T", "G>A"]
    
    # Sort by custom substitution order, then by sample display order
    df["sample_display"] = pd.Categorical(df["sample_display"], categories=display_order, ordered=True)
    df["substitution"] = pd.Categorical(df["substitution"], categories=substitution_order, ordered=True)
    df = df.sort_values(["substitution", "sample_display"])
    
    # Perform Mann-Whitney U tests
    p_values = perform_mann_whitney_tests(df, NT_samples_sorted, other_samples_sorted, substitution_order)
    
    # Check if all values are zero or below log scale threshold
    max_proportion = df['proportion'].max()
    if max_proportion == 0.0:
        print("WARNING: All mutation rates are 0.0 - plot will appear empty!")
        print("  This may indicate:")
        print("    - No mutations found in the data")
        print("    - All sites filtered out (check min_depth, min_alt, excluded positions)")
        print("    - Issue with finite-sites calculation (check total_depth_all)")
    elif max_proportion < 1e-4:
        print(f"WARNING: Maximum rate ({max_proportion:.2e}) is below log scale minimum (1e-4)")
        print("  Some bars may not be visible on the plot")
    
    # Create figure if ax not provided
    if ax is None:
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        save_figure = True
    else:
        save_figure = False
   
    # Filter out zero values for barplot (they won't show on log scale anyway)
    # But keep the dataframe structure for proper axis labels
    df_plot = df.copy()
    # Suppress seaborn's automatic legend if suppress_legend is True
    show_legend_in_barplot = not suppress_legend
    
    if max_proportion > 0:
        # Only filter zeros if we have some non-zero data
        # This ensures we still get proper axis labels even if some values are zero
        bp = sns.barplot(
            data=df_plot,
            x="substitution",
            y="proportion",
            hue="sample_display",
            palette=palette_list,
            hue_order=display_order,
            ax=ax,
            legend=show_legend_in_barplot
        )
    else:
        # If all zeros, still create the plot structure but with empty bars
        # This ensures the plot has proper labels and structure
        bp = sns.barplot(
            data=df_plot,
            x="substitution",
            y="proportion",
            hue="sample_display",
            palette=palette_list,
            hue_order=display_order,
            ax=ax,
            legend=show_legend_in_barplot
        )
        print("  Creating plot structure with zero values (bars will not be visible on log scale)")
    
    # Extract actual colors from the plot patches - this is critical!
    # We need the EXACT colors seaborn assigned, not what we think they should be
    actual_color_dict = {}
    if suppress_legend and len(bp.patches) > 0:
        # Extract from patches - get first patch for each sample
        seen_samples = set()
        n_subs = len(substitution_order)
        n_samples = len(display_order)
        print(f"    DEBUG: Extracting colors from {len(bp.patches)} patches, {n_subs} substitutions, {n_samples} samples")
        if n_samples > 0:
            for i, patch in enumerate(bp.patches):
                sub_idx = i // n_samples
                sample_idx = i % n_samples
                # Only extract from first substitution (sub_idx == 0) for each sample
                if sub_idx == 0 and sample_idx < len(display_order):
                    label = display_order[sample_idx]
                    if label not in seen_samples:
                        seen_samples.add(label)
                        # Get color from patch - this is the ACTUAL color seaborn used
                        color = patch.get_facecolor()
                        if hasattr(color, '__len__') and len(color) >= 3:
                            try:
                                if len(color) >= 4:  # RGBA
                                    color_hex = mcolors.rgb2hex(color[:3])
                                else:  # RGB
                                    color_hex = mcolors.rgb2hex(color)
                                # Map display label back to original sample name
                                for orig_sample, display_label in sample_label_map.items():
                                    if display_label == label:
                                        actual_color_dict[orig_sample] = color_hex
                                        print(f"      Extracted: {orig_sample} (display: {label}) -> {color_hex}")
                                        break
                            except Exception as e:
                                print(f"      ERROR extracting color for {label}: {e}")
                                pass
    else:
        # If legend is shown, extract from legend handles
        handles, labels = ax.get_legend_handles_labels()
        print(f"    DEBUG: Extracting colors from {len(handles)} legend handles")
        for handle, label in zip(handles, labels):
            color = handle.get_facecolor()
            if hasattr(color, '__len__') and len(color) >= 3:
                try:
                    if len(color) >= 4:  # RGBA
                        color_hex = mcolors.rgb2hex(color[:3])
                    else:  # RGB
                        color_hex = mcolors.rgb2hex(color)
                    # Map display label back to original sample name
                    for orig_sample, display_label in sample_label_map.items():
                        if display_label == label:
                            actual_color_dict[orig_sample] = color_hex
                            print(f"      Extracted: {orig_sample} (display: {label}) -> {color_hex}")
                            break
                except Exception as e:
                    print(f"      ERROR extracting color for {label}: {e}")
                    pass
    
    # Build ylabel matching plot_spectra.py
    ylabel = "Proportion of sites with mutant alleles"
    
            
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_yscale('log')  # Log scale for y-axis
    ax.set_ylim(bottom=1e-4)  # Set minimum y value to 10^-4
    
    # Adjust upper y-limit to accommodate p-value boxes
    current_ylim = ax.get_ylim()
    ax.set_ylim(top=current_ylim[1] * 1.3)  # Increase upper limit by 30%
    
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    ax.set_xlabel('')  # Remove x-axis label
    
    # Create custom legend that groups NT controls (unless suppressed)
    if not suppress_legend:
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
        
        if legend_handles:
            ax.legend(legend_handles, legend_labels, fontsize=14, bbox_to_anchor=(1.05, 0.5), loc="center left")
    
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
                fontsize=9, color=color, fontweight=weight,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=22, fontweight='bold')
    
    # Save figure if standalone plot
    if save_figure:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {output_path}")
    
    # Return the color_dict we created
    return color_dict


def plot_spectra_finite_sites(sample_to_rates: Dict[str, Dict[str, float]], output_path: str, 
                              filtering_stats: dict = None, ax=None, title=None,
                              color_dict: Dict[str, str] = None, suppress_legend: bool = False,
                              extract_colors_from_legend: bool = False):
    """
    Plot mutation spectra for finite-sites rate model.
    
    Args:
        sample_to_rates: Dict mapping sample names to substitution rates
        output_path: Output file path (ignored if ax is provided)
        filtering_stats: Dict with filtering statistics
        ax: Optional matplotlib axis to plot on (for subplots)
        title: Optional title for the plot
        color_dict: Optional pre-computed color mapping (if None, computed internally)
        suppress_legend: If True, don't create a legend (for shared legends in comparison plots)
        extract_colors_from_legend: If True, temporarily show legend to extract colors, then remove it
    
    Returns:
        Dict mapping sample names to colors that were used
        If return_legend_info is True, returns (legend_handles, legend_labels, sample_label_map, NT_samples_sorted)
        Otherwise returns None
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
    
    # Diagnostic output
    print(f"\nPlotting finite-sites data summary:")
    print(f"  Total rows in dataframe: {len(df)}")
    print(f"  Unique samples: {df['sample'].nunique()}")
    print(f"  Unique substitutions: {df['substitution'].nunique()}")
    print(f"  Non-zero proportions: {(df['proportion'] > 0).sum()} / {len(df)}")
    if len(df) > 0:
        print(f"  Rate range: [{df['proportion'].min():.2e}, {df['proportion'].max():.2e}]")
        print(f"  Sample rate totals:")
        for sample in df['sample'].unique():
            sample_total = df[df['sample'] == sample]['proportion'].sum()
            sample_nonzero = (df[df['sample'] == sample]['proportion'] > 0).sum()
            print(f"    {sample}: total={sample_total:.2e}, non-zero={sample_nonzero}/12")
    
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
    
    # Use provided color_dict if available, otherwise create one
    if color_dict is None:
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
            print(f"    Creating gradient for {n_treated} treated samples")
            if n_treated == 1:
                colors.append('#FF6B35')  # Single orange-red color
                print(f"      Single treated sample, using #FF6B35")
            else:
                # Create gradient from light orange to dark red
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'orange_red', ['#FFB366', '#FF8C42', '#FF6B35', '#E63946', '#C1121F']
                )
                treated_colors = [mcolors.rgb2hex(cmap(i / (n_treated - 1))) for i in range(n_treated)]
                print(f"      Created {len(treated_colors)} gradient colors: {treated_colors}")
                colors.extend(treated_colors)
        
        color_dict = dict(zip(sample_order, colors))
    else:
        # Use provided color_dict, but ensure all samples have colors (use default if missing)
        NT_GREY = '#808080'
        for sample in sample_order:
            if sample not in color_dict:
                # Try cleaned label too for cross-panel compatibility
                cleaned = clean_sample_label(sample)
                if cleaned in color_dict:
                    color_dict[sample] = color_dict[cleaned]
                else:
                    # Assign grey for NT, default orange for others
                    if sample in NT_samples_sorted:
                        color_dict[sample] = NT_GREY
                    else:
                        color_dict[sample] = '#FF6B35'
    
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
    # Handle missing samples in provided color_dict
    NT_GREY = '#808080'
    color_dict_display = {}
    for s in sample_order:
        display_label = sample_label_map[s]
        # Try to find color: first by original sample name, then by cleaned label
        if s in color_dict:
            color_dict_display[display_label] = color_dict[s]
        elif display_label in color_dict:
            # Try cleaned label
            color_dict_display[display_label] = color_dict[display_label]
        else:
            # Default color if not in provided color_dict
            if s in NT_samples_sorted:
                color_dict_display[display_label] = NT_GREY
            else:
                color_dict_display[display_label] = '#FF6B35'
    
    # Convert to list in display_order - seaborn applies colors in order
    palette_list = [color_dict_display[label] for label in display_order]
    
    # Create custom substitution order with complementary pairs adjacent (matching plot_spectra.py)
    substitution_order = ["A>C", "T>G", "A>G", "T>C", "A>T", "T>A", 
                         "C>A", "G>T", "C>G", "G>C", "C>T", "G>A"]
    
    # Sort by custom substitution order, then by sample display order
    df["sample_display"] = pd.Categorical(df["sample_display"], categories=display_order, ordered=True)
    df["substitution"] = pd.Categorical(df["substitution"], categories=substitution_order, ordered=True)
    df = df.sort_values(["substitution", "sample_display"])
    
    # Perform Mann-Whitney U tests
    p_values = perform_mann_whitney_tests(df, NT_samples_sorted, other_samples_sorted, substitution_order)
    
    # Check if all values are zero
    max_rate = df['proportion'].max()
    min_rate = df[df['proportion'] > 0]['proportion'].min() if (df['proportion'] > 0).any() else None
    
    if max_rate == 0.0:
        print("WARNING: All mutation rates are 0.0 - plot will appear empty!")
        print("  This may indicate:")
        print("    - No mutations found in the data")
        print("    - All sites filtered out (check min_alt, excluded positions)")
        print("    - Issue with finite-sites calculation (check total_depth_all)")
    
    # Create figure if ax not provided
    if ax is None:
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        save_figure = True
    else:
        save_figure = False
   
    # Filter out zero values for barplot (they won't show on log scale anyway)
    # But keep the dataframe structure for proper axis labels
    df_plot = df.copy()
    # Suppress seaborn's automatic legend if suppress_legend is True
    # But if extract_colors_from_legend is True, we need to show it temporarily
    show_legend_in_barplot = not suppress_legend or extract_colors_from_legend
    if max_rate > 0:
        # Only filter zeros if we have some non-zero data
        # This ensures we still get proper axis labels even if some values are zero
        bp = sns.barplot(
            data=df_plot,
            x="substitution",
            y="proportion",
            hue="sample_display",
            palette=palette_list,
            hue_order=display_order,
            ax=ax,
            legend=show_legend_in_barplot
        )
    else:
        # If all zeros, still create the plot structure but with empty bars
        # This ensures the plot has proper labels and structure
        bp = sns.barplot(
            data=df_plot,
            x="substitution",
            y="proportion",
            hue="sample_display",
            palette=palette_list,
            hue_order=display_order,
            ax=ax,
            legend=show_legend_in_barplot
        )
        print("  Creating plot structure with zero values (bars will not be visible on log scale)")
    
    # Extract actual colors from legend handles - this is the ONLY reliable method!
    # Patches don't work (they're all grey), but legend handles have the actual colors
    actual_color_dict = {}
    if extract_colors_from_legend or not suppress_legend:
        print("  Extracting colors from legend handles")
        # Force a draw to ensure legend is rendered before extraction
        if ax.figure:
            ax.figure.canvas.draw()
        # Extract from legend handles - this is the ONLY method that works
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            color = handle.get_facecolor()
            if hasattr(color, '__len__') and len(color) >= 3:
                try:
                    if len(color) >= 4:  # RGBA
                        color_hex = mcolors.rgb2hex(color[:3])
                    else:  # RGB
                        color_hex = mcolors.rgb2hex(color)
                    # Map display label back to original sample name
                    for orig_sample, display_label in sample_label_map.items():
                        if display_label == label:
                            actual_color_dict[orig_sample] = color_hex
                            break
                except Exception:
                    pass
        
        # If we extracted colors from legend but want to suppress it, remove the legend now
        if extract_colors_from_legend and suppress_legend:
            ax.get_legend().remove()
    
    # Set y-axis label for finite-sites model
    ax.set_ylabel("Finite-sites model mutation rate", fontsize=20)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_yscale('log')  # Log scale for y-axis
    
    # Set y-axis bounds to match data ranges
    if max_rate > 0 and min_rate is not None:
        # Calculate appropriate bounds based on data
        # Set bottom to slightly below minimum non-zero value
        y_min = min_rate * 0.5  # Half of minimum to give some padding
        # Set top to accommodate p-value boxes and give some padding
        y_max = max_rate * 1.5  # 50% padding above maximum
        ax.set_ylim(bottom=y_min, top=y_max)
        print(f"  Y-axis bounds set to data range: [{y_min:.2e}, {y_max:.2e}]")
    elif max_rate > 0:
        # If we have data but no non-zero minimum (shouldn't happen, but handle it)
        y_min = max_rate * 1e-4  # Set minimum to 4 orders of magnitude below max
        y_max = max_rate * 1.5
        ax.set_ylim(bottom=y_min, top=y_max)
        print(f"  Y-axis bounds set to: [{y_min:.2e}, {y_max:.2e}]")
    else:
        # All zeros - set default range
        ax.set_ylim(bottom=1e-6, top=1e-2)
        print("  Y-axis bounds set to default (all rates are zero)")
    
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    ax.set_xlabel('')  # Remove x-axis label
    
    # Create custom legend that groups NT controls (unless suppressed)
    # Note: If extract_colors_from_legend was True, legend was already removed
    if not suppress_legend and not extract_colors_from_legend:
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
        
        if legend_handles:
            ax.legend(legend_handles, legend_labels, fontsize=14, bbox_to_anchor=(1.05, 0.5), loc="center left")
    
    # Add p-value annotations below the title but above the bars
    y_max_plot = ax.get_ylim()[1]
    y_pos = y_max_plot * 0.72  # Position lower to avoid overlap with bars
    
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
                fontsize=9, color=color, fontweight=weight,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=22, fontweight='bold')
    
    # Save figure if standalone plot
    if save_figure:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {output_path}")
    
    # Return extracted colors if available (from legend handles), otherwise use color_dict
    if actual_color_dict and len(actual_color_dict) > 0:
        print("  Returning extracted colors from legend handles")
        # Use extracted colors - these are the actual colors seaborn used
        final_color_dict = color_dict.copy()
        final_color_dict.update(actual_color_dict)
        # Also add entries keyed by cleaned labels for cross-panel compatibility
        for sample, color in list(final_color_dict.items()):
            cleaned = clean_sample_label(sample)
            if cleaned != sample:
                final_color_dict[cleaned] = color
        return final_color_dict
    else:
        print("  Returning color_dict (no colors extracted from legend)")
        # Add entries keyed by cleaned labels for cross-panel compatibility
        for sample, color in list(color_dict.items()):
            cleaned = clean_sample_label(sample)
            if cleaned != sample:
                color_dict[cleaned] = color
        return color_dict


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
    Uses colors from first plot for second plot, with single shared legend.
    
    Args:
        sample_to_rates1: First dataset rates (upper panel, determines colors)
        sample_to_rates2: Second dataset rates (lower panel, uses colors from first)
        output_path: Output file path
        filtering_stats1: Filtering stats for first dataset
        filtering_stats2: Filtering stats for second dataset
        depth_correction: Depth correction method
        title1: Title for first subplot
        title2: Title for second subplot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot first dataset (upper panel) with no legend - this determines the colors
    print("\n=== Plotting first dataset (non-consensus, top panel) ===")
    color_dict = plot_spectra(sample_to_rates1, output_path, filtering_stats1, depth_correction, 
                             ax=ax1, title=title1, suppress_legend=True)
    
    # Plot second dataset (lower panel) using the exact same colors, no legend
    print("\n=== Plotting second dataset (consensus, bottom panel) ===")
    plot_spectra(sample_to_rates2, output_path, filtering_stats2, depth_correction, 
                ax=ax2, title=title2, color_dict=color_dict, suppress_legend=True)
    
    # Create single shared legend from the color_dict
    # Get sample order from first dataset
    all_samples = list(sample_to_rates1.keys())
    NT_samples = [s for s in all_samples if "NT" in s]
    other_samples = [s for s in all_samples if "NT" not in s]
    
    # Get C>T rates for sorting
    ct_rates = {}
    for sample in all_samples:
        if sample in sample_to_rates1:
            ct_rates[sample] = sample_to_rates1[sample].get('C>T', 0.0)
        else:
            ct_rates[sample] = 0.0
    
    NT_samples_sorted = sorted(NT_samples, key=lambda x: ct_rates.get(x, 0))
    other_samples_sorted = sorted(other_samples, key=lambda x: ct_rates.get(x, 0))
    
    # Create clean labels
    sample_label_map = {}
    seen_labels = {}
    for s in all_samples:
        base_label = clean_sample_label(s)
        if base_label in seen_labels:
            seen_labels[base_label] += 1
            unique_label = f"{base_label}_{seen_labels[base_label]}"
        else:
            seen_labels[base_label] = 0
            unique_label = base_label
        sample_label_map[s] = unique_label
    
    # Create legend handles and labels
    import matplotlib.patches as mpatches
    legend_handles = []
    legend_labels = []
    
    # Add NT controls entry (single entry for all NT samples)
    if NT_samples_sorted:
        # Use the color from the first NT sample
        nt_sample = NT_samples_sorted[0]
        nt_color = color_dict.get(nt_sample, '#808080')
        legend_handles.append(mpatches.Patch(color=nt_color))
        legend_labels.append("NT controls")
    
    # Add individual treated sample entries
    for sample in other_samples_sorted:
        color = color_dict.get(sample, '#FF6B35')
        label = sample_label_map.get(sample, sample)
        legend_handles.append(mpatches.Patch(color=color))
        legend_labels.append(label)
    
    # Create shared legend on the figure
    if legend_handles:
        fig.legend(legend_handles, legend_labels, fontsize=14, bbox_to_anchor=(1.02, 0.5), loc="center left")
    
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
    ap.add_argument('--min-rate', type=float, default=0.0, 
                    help='Minimum mutation rate for normalize mode (default: 0.0 = no rate filter)')
    ap.add_argument('--skip-nt-filter', action='store_true', 
                    help='Skip NT control position filtering')
    ap.add_argument('--max-nt-proportion', type=float, default=0.15,
                    help='Max proportion of NT controls with variants at position (default: 0.15)')
    ap.add_argument('--title1', default=None, help='Title for first plot/dataset')
    ap.add_argument('--title2', default=None, help='Title for second plot/dataset (comparison mode only)')
    ap.add_argument('--rate-model', choices=['depth-weighted', 'finite-sites'], default='depth-weighted',
                    help='Rate calculation model: depth-weighted (default) or finite-sites (from estimate_rates.py high_rate)')
    args = ap.parse_args()
    
    # Check if user requested comparison mode
    user_comparison_mode = len(args.tsv_dir) == 2
    
    # Check for existing TSV file first
    tsv_output = args.out.replace('.png', '.tsv').replace('.pdf', '.tsv')
    tsv_fully_loaded = False  # Flag to skip count file processing
    
    if os.path.exists(tsv_output):
        print(f"\n=== Found existing TSV: {tsv_output} ===")
        try:
            loaded_rates = load_rates_from_tsv(tsv_output)
            print(f"Loaded {len(loaded_rates)} samples from TSV")
            
            # Categorize samples
            # Samples with "initial" or "inital" in name are non-consensus
            consensus_samples = {}
            nonconsensus_samples = {}
            for sample, rates in loaded_rates.items():
                if 'initial' in sample.lower() or 'inital' in sample.lower():
                    nonconsensus_samples[sample] = rates
                else:
                    consensus_samples[sample] = rates
            
            tsv_has_both = len(consensus_samples) > 0 and len(nonconsensus_samples) > 0
            tsv_has_consensus_only = len(consensus_samples) > 0 and len(nonconsensus_samples) == 0
            tsv_has_nonconsensus_only = len(nonconsensus_samples) > 0 and len(consensus_samples) == 0
            
            if not user_comparison_mode:
                # Single directory mode - use TSV as-is
                if tsv_has_both:
                    print(f"Warning: TSV has both types but single-dir mode requested. Skipping TSV reuse.\n")
                    tsv_fully_loaded = False
                else:
                    sample_to_rates1 = loaded_rates
                    sample_to_rates2 = {}
                    comparison_mode = False
                    filtering_stats1 = {}
                    filtering_stats2 = {}
                    tsv_fully_loaded = True
                    print(f"Using TSV data ({len(loaded_rates)} samples)\n")
            else:
                # Comparison mode (2 directories)
                if tsv_has_both:
                    # Perfect - TSV has both datasets
                    sample_to_rates1 = consensus_samples
                    sample_to_rates2 = nonconsensus_samples
                    comparison_mode = True
                    filtering_stats1 = {}
                    filtering_stats2 = {}
                    tsv_fully_loaded = True
                    print(f"Using TSV data: {len(sample_to_rates1)} consensus, {len(sample_to_rates2)} non-consensus\n")
                else:
                    # TSV has only one type - will reuse what we have and process the other
                    print(f"TSV contains only {'consensus' if tsv_has_consensus_only else 'non-consensus'} samples.")
                    print(f"Will reuse TSV for that dataset and process count files for the other.\n")
                    # Set loaded_rates to partial data; will be handled below
                    if tsv_has_consensus_only:
                        sample_to_rates1 = consensus_samples
                        sample_to_rates2 = None  # Signal to process this
                    else:
                        sample_to_rates1 = None  # Signal to process this
                        sample_to_rates2 = nonconsensus_samples
                    comparison_mode = True
                    filtering_stats1 = {}
                    filtering_stats2 = {}
                    tsv_fully_loaded = False  # Will need to process count files for missing dataset
                    
        except Exception as e:
            print(f"Error loading TSV: {e}")
            print("Will process count files instead...\n")
            tsv_fully_loaded = False
    
    # Only process count files if TSV wasn't fully loaded
    if not tsv_fully_loaded:
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
        print(f"Parameters: min_alt={args.min_alt}, min_depth={args.min_depth}, depth_correction={args.depth_correction}, rate_model={args.rate_model}")
        if args.rate_model == 'finite-sites':
            print(f"  Note: finite-sites mode now applies min_depth filter and uses depth-weighted counting")
            print(f"        (same filtering and counting logic as depth-weighted model)")
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
                                    args.depth_correction, excluded_positions1, args.rate_model)
            # Always add sample if rates dict is returned (even if all zeros)
            # The function now always returns a dict with 12 substitution types
            if rates and len(rates) > 0:
                sample_to_rates1[sample] = rates
                if args.rate_model == 'finite-sites':
                    total_rate = sum(rates.values())
                    num_nonzero = len([r for r in rates.values() if r > 0])
                    print(f"    Finite-sites: total_rate={total_rate:.6e}, num_substitutions={num_nonzero}/12, dict_size={len(rates)}")
            else:
                print(f"    Warning: No rates returned for {sample} (rates={rates})")
        
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
                                        args.depth_correction, excluded_positions2, args.rate_model)
                # Always add sample if rates dict is returned (even if all zeros)
                # The function now always returns a dict with 12 substitution types
                if rates and len(rates) > 0:
                    sample_to_rates2[sample] = rates
                    if args.rate_model == 'finite-sites':
                        total_rate = sum(rates.values())
                        num_nonzero = len([r for r in rates.values() if r > 0])
                        print(f"    Finite-sites: total_rate={total_rate:.6e}, num_substitutions={num_nonzero}/12, dict_size={len(rates)}")
                else:
                    print(f"    Warning: No rates returned for {sample} (rates={rates})")
            
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
        # Generate titles - handle case where TSV was loaded instead of processing dirs
        if loaded_rates is not None:
            title1 = args.title1 if args.title1 else "Dataset 1"
            title2 = args.title2 if args.title2 else "Dataset 2"
        else:
            title1 = args.title1 if args.title1 else f"Dataset 1 ({os.path.basename(tsv_dir1)})"
            title2 = args.title2 if args.title2 else f"Dataset 2 ({os.path.basename(args.tsv_dir[1])})"
        if args.rate_model == 'finite-sites':
            # For finite-sites, create stacked plots with shared legend
            # Non-consensus on top (ax1), consensus on bottom (ax2)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # Plot non-consensus dataset (upper panel) - extract colors from legend, then remove it
            print("\n=== Plotting non-consensus dataset (top panel) - finite-sites ===")
            color_dict = plot_spectra_finite_sites(sample_to_rates2, args.out, filtering_stats2, 
                                                  ax=ax1, title=title2, suppress_legend=True, 
                                                  extract_colors_from_legend=True)
            
            # Plot consensus dataset (lower panel) using the exact same colors, no legend
            print("\n=== Plotting consensus dataset (bottom panel) - finite-sites ===")
            plot_spectra_finite_sites(sample_to_rates1, args.out, filtering_stats1, 
                                     ax=ax2, title=title1, color_dict=color_dict, suppress_legend=True)
            
            # Create single shared legend from the color_dict
            # Get sample order from non-consensus dataset (which determined the colors)
            all_samples = list(sample_to_rates2.keys())
            NT_samples = [s for s in all_samples if "NT" in s]
            other_samples = [s for s in all_samples if "NT" not in s]
            
            # Get C>T rates for sorting (from non-consensus dataset which determined colors)
            ct_rates = {}
            for sample in all_samples:
                if sample in sample_to_rates2:
                    ct_rates[sample] = sample_to_rates2[sample].get('C>T', 0.0)
                else:
                    ct_rates[sample] = 0.0
            
            NT_samples_sorted = sorted(NT_samples, key=lambda x: ct_rates.get(x, 0))
            other_samples_sorted = sorted(other_samples, key=lambda x: ct_rates.get(x, 0))
            
            # Create clean labels
            sample_label_map = {}
            seen_labels = {}
            for s in all_samples:
                base_label = clean_sample_label(s)
                if base_label in seen_labels:
                    seen_labels[base_label] += 1
                    unique_label = f"{base_label}_{seen_labels[base_label]}"
                else:
                    seen_labels[base_label] = 0
                    unique_label = base_label
                sample_label_map[s] = unique_label
            
            # Create legend handles and labels
            import matplotlib.patches as mpatches
            legend_handles = []
            legend_labels = []
            
            # Add NT controls entry (single entry for all NT samples)
            if NT_samples_sorted:
                # Use the color from the first NT sample
                nt_sample = NT_samples_sorted[0]
                nt_color = color_dict.get(nt_sample, '#808080')
                print(f"  Legend (finite-sites): NT controls using color {nt_color} from sample {nt_sample}")
                legend_handles.append(mpatches.Patch(color=nt_color))
                legend_labels.append("NT controls")
            
            # Add individual treated sample entries
            for sample in other_samples_sorted:
                color = color_dict.get(sample, '#FF6B35')
                label = sample_label_map.get(sample, sample)
                print(f"  Legend (finite-sites): {label} using color {color} from sample {sample}")
                legend_handles.append(mpatches.Patch(color=color))
                legend_labels.append(label)
            
            # Create shared legend on the figure
            if legend_handles:
                fig.legend(legend_handles, legend_labels, fontsize=14, bbox_to_anchor=(1.02, 0.5), loc="center left")
            
            plt.tight_layout()
            plt.savefig(args.out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Comparison plot saved: {args.out}")
        else:
            plot_spectra_comparison(sample_to_rates1, sample_to_rates2, args.out, 
                                   filtering_stats1, filtering_stats2, args.depth_correction,
                                   title1, title2)
    else:
        title = args.title1 if args.title1 else None
        if args.rate_model == 'finite-sites':
            plot_spectra_finite_sites(sample_to_rates1, args.out, filtering_stats1, title=title)
        else:
            plot_spectra(sample_to_rates1, args.out, filtering_stats1, args.depth_correction, title=title)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

