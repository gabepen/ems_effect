#!/usr/bin/env python3
"""
Enhanced mutation rate estimator with statistical improvements.

This script implements multiple mutation rate estimation methods:
- Simple rates: Low estimate (mutated positions / total depth) and High estimate (total alt alleles / total depth)
- Alpha correction: Background false positive rate estimation from controls
- GLM analysis: Poisson/Negative Binomial regression for better uncertainty quantification
- Coverage-dependent analysis: Rate estimation across coverage bins

Processes only EMS mutations: C>T and G>A at G/C sites.
Includes sample name simplification and comprehensive ranking plots.
"""
import argparse
import glob
import csv
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import re
import scipy.stats as stats
from scipy.stats import binom
import statsmodels.api as sm



def proportion_ci(successes, total, alpha=0.05):
    """Return estimate, lower, upper using Wald normal approximation."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    se = math.sqrt(p_hat * (1 - p_hat) / total)
    z = 1.96
    lower = max(0.0, p_hat - z * se)
    upper = min(1.0, p_hat + z * se)
    return p_hat, lower, upper


def derive_sample_label(sample_name, treated_str='EMS', control_str='NT'):
    """Derive cleaner sample labels using the logic from ems_mutation_rate.py"""
    s_str = str(sample_name)
    
    if treated_str in s_str:
        # Capture group number with optional separators and optional leading zeros, anywhere in string
        m = re.search(f'(?i){treated_str}[\\s\\-_]*0*(\\d+)', s_str)
        base = f"{treated_str}{int(m.group(1))}" if m else None
    elif control_str in s_str:
        m = re.search(f'(?i){control_str}[\\s\\-_]*0*(\\d+)', s_str)
        base = f"{control_str}{int(m.group(1))}" if m else None
    if not base:
        return s_str
    # Capture 3d/7d anywhere (case-insensitive), with flexible separators
    d = re.search(r'(?i)(?:^|[^A-Za-z0-9])(3|7)[\s\-_]*d(?:[^A-Za-z0-9]|$)', s_str)
    if d:
        return f"{base}_{d.group(1).lower()}d"
    return base


def load_exclusion_mask(mask_file):
    """Load exclusion mask from file."""
    excluded_sites = set()
    with open(mask_file) as f:
        header = next(f, None)  # Skip header
        for line in f:
            chrom, pos = line.strip().split('\t')
            excluded_sites.add((chrom, pos))
    return excluded_sites


def estimate_alpha_from_controls(df):
    """Estimate background false positive rate from control samples using weighted average."""
    controls = df[df['is_control']]
    if len(controls) == 0:
        return 0.0
    
    # Weight by total depth (more reads = more reliable estimate)
    total_mutations = (controls['finite_sites_mutations']).sum()
    total_depth = controls['total_depth'].sum()
    alpha = total_mutations / total_depth if total_depth > 0 else 0.0
    return alpha


def estimate_alpha_from_control_sites(control_files, min_alt=1, excluded_sites=None):
    """Estimate alpha from raw control site data (most accurate approach)."""
    total_control_mutations = 0
    total_control_depth = 0
    
    for filepath in control_files:
        with open(filepath) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                
                # Convert to appropriate types
                ref_count = int(ref_count)
                A_count = int(A_count)
                C_count = int(C_count)
                G_count = int(G_count)
                T_count = int(T_count)
                depth = int(depth)
                
                # Only consider G/C sites (EMS targets)
                if ref not in {"G", "C"}:
                    continue
                
                # Apply exclusion mask if provided
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                    
                total_control_depth += depth
                
                # Calculate EMS mutations: C>T and G>A only
                if ref == "C":
                    ems_count = T_count  # C>T mutations
                elif ref == "G":
                    ems_count = A_count  # G>A mutations
                else:
                    ems_count = 0
                
                if ems_count >= min_alt:
                    total_control_mutations += ems_count
    
    alpha = total_control_mutations / total_control_depth if total_control_depth > 0 else 0.0
    return alpha


def parse_gff(gff_file):
    """Parse GFF file to extract gene regions."""
    gene_regions = []
    with open(gff_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.strip().split('\t')
            if len(fields) >= 9 and fields[2] == 'gene':
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                gene_id = fields[8].split(';')[0].replace('ID=', '') if 'ID=' in fields[8] else f"gene_{len(gene_regions)}"
                gene_regions.append((chrom, start, end, gene_id))
    return gene_regions


def create_genomic_bins(gene_regions, bin_size=2000):
    """Create 2kb bins for intergenic regions."""
    bins = []
    
    # Group genes by chromosome
    chrom_genes = {}
    for chrom, start, end, gene_id in gene_regions:
        if chrom not in chrom_genes:
            chrom_genes[chrom] = []
        chrom_genes[chrom].append((start, end, gene_id))
    
    # Create bins for each chromosome
    for chrom, genes in chrom_genes.items():
        genes.sort()  # Sort by start position
        
        # Add gene regions
        for start, end, gene_id in genes:
            bins.append((chrom, start, end, f"gene_{gene_id}"))
        
        # Add intergenic bins
        for i in range(len(genes) - 1):
            current_end = genes[i][1]
            next_start = genes[i + 1][0]
            
            # Create 2kb bins in the intergenic region
            pos = current_end
            while pos < next_start:
                bin_end = min(pos + bin_size, next_start)
                if bin_end - pos >= 100:  # Only create bins with at least 100bp
                    bins.append((chrom, pos, bin_end, f"intergenic_{chrom}_{pos}"))
                pos = bin_end
    
    return bins


def estimate_f_clonal_from_sites(counts_file, genomic_regions, min_alt=5, vaf_threshold=0.01, min_cov=10, excluded_sites=None):
    """Estimate f-clonal for each genomic region from high-confidence mutations."""
    region_f_clonal = {}
    
    # Collect VAFs for each region
    region_vafs = {region[3]: [] for region in genomic_regions}
    
    with open(counts_file) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
            
            # Convert to appropriate types
            ref_count = int(ref_count)
            A_count = int(A_count)
            C_count = int(C_count)
            G_count = int(G_count)
            T_count = int(T_count)
            depth = int(depth)
            pos = int(pos)
            
            # Only consider G/C sites (EMS targets)
            if ref not in {"G", "C"}:
                continue
            
            # Apply exclusion mask if provided
            if excluded_sites and (chrom, pos) in excluded_sites:
                continue
            
            # Check coverage
            if depth < min_cov:
                continue
            
            # Calculate EMS mutations: C>T and G>A only
            if ref == "C":
                ems_count = T_count  # C>T mutations
            elif ref == "G":
                ems_count = A_count  # G>A mutations
            else:
                ems_count = 0
            
            # Only consider high-confidence mutations
            if ems_count >= min_alt:
                vaf = ems_count / depth
                if vaf >= vaf_threshold:
                    # Find which region this position belongs to
                    for region_chrom, region_start, region_end, region_id in genomic_regions:
                        if region_chrom == chrom and region_start <= pos <= region_end:
                            region_vafs[region_id].append(vaf)
                            break
    
    # Estimate f-clonal for each region
    for region_id, vafs in region_vafs.items():
        if len(vafs) >= 3:  # Need at least 3 mutations for reliable estimate
            # Use median VAF as f-clonal estimate
            f_clonal = np.median(vafs)
            region_f_clonal[region_id] = f_clonal
        else:
            # Not enough data, use default
            region_f_clonal[region_id] = 0.5
    
    return region_f_clonal


def calculate_detection_probability(depth, f_clonal=0.5, vaf_threshold=0.01, min_alt=1):
    """Calculate per-site detection probability s(n) given coverage."""
    if depth == 0:
        return 0.0
    
    # Calculate minimum alt reads needed to call mutation
    k_vaf = int(np.ceil(vaf_threshold * depth))
    k_thresh = max(k_vaf, min_alt)
    
    # Probability that Bin(n, f_clonal) >= k_thresh
    return binom.sf(k_thresh - 1, depth, f_clonal)


def apply_alpha_correction(df, alpha):
    """Apply alpha correction to existing rate estimates."""
    df = df.copy()
    
    # Apply alpha correction: adjusted_rate = max(0, raw_rate - alpha)
    df['low_rate_alpha_corrected'] = np.maximum(0, df['low_rate'] - alpha)
    df['high_rate_alpha_corrected'] = np.maximum(0, df['high_rate'] - alpha)
    
    # Adjust confidence intervals (subtract alpha from bounds)
    df['low_CI_low_alpha_corrected'] = np.maximum(0, df['low_CI_low'] - alpha)
    df['low_CI_high_alpha_corrected'] = np.maximum(0, df['low_CI_high'] - alpha)
    df['high_CI_low_alpha_corrected'] = np.maximum(0, df['high_CI_low'] - alpha)
    df['high_CI_high_alpha_corrected'] = np.maximum(0, df['high_CI_high'] - alpha)
    
    # Store alpha used for reference
    df['alpha_used'] = alpha
    
    return df


def estimate_rates_glm_genome_wide(df, method='poisson', use_f_clonal=False, f_clonal_regions=None, no_f_clonal=False):
    """Fit GLM to estimate mutation rate with better uncertainty quantification."""
    df = df.copy()
    
    # Prepare data for GLM
    # Response: mutation counts, Exposure: total depth
    mutation_counts = df['finite_sites_mutations'].values
    total_depths = df['total_depth'].values
    
    # Remove samples with zero depth
    valid_mask = total_depths > 0
    if not valid_mask.any():
        df['glm_rate'] = np.nan
        df['glm_CI_low'] = np.nan
        df['glm_CI_high'] = np.nan
        return df
    
    y = mutation_counts[valid_mask]
    exposure = total_depths[valid_mask]
    
    # Calculate detection probability if using f-clonal (and not disabled)
    if use_f_clonal and f_clonal_regions and not no_f_clonal:
        # For genome-wide GLM, use average f-clonal across all regions
        avg_f_clonal = np.mean(list(f_clonal_regions.values()))
        detection_probs = np.array([calculate_detection_probability(d, avg_f_clonal) for d in exposure])
        # Adjust exposure by detection probability
        exposure = exposure * detection_probs
    
    # Alpha to subtract at counts level (same alpha used in simple model)
    alpha_for_glm = float(df['alpha_used'].iloc[0]) if 'alpha_used' in df.columns and len(df['alpha_used']) > 0 else 0.0

    # Fit individual GLM for each sample
    for i, (idx, row) in enumerate(df[valid_mask].iterrows()):
        sample_mutations = mutation_counts[valid_mask][i]
        sample_exposure = total_depths[valid_mask][i]
        
        if sample_exposure > 0:
            try:
                # Suppress warnings for GLM fitting
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Subtract background alpha at counts level
                    sample_mutations_adj = max(0, sample_mutations - alpha_for_glm * sample_exposure)
                    if method == 'poisson':
                        # Poisson GLM: log(E[counts]) = log(exposure) + log(rate)
                        model = sm.GLM([sample_mutations_adj], [1], family=sm.families.Poisson(), 
                                      offset=[np.log(sample_exposure)])
                        result = model.fit()
                        rate = np.exp(result.params[0])
                        ci = np.exp(result.conf_int()[0])
                        
                    elif method == 'negative_binomial':
                        # Negative binomial GLM for overdispersion
                        model = sm.GLM([sample_mutations_adj], [1], family=sm.families.NegativeBinomial(), 
                                      offset=[np.log(sample_exposure)])
                        result = model.fit()
                        rate = np.exp(result.params[0])
                        ci = np.exp(result.conf_int()[0])
                
                # Fill in results for this sample
                df.loc[idx, 'glm_rate'] = rate
                df.loc[idx, 'glm_CI_low'] = ci[0]
                df.loc[idx, 'glm_CI_high'] = ci[1]
                
            except Exception as e:
                print(f"Warning: GLM failed for sample {row['sample']}: {e}")
                df.loc[idx, 'glm_rate'] = np.nan
                df.loc[idx, 'glm_CI_low'] = np.nan
                df.loc[idx, 'glm_CI_high'] = np.nan
        
    # Fill invalid samples with NaN
    df.loc[~valid_mask, 'glm_rate'] = np.nan
    df.loc[~valid_mask, 'glm_CI_low'] = np.nan
    df.loc[~valid_mask, 'glm_CI_high'] = np.nan
    
    return df


def estimate_rates_glm_per_gene(counts_files, genomic_regions, f_clonal_regions, method='poisson', min_alt=1, excluded_sites=None, alpha_for_glm: float = 0.0, min_alt_c: int | None = None, min_alt_g: int | None = None):
    """Fit per-gene/region GLM analysis using region-specific f-clonal values."""
    results = []
    
    for filepath in counts_files:
        basename = os.path.basename(filepath)
        sample_name = basename.replace(".counts", "")
        
        # Collect data for each region
        region_data = {}
        for region_chrom, region_start, region_end, region_id in genomic_regions:
            region_data[region_id] = {
                'mutations': 0,
                'depth': 0,
                'f_clonal': f_clonal_regions.get(region_id, 0.5)
            }
        
        # Process sites and assign to regions
        with open(filepath) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                
                # Convert to appropriate types
                ref_count = int(ref_count)
                A_count = int(A_count)
                C_count = int(C_count)
                G_count = int(G_count)
                T_count = int(T_count)
                depth = int(depth)
                pos = int(pos)
                
                # Only consider G/C sites (EMS targets)
                if ref not in {"C", "G"}:
                    continue
                
                # Apply exclusion mask if provided
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                
                # Find which region this position belongs to
                for region_chrom, region_start, region_end, region_id in genomic_regions:
                    if region_chrom == chrom and region_start <= pos <= region_end:
                        # Calculate EMS mutations depending on reference base
                        if ref == "C":
                            ems_count = T_count
                            thr_c = min_alt_c if min_alt_c is not None else min_alt
                            if ems_count >= thr_c:
                                region_data[region_id]['mutations'] += ems_count
                        else:  # ref == "G"
                            ems_count = A_count
                            thr_g = min_alt_g if min_alt_g is not None else min_alt
                            if ems_count >= thr_g:
                                region_data[region_id]['mutations'] += ems_count
                        region_data[region_id]['depth'] += depth
                        break
        
        # Fit GLM for each region
        region_rates = {}
        # Track totals across INCLUDED regions (depth gate applied)
        included_total_mutations = 0
        included_total_depth = 0
        for region_id, data in region_data.items():
            depth = data['depth']
            if depth <= 0:
                continue
            mutations = data['mutations']
            f_clonal = data['f_clonal']

            # Keep a simple depth gate to ensure stability
            if depth < 10:
                continue

            # Use raw depth as exposure (no detection probability adjustment)
            effective_exposure = depth
            # Subtract background alpha at counts level BEFORE gating on counts
            mutations_adj = max(0, mutations - alpha_for_glm * effective_exposure)

            # Accumulate totals over included regions (post depth gate)
            included_total_mutations += mutations
            included_total_depth += depth

            # If adjusted count is zero or below, include region with rate=0 in aggregation
            if mutations_adj <= 0:
                region_rates[region_id] = {
                    'rate': 0.0,
                    'CI_low': 0.0,
                    'CI_high': 0.0,
                    'mutations': mutations,
                    'depth': depth,
                    'f_clonal': f_clonal
                }
                continue

            try:
                # Suppress warnings for GLM fitting
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if method == 'poisson':
                        model = sm.GLM([mutations_adj], [1], family=sm.families.Poisson(), 
                                      offset=[np.log(effective_exposure)])
                        result = model.fit(maxiter=50)  # Limit iterations
                        rate = np.exp(result.params[0])
                        ci = np.exp(result.conf_int()[0])
                    elif method == 'negative_binomial':
                        model = sm.GLM([mutations_adj], [1], family=sm.families.NegativeBinomial(), 
                                      offset=[np.log(effective_exposure)])
                        result = model.fit(maxiter=50)  # Limit iterations
                        rate = np.exp(result.params[0])
                        ci = np.exp(result.conf_int()[0])

                # Validate results - only keep finite, reasonable rates
                if np.isfinite(rate) and rate > 0 and rate < 1:
                    region_rates[region_id] = {
                        'rate': rate,
                        'CI_low': ci[0],
                        'CI_high': ci[1],
                        'mutations': mutations,
                        'depth': depth,
                        'f_clonal': f_clonal
                    }

            except Exception as e:
                # Only print warning for regions with substantial data
                if mutations >= 5 and depth >= 50:
                    print(f"Warning: GLM failed for {sample_name} region {region_id}: {e}")
                continue
        
        # Compute overall per-sample rate using GLOBAL alpha subtraction
        # across INCLUDED regions only (depth gate applied), then truncate at zero
        if included_total_depth > 0:
            adjusted_total = max(0.0, included_total_mutations - alpha_for_glm * included_total_depth)
            overall_rate = adjusted_total / included_total_depth
        else:
            overall_rate = np.nan
        
        results.append({
            'sample': sample_name,
            'per_gene_glm_rate': overall_rate,
            'per_gene_glm_mutations': included_total_mutations,
            'per_gene_glm_depth': included_total_depth,
            'num_regions': len(region_rates),
            'region_rates': region_rates
        })
    
    return results


def estimate_rates_glm_by_depth(counts_files, n_bins=30, method='poisson', min_alt=1, excluded_sites=None, alpha_for_glm: float = 0.0, min_alt_c: int | None = None, min_alt_g: int | None = None):
    """Fit GLM grouping sites by depth bins across all samples."""
    # Collect all depth data to create consistent bins
    all_depths = []
    for filepath in counts_files:
        with open(filepath) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                
                if ref != "C":
                    continue
                
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                
                all_depths.append(int(depth))
    
    # Create depth bins based on all data
    try:
        depth_bins = pd.qcut(all_depths, q=n_bins, labels=False, duplicates='drop')
        bin_edges = pd.qcut(all_depths, q=n_bins, duplicates='drop').categories
    except ValueError:
        # If qcut fails due to too many duplicates, use cut instead
        depth_bins = pd.cut(all_depths, bins=n_bins, labels=False, duplicates='drop')
        bin_edges = pd.cut(all_depths, bins=n_bins, duplicates='drop').categories
    
    # Convert bin_edges to a list of intervals for easier access
    if hasattr(bin_edges, 'left') and hasattr(bin_edges, 'right'):
        # Already interval index
        bin_intervals = bin_edges
    else:
        # Convert to interval index
        bin_intervals = bin_edges
    
    results = []
    
    for filepath in counts_files:
        basename = os.path.basename(filepath)
        sample_name = basename.replace(".counts", "")
        
        # Collect data for each depth bin
        bin_data = {i: {'mutations': 0, 'depth': 0, 'sites': 0} for i in range(n_bins)}
        
        with open(filepath) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                
                # Convert to appropriate types
                ref_count = int(ref_count)
                A_count = int(A_count)
                C_count = int(C_count)
                G_count = int(G_count)
                T_count = int(T_count)
                depth = int(depth)
                
                # Only consider G/C sites (EMS targets)
                if ref not in {"G", "C"}:
                    continue
                
                # Apply exclusion mask if provided
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                
                # Find depth bin for this site
                bin_idx = None
                for i, interval in enumerate(bin_intervals):
                    if interval.left <= depth <= interval.right:
                        bin_idx = i
                        break
                
                if bin_idx is not None:
                    bin_data[bin_idx]['sites'] += 1
                    bin_data[bin_idx]['depth'] += depth
                    
                    # Calculate EMS mutations: C>T and G>A only
                    if ref == "C":
                        ems_count = T_count  # C>T mutations
                        thr_c = min_alt_c if min_alt_c is not None else min_alt
                        if ems_count >= thr_c:
                            bin_data[bin_idx]['mutations'] += ems_count
                    elif ref == "G":
                        ems_count = A_count  # G>A mutations
                        thr_g = min_alt_g if min_alt_g is not None else min_alt
                        if ems_count >= thr_g:
                            bin_data[bin_idx]['mutations'] += ems_count
        
        # Fit GLM for each depth bin
        bin_rates = {}
        total_mutations = 0
        total_depth = 0
        
        for bin_idx, data in bin_data.items():
            if data['depth'] > 0 and data['sites'] >= 20:  # Require minimum sites per bin
                mutations = data['mutations']
                depth = data['depth']
                
                # Skip bins with very low mutation counts (likely to cause convergence issues)
                if mutations < 1:
                    continue
                    
                try:
                    # Suppress warnings for GLM fitting
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        # Subtract background alpha at counts level
                        mutations_adj = max(0, mutations - alpha_for_glm * depth)
                        if method == 'poisson':
                            model = sm.GLM([mutations_adj], [1], family=sm.families.Poisson(), 
                                          offset=[np.log(depth)])
                            result = model.fit(maxiter=50)  # Limit iterations
                            rate = np.exp(result.params[0])
                            ci = np.exp(result.conf_int()[0])
                        elif method == 'negative_binomial':
                            model = sm.GLM([mutations_adj], [1], family=sm.families.NegativeBinomial(), 
                                          offset=[np.log(depth)])
                            result = model.fit(maxiter=50)  # Limit iterations
                            rate = np.exp(result.params[0])
                            ci = np.exp(result.conf_int()[0])
                    
                    # Check for reasonable results
                    if np.isfinite(rate) and rate > 0 and rate < 1:
                        bin_rates[bin_idx] = {
                            'rate': rate,
                            'CI_low': ci[0],
                            'CI_high': ci[1],
                            'mutations': mutations,
                            'depth': depth,
                            'sites': data['sites']
                        }
                        
                        total_mutations += mutations
                        total_depth += depth
                    
                except Exception as e:
                    print(f"Warning: GLM failed for {sample_name} depth bin {bin_idx}: {e}")
                    continue
        
        # Calculate weighted average rate across depth bins
        if total_depth > 0:
            weighted_rate = 0
            total_weight = 0
            for bin_idx, rate_data in bin_rates.items():
                if not np.isnan(rate_data['rate']):
                    weight = rate_data['depth']
                    weighted_rate += rate_data['rate'] * weight
                    total_weight += weight
            
            overall_rate = weighted_rate / total_weight if total_weight > 0 else np.nan
        else:
            overall_rate = np.nan
        
        results.append({
            'sample': sample_name,
            'depth_glm_rate': overall_rate,
            'depth_glm_mutations': total_mutations,
            'depth_glm_depth': total_depth,
            'num_depth_bins': len(bin_rates),
            'depth_bin_rates': bin_rates
        })
    
    return results


def estimate_rates_from_counts(filename, min_alt=1, excluded_sites=None):
    """
    Estimate mutation rates from a .counts file (output from collect_mutation_counts.py).
    
    Returns:
        dict with rate estimates and metadata
    """
    total_depth = 0
    infinite_sites_mutations = 0  # count of positions with mutations
    finite_sites_mutations = 0    # total count of alt alleles
    
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
            
            # Convert to appropriate types
            ref_count = int(ref_count)
            A_count = int(A_count)
            C_count = int(C_count)
            G_count = int(G_count)
            T_count = int(T_count)
            depth = int(depth)
            
            # Only consider G/C sites (EMS targets)
            if ref not in {"G", "C"}:
                continue
            
            # Apply exclusion mask if provided
            if excluded_sites and (chrom, pos) in excluded_sites:
                continue
                
            total_depth += depth
            
            # Calculate EMS mutations: C>T and G>A only
            if ref == "C":
                ems_count = T_count  # C>T mutations
            elif ref == "G":
                ems_count = A_count  # G>A mutations
            else:
                ems_count = 0
            
            if ems_count >= min_alt:
                infinite_sites_mutations += 1
                finite_sites_mutations += ems_count

    # Calculate rate estimates
    inf_est, inf_lo, inf_hi = proportion_ci(infinite_sites_mutations, total_depth)
    fin_est, fin_lo, fin_hi = proportion_ci(finite_sites_mutations, total_depth)

    basename = os.path.basename(filename)
    sample_name = basename.replace(".counts", "")
    label = derive_sample_label(sample_name, treated_str='EMS', control_str='NT')

    return {
        "sample": sample_name,
        "label": label,
        "low_rate": inf_est,
        "low_CI_low": inf_lo,
        "low_CI_high": inf_hi,
        "high_rate": fin_est,
        "high_CI_low": fin_lo,
        "high_CI_high": fin_hi,
        "total_depth": total_depth,
        "infinite_sites_mutations": infinite_sites_mutations,
        "finite_sites_mutations": finite_sites_mutations,
        "is_control": "NT" in basename,
    }


def create_ranking_plots(df, output_prefix):
    """Create clear ranking plots for mutation rates."""
    
    # Sort by high rate for ranking
    df_sorted = df.sort_values('high_rate', ascending=False)
    
    # Create figure with subplots - much taller figure to give more room for y-axis labels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 20))
    
    # Adjust subplot spacing to give more room for y-axis labels
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08, hspace=0.2, wspace=0.25)
    
    # 1) High rate ranking (horizontal bar plot)
    colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
    bars1 = ax1.barh(range(len(df_sorted)), df_sorted['high_rate'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['label'], fontsize=12)
    ax1.set_xlabel('Per-base mutation probability (high estimate)', fontsize=14)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax1.xaxis.set_major_formatter(fmt)
    ax1.set_title('Mutation Rate Ranking - High Estimate', fontsize=16)
    ax1.invert_yaxis()
    
    # Add error bars for high rate
    y_pos = range(len(df_sorted))
    ax1.errorbar(df_sorted['high_rate'], y_pos, 
                xerr=[df_sorted['high_rate'] - df_sorted['high_CI_low'], 
                      df_sorted['high_CI_high'] - df_sorted['high_rate']],
                fmt='none', color='black', capsize=2, alpha=0.6)
    
    # 2) Low rate ranking (horizontal bar plot)
    bars2 = ax2.barh(range(len(df_sorted)), df_sorted['low_rate'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted['label'], fontsize=12)
    ax2.set_xlabel('Per-base mutation probability (low estimate)', fontsize=14)
    ax2.xaxis.set_major_formatter(fmt)
    ax2.set_title('Mutation Rate Ranking - Low Estimate', fontsize=16)
    ax2.invert_yaxis()
    
    # Add error bars for low rate
    ax2.errorbar(df_sorted['low_rate'], y_pos,
                xerr=[df_sorted['low_rate'] - df_sorted['low_CI_low'],
                      df_sorted['low_CI_high'] - df_sorted['low_rate']],
                fmt='none', color='black', capsize=2, alpha=0.6)
    
    # 3) Scatter plot: Low vs High rates
    control_mask = df['is_control']
    sample_mask = ~control_mask
    
    if control_mask.any():
        ax3.scatter(df.loc[control_mask, 'low_rate'], df.loc[control_mask, 'high_rate'], 
                   color='#1F77B4', s=100, alpha=0.7, label='Controls')
    
    if sample_mask.any():
        ax3.scatter(df.loc[sample_mask, 'low_rate'], df.loc[sample_mask, 'high_rate'], 
                   color='#FF7F0E', s=100, alpha=0.7, label='Samples')
    
    # Add error bars to scatter plot
    for _, row in df.iterrows():
        color = '#1F77B4' if row['is_control'] else '#FF7F0E'
        ax3.errorbar(row['low_rate'], row['high_rate'],
                    xerr=[[row['low_rate'] - row['low_CI_low']], [row['low_CI_high'] - row['low_rate']]],
                    yerr=[[row['high_rate'] - row['high_CI_low']], [row['high_CI_high'] - row['high_rate']]],
                    fmt='none', color=color, alpha=0.5, capsize=2)
    
    ax3.set_xlabel('Per-base mutation probability (low)', fontsize=14)
    ax3.set_ylabel('Per-base mutation probability (high)', fontsize=14)
    ax3.xaxis.set_major_formatter(fmt)
    ax3.yaxis.set_major_formatter(fmt)
    ax3.set_title('Low vs High Rate Estimates', fontsize=16)
    ax3.legend()
    
    
    # 4) Alpha-corrected rates (if available, otherwise rate difference)
    if 'high_rate_alpha_corrected' in df_sorted.columns:
        # Use alpha-corrected high rate
        corrected_rates = df_sorted['high_rate_alpha_corrected']
        colors_corrected = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars4 = ax4.barh(range(len(df_sorted)), corrected_rates, color=colors_corrected, alpha=0.7)
        ax4.set_xlabel('Per-base mutation probability (alpha-corrected)')
        ax4.xaxis.set_major_formatter(fmt)
        ax4.set_title('Alpha-Corrected Mutation Rates')
        
        # Add error bars for alpha-corrected rates if available
        if 'high_CI_low_alpha_corrected' in df_sorted.columns:
            ax4.errorbar(corrected_rates, range(len(df_sorted)),
                        xerr=[corrected_rates - df_sorted['high_CI_low_alpha_corrected'],
                              df_sorted['high_CI_high_alpha_corrected'] - corrected_rates],
                        fmt='none', color='black', capsize=2, alpha=0.6)
    else:
        # Fallback to rate difference if alpha correction not available
        df_sorted['rate_diff'] = df_sorted['high_rate'] - df_sorted['low_rate']
        colors_diff = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars4 = ax4.barh(range(len(df_sorted)), df_sorted['rate_diff'], color=colors_diff, alpha=0.7)
        ax4.set_xlabel('Rate Difference (High - Low)')
        ax4.xaxis.set_major_formatter(fmt)
        ax4.set_title('Difference Between High and Low Estimates')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax4.set_yticks(range(len(df_sorted)))
    ax4.set_yticklabels(df_sorted['label'], fontsize=12)
    ax4.invert_yaxis()
    
    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F0E', 
                             markersize=10, label='Samples'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4', 
                             markersize=10, label='Controls')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Save plot
    plot_path = f"{output_prefix}_mutation_rate_ranking.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple rate ranking plots saved to {plot_path}")
    
    return df_sorted


def save_individual_simple_plots(df, output_prefix):
    """Save individual plots for simple rate analysis."""
    
    # 1) Low rate ranking
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    df_sorted = df.sort_values('low_rate', ascending=False)
    colors = ['#E74C3C' if not is_ctrl else '#3498DB' for is_ctrl in df_sorted['is_control']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['low_rate'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['label'], fontsize=10)
    ax.set_xlabel('Low Estimate Rate')
    ax.set_title('Low Rate Ranking')
    ax.invert_yaxis()
    
    # Add error bars if available
    if 'low_CI_low' in df_sorted.columns:
        y_pos = range(len(df_sorted))
        ax.errorbar(df_sorted['low_rate'], y_pos, 
                   xerr=[df_sorted['low_rate'] - df_sorted['low_CI_low'], 
                         df_sorted['low_CI_high'] - df_sorted['low_rate']],
                   fmt='none', color='black', capsize=2, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_low_rate_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Low rate ranking saved to {output_prefix}_low_rate_ranking.png")
    
    # 2) High rate ranking
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    df_sorted = df.sort_values('high_rate', ascending=False)
    colors = ['#E74C3C' if not is_ctrl else '#3498DB' for is_ctrl in df_sorted['is_control']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['high_rate'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['label'], fontsize=10)
    ax.set_xlabel('High Estimate Rate')
    ax.set_title('High Rate Ranking')
    ax.invert_yaxis()
    
    # Add error bars if available
    if 'high_CI_low' in df_sorted.columns:
        y_pos = range(len(df_sorted))
        ax.errorbar(df_sorted['high_rate'], y_pos, 
                   xerr=[df_sorted['high_rate'] - df_sorted['high_CI_low'], 
                         df_sorted['high_CI_high'] - df_sorted['high_rate']],
                   fmt='none', color='black', capsize=2, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_high_rate_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"High rate ranking saved to {output_prefix}_high_rate_ranking.png")
    
    # 3) Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    control_mask = df['is_control']
    sample_mask = ~control_mask
    
    if control_mask.any():
        ax.scatter(df.loc[control_mask, 'low_rate'], df.loc[control_mask, 'high_rate'], 
                   color='#3498DB', s=100, alpha=0.7, label='Controls')
    
    if sample_mask.any():
        ax.scatter(df.loc[sample_mask, 'low_rate'], df.loc[sample_mask, 'high_rate'], 
                   color='#E74C3C', s=100, alpha=0.7, label='Samples')
    
    # Add error bars to scatter plot
    for _, row in df.iterrows():
        color = '#3498DB' if row['is_control'] else '#E74C3C'
        ax.errorbar(row['low_rate'], row['high_rate'],
                    xerr=[[row['low_rate'] - row['low_CI_low']], [row['low_CI_high'] - row['low_rate']]],
                    yerr=[[row['high_rate'] - row['high_CI_low']], [row['high_CI_high'] - row['high_rate']]],
                    fmt='none', color=color, alpha=0.5, capsize=2)
    
    ax.set_xlabel('Low Estimate Rate')
    ax.set_ylabel('High Estimate Rate')
    ax.set_title('Low vs High Rate Estimates')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rate comparison scatter plot saved to {output_prefix}_rate_comparison.png")
    
    # 4) Alpha-corrected rate (if available)
    if 'high_rate_alpha_corrected' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        df_sorted = df.sort_values('high_rate_alpha_corrected', ascending=False)
        colors = ['#E74C3C' if not is_ctrl else '#3498DB' for is_ctrl in df_sorted['is_control']]
        bars = ax.barh(range(len(df_sorted)), df_sorted['high_rate_alpha_corrected'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['label'], fontsize=10)
        ax.set_xlabel('Alpha-Corrected High Rate')
        ax.set_title('Alpha-Corrected Rate Ranking')
        ax.invert_yaxis()
        
        # Add error bars if available
        if 'high_rate_alpha_corrected_CI_low' in df_sorted.columns:
            y_pos = range(len(df_sorted))
            ax.errorbar(df_sorted['high_rate_alpha_corrected'], y_pos, 
                       xerr=[df_sorted['high_rate_alpha_corrected'] - df_sorted['high_rate_alpha_corrected_CI_low'], 
                             df_sorted['high_rate_alpha_corrected_CI_high'] - df_sorted['high_rate_alpha_corrected']],
                       fmt='none', color='black', capsize=2, alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_alpha_corrected_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Alpha-corrected rate ranking saved to {output_prefix}_alpha_corrected_ranking.png")


def create_glm_plots(df, output_prefix):
    """Create GLM ranking plots similar to simple rate plots."""
    
    # Create figure with subplots - much taller figure to give more room for y-axis labels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 20))
    # Swap third and fourth axes per request
    ax3, ax4 = ax4, ax3
    
    # Adjust subplot spacing to give more room for y-axis labels
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08, hspace=0.2, wspace=0.25)
    
    # Determine which GLM methods are available
    has_per_sample = 'glm_rate' in df.columns and df['glm_rate'].notna().any()
    has_depth = 'depth_glm_rate' in df.columns and df['depth_glm_rate'].notna().any()
    has_per_gene = 'per_gene_glm_rate' in df.columns and df['per_gene_glm_rate'].notna().any()
    
    # 1) Per-sample GLM rate ranking
    if has_per_sample:
        df_sorted = df.sort_values('glm_rate', ascending=False)
        colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars1 = ax1.barh(range(len(df_sorted)), df_sorted['glm_rate'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(df_sorted)))
        ax1.set_yticklabels(df_sorted['label'], fontsize=12)
        ax1.set_xlabel('Per-base mutation probability (per-sample GLM)', fontsize=14)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 2))
        ax1.xaxis.set_major_formatter(fmt)
        ax1.set_title('GLM Rate Ranking - Per-Sample', fontsize=16)
        ax1.invert_yaxis()
        
        # Add error bars if available
        if 'glm_CI_low' in df_sorted.columns:
            y_pos = range(len(df_sorted))
            ax1.errorbar(df_sorted['glm_rate'], y_pos, 
                        xerr=[df_sorted['glm_rate'] - df_sorted['glm_CI_low'], 
                              df_sorted['glm_CI_high'] - df_sorted['glm_rate']],
                        fmt='none', color='black', capsize=2, alpha=0.6)
    else:
        ax1.text(0.5, 0.5, 'Per-Sample GLM\nNot Available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('GLM Rate Ranking - Per-Sample', fontsize=16)
    
    # 2) Depth-based GLM rate ranking
    if has_depth:
        df_sorted = df.sort_values('depth_glm_rate', ascending=False)
        colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars2 = ax2.barh(range(len(df_sorted)), df_sorted['depth_glm_rate'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(df_sorted)))
        ax2.set_yticklabels(df_sorted['label'], fontsize=12)
        ax2.set_xlabel('Per-base mutation probability (depth-based GLM)', fontsize=14)
        ax2.xaxis.set_major_formatter(fmt)
        ax2.set_title('GLM Rate Ranking - Depth-Based', fontsize=16)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'Depth-Based GLM\nNot Available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('GLM Rate Ranking - Depth-Based', fontsize=16)
    
    # 3) Per-gene GLM rate ranking
    if has_per_gene:
        df_sorted = df.sort_values('per_gene_glm_rate', ascending=False)
        colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars3 = ax3.barh(range(len(df_sorted)), df_sorted['per_gene_glm_rate'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(df_sorted)))
        ax3.set_yticklabels(df_sorted['label'], fontsize=12)
        ax3.set_xlabel('Per-base mutation probability (per-gene GLM)', fontsize=14)
        ax3.xaxis.set_major_formatter(fmt)
        ax3.set_title('GLM Rate Ranking - Per-Gene', fontsize=16)
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, 'Per-Gene GLM\nNot Available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('GLM Rate Ranking - Per-Gene', fontsize=16)
    
    # 4) Comparison scatter plot
    if has_per_sample and has_depth:
        control_mask = df['is_control']
        sample_mask = ~control_mask
        
        if control_mask.any():
            ax4.scatter(df.loc[control_mask, 'glm_rate'], df.loc[control_mask, 'depth_glm_rate'], 
                       color='#1F77B4', s=100, alpha=0.7, label='Controls')
        
        if sample_mask.any():
            ax4.scatter(df.loc[sample_mask, 'glm_rate'], df.loc[sample_mask, 'depth_glm_rate'], 
                       color='#FF7F0E', s=100, alpha=0.7, label='Samples')
        
        ax4.set_xlabel('Per-base probability (per-sample GLM)', fontsize=14)
        ax4.set_ylabel('Per-base probability (depth-based GLM)', fontsize=14)
        ax4.xaxis.set_major_formatter(fmt)
        ax4.yaxis.set_major_formatter(fmt)
        ax4.set_title('Per-Sample vs Depth-Based GLM', fontsize=16)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'GLM Comparison\nNot Available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('GLM Method Comparison')
    
    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F0E', 
                             markersize=10, label='Samples'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4', 
                             markersize=10, label='Controls')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Save plot
    plot_path = f"{output_prefix}_glm_ranking.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"GLM analysis plots saved to {plot_path}")


def save_individual_glm_plots(df, output_prefix):
    """Save individual plots for GLM analysis."""
    
    # 1) Per-sample GLM ranking
    if 'glm_rate' in df.columns and df['glm_rate'].notna().any():
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        df_sorted = df.sort_values('glm_rate', ascending=False)
        colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars = ax.barh(range(len(df_sorted)), df_sorted['glm_rate'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['label'], fontsize=10)
        ax.set_xlabel('Per-base mutation probability (per-sample GLM)')
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(fmt)
        ax.set_title('Per-Sample GLM Rate Ranking')
        ax.invert_yaxis()
        
        # Add error bars if available
        if 'glm_CI_low' in df_sorted.columns:
            y_pos = range(len(df_sorted))
            ax.errorbar(df_sorted['glm_rate'], y_pos, 
                       xerr=[df_sorted['glm_rate'] - df_sorted['glm_CI_low'], 
                             df_sorted['glm_CI_high'] - df_sorted['glm_rate']],
                       fmt='none', color='black', capsize=2, alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_per_sample_glm_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-sample GLM ranking saved to {output_prefix}_per_sample_glm_ranking.png")
    
    # 2) Per-gene GLM ranking
    if 'per_gene_glm_rate' in df.columns and df['per_gene_glm_rate'].notna().any():
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        df_sorted = df.sort_values('per_gene_glm_rate', ascending=False)
        colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars = ax.barh(range(len(df_sorted)), df_sorted['per_gene_glm_rate'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['label'], fontsize=10)
        ax.set_xlabel('Per-base mutation probability (per-gene GLM)')
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(fmt)
        ax.set_title('Per-Gene GLM Rate Ranking')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_per_gene_glm_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-gene GLM ranking saved to {output_prefix}_per_gene_glm_ranking.png")
    
    # 3) Comparison scatter plot
    if 'glm_rate' in df.columns and 'per_gene_glm_rate' in df.columns:
        if df['glm_rate'].notna().any() and df['per_gene_glm_rate'].notna().any():
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            control_mask = df['is_control']
            sample_mask = ~control_mask
            
            if control_mask.any():
                ax.scatter(df.loc[control_mask, 'glm_rate'], df.loc[control_mask, 'per_gene_glm_rate'], 
                           color='#1F77B4', s=100, alpha=0.7, label='Controls')
            
            if sample_mask.any():
                ax.scatter(df.loc[sample_mask, 'glm_rate'], df.loc[sample_mask, 'per_gene_glm_rate'], 
                           color='#FF7F0E', s=100, alpha=0.7, label='Samples')
            
            ax.set_xlabel('Per-base mutation probability (per-sample GLM)')
            ax.set_ylabel('Per-base mutation probability (per-gene GLM)')
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((-2, 2))
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            ax.set_title('Per-Sample vs Per-Gene GLM Comparison')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_glm_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"GLM comparison scatter plot saved to {output_prefix}_glm_comparison.png")


def create_depth_glm_plots(df, output_prefix):
    """Create depth-based GLM plots showing depth bin analysis."""
    
    if 'depth_glm_rate' not in df.columns or df['depth_glm_rate'].notna().sum() == 0:
        print("No depth-based GLM data available for plotting")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1) Depth-based GLM rate ranking
    df_sorted = df.sort_values('depth_glm_rate', ascending=False)
    colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
    bars1 = ax1.barh(range(len(df_sorted)), df_sorted['depth_glm_rate'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['label'], fontsize=12)
    ax1.set_xlabel('Per-base mutation probability (depth-based GLM)')
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax1.xaxis.set_major_formatter(fmt)
    ax1.set_title('GLM Rate Ranking - Depth-Based')
    ax1.invert_yaxis()
    
    # 2) Comparison scatter plot (depth vs per-sample GLM) - bottom right position
    if 'glm_rate' in df.columns and df['glm_rate'].notna().any():
        control_mask = df['is_control']
        sample_mask = ~control_mask
        
        if control_mask.any():
            ax2.scatter(df.loc[control_mask, 'glm_rate'], df.loc[control_mask, 'depth_glm_rate'], 
                       color='#1F77B4', s=100, alpha=0.7, label='Controls')
        
        if sample_mask.any():
            ax2.scatter(df.loc[sample_mask, 'glm_rate'], df.loc[sample_mask, 'depth_glm_rate'], 
                       color='#FF7F0E', s=100, alpha=0.7, label='Samples')
        
        ax2.set_xlabel('Per-base mutation probability (per-sample GLM)')
        ax2.set_ylabel('Per-base mutation probability (depth-based GLM)')
        ax2.xaxis.set_major_formatter(fmt)
        ax2.yaxis.set_major_formatter(fmt)
        ax2.set_title('Per-Sample vs Depth-Based GLM')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Per-Sample GLM\nNot Available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Depth-Based GLM Analysis')
    
    plt.tight_layout()
    plot_path = f"{output_prefix}_depth_glm_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Depth GLM analysis plots saved to {plot_path}")


def save_individual_depth_glm_plots(df, output_prefix):
    """Save individual plots for depth-based GLM analysis."""
    
    # 1) Depth-based GLM ranking
    if 'depth_glm_rate' in df.columns and df['depth_glm_rate'].notna().any():
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        df_sorted = df.sort_values('depth_glm_rate', ascending=False)
        colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
        bars = ax.barh(range(len(df_sorted)), df_sorted['depth_glm_rate'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['label'], fontsize=10)
        ax.set_xlabel('Per-base mutation probability (depth-based GLM)')
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(fmt)
        ax.set_title('Depth-Based GLM Rate Ranking')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_depth_glm_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Depth-based GLM ranking saved to {output_prefix}_depth_glm_ranking.png")
    
    # 2) Comparison scatter plot
    if 'glm_rate' in df.columns and 'depth_glm_rate' in df.columns:
        if df['glm_rate'].notna().any() and df['depth_glm_rate'].notna().any():
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            control_mask = df['is_control']
            sample_mask = ~control_mask
            
            if control_mask.any():
                ax.scatter(df.loc[control_mask, 'glm_rate'], df.loc[control_mask, 'depth_glm_rate'], 
                           color='#1F77B4', s=100, alpha=0.7, label='Controls')
            
            if sample_mask.any():
                ax.scatter(df.loc[sample_mask, 'glm_rate'], df.loc[sample_mask, 'depth_glm_rate'], 
                           color='#FF7F0E', s=100, alpha=0.7, label='Samples')
            
            ax.set_xlabel('Per-base mutation probability (per-sample GLM)')
            ax.set_ylabel('Per-base mutation probability (depth-based GLM)')
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((-2, 2))
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            ax.set_title('Per-Sample vs Depth-Based GLM Comparison')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_depth_glm_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Depth GLM comparison scatter plot saved to {output_prefix}_depth_glm_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate mutation rates with enhanced statistical methods"
    )
    parser.add_argument("--counts-dir", type=str, required=True,
                        help="Directory containing .counts files")
    parser.add_argument("--output-prefix", type=str, default="simple_mutation_rates",
                        help="Prefix for output files (default: simple_mutation_rates)")
    parser.add_argument("--min-alt", type=int, default=1,
                        help="Minimum alt allele count to include site in numerator (default 1)")
    parser.add_argument("--exclude-controls", action="store_true",
                        help="Exclude control samples from analysis")
    parser.add_argument("--exclusion-mask", type=str, default=None,
                        help="Path to exclusion mask file (optional)")
    
    # Statistical enhancement arguments (all enabled by default)
    parser.add_argument("--no-alpha-correction", action="store_true",
                        help="Skip alpha correction even if controls present")
    parser.add_argument("--glm", action="store_true",
                        help="Run GLM analysis: per-sample, depth-binned, and per-gene methods")
    parser.add_argument("--glm-method", type=str, default="poisson", 
                        choices=["poisson", "negative_binomial"],
                        help="GLM method: poisson or negative_binomial (default: poisson)")
    parser.add_argument("--gff-file", type=str, default=None,
                        help="GFF file for gene annotations (required for per-gene GLM)")
    parser.add_argument("--min-alt-c", type=int, default=None,
                        help="Minimum alt reads at C sites (C>T) to count a site; overrides --min-alt for C")
    parser.add_argument("--min-alt-g", type=int, default=None,
                        help="Minimum alt reads at G sites (G>A) to count a site; overrides --min-alt for G (unused in C-only mode)")
    parser.add_argument("--fclonal-min-alt", type=int, default=4,
                        help="Minimum alt-read count for f-clonal estimation (default 4)")
    parser.add_argument("--depth-bins", type=int, default=10,
                        help="Number of depth bins for depth-based GLM (default 10)")
    parser.add_argument("--no-depth-glm", action="store_true",
                        help="Disable depth-based GLM analysis")
    
    args = parser.parse_args()

    # Find all .counts files
    counts_files = sorted(glob.glob(os.path.join(args.counts_dir, "*.counts")))
    if not counts_files:
        print(f"No .counts files found in {args.counts_dir}")
        return

    print(f"Found {len(counts_files)} .counts files")

    # Load exclusion mask if provided
    excluded_sites = None
    if args.exclusion_mask:
        if os.path.exists(args.exclusion_mask):
            excluded_sites = load_exclusion_mask(args.exclusion_mask)
            print(f"Loaded exclusion mask: {len(excluded_sites)} sites to exclude")
        else:
            print(f"Warning: Exclusion mask file not found: {args.exclusion_mask}")

    # Step 1: Basic rate estimation (existing)
    results = []
    for filepath in counts_files:
        result = estimate_rates_from_counts(filepath, min_alt=args.min_alt, excluded_sites=excluded_sites)
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Optionally exclude controls
    if args.exclude_controls:
        df = df[~df['is_control']].reset_index(drop=True)
        print(f"Excluded controls, analyzing {len(df)} samples")

    # Step 2: Estimate alpha from controls (always run if controls present)
    alpha = 0.0
    if not args.no_alpha_correction and df['is_control'].any():
        # Use raw site-level data from control files (most accurate)
        control_files = [f for f in counts_files if "NT" in os.path.basename(f)]
        if control_files:
            alpha = estimate_alpha_from_control_sites(control_files, min_alt=args.min_alt, excluded_sites=excluded_sites)
            print(f"Estimated alpha from pooled control sites (background rate): {alpha:.2e}")
        else:
            print("No control files found for pooled alpha estimation, falling back to weighted method")
            alpha = estimate_alpha_from_controls(df)
            print(f"Estimated alpha from weighted control rates (background rate): {alpha:.2e}")
    elif args.no_alpha_correction:
        print("Skipping alpha correction (--no-alpha-correction specified)")
    else:
        print("No control samples found, skipping alpha correction")

    # Step 3: Apply alpha correction
    if alpha > 0:
        df = apply_alpha_correction(df, alpha)
        print("Applied alpha correction to rate estimates")

    # Step 4: GLM analysis (if requested)
    if args.glm:
        print(f"Running GLM analysis (method={args.glm_method})")
        
        # 1) Per-sample GLM (genome-wide, no f-clonal)
        print("Fitting per-sample GLM...")
        df = estimate_rates_glm_genome_wide(df, method=args.glm_method, use_f_clonal=False, f_clonal_regions=None, no_f_clonal=True)
        
        # 2) Depth-based GLM (30 bins, no f-clonal)
        print(f"Fitting depth-based GLM ({args.depth_bins} bins)...")
        depth_results = estimate_rates_glm_by_depth(
            counts_files, n_bins=args.depth_bins, method=args.glm_method, 
            min_alt=args.min_alt, excluded_sites=excluded_sites, alpha_for_glm=alpha,
            min_alt_c=args.min_alt_c, min_alt_g=args.min_alt_g
        )
        depth_df = pd.DataFrame(depth_results)
        df = df.merge(depth_df[['sample', 'depth_glm_rate', 'depth_glm_mutations', 'depth_glm_depth', 'num_depth_bins']], 
                     on='sample', how='left')
        
        # 3) Per-gene GLM (with f-clonal, requires GFF)
        if args.gff_file and os.path.exists(args.gff_file):
            print("Parsing GFF file and creating genomic regions...")
            gene_regions = parse_gff(args.gff_file)
            genomic_regions = create_genomic_bins(gene_regions, bin_size=2000)
            print(f"Created {len(genomic_regions)} genomic regions ({len(gene_regions)} genes + {len(genomic_regions) - len(gene_regions)} intergenic bins)")
            
            # Estimate f-clonal for first sample
            print("Estimating f-clonal values from high-confidence mutations...")
            first_file = counts_files[0]
            f_clonal_regions = estimate_f_clonal_from_sites(
                first_file, genomic_regions, 
                min_alt=args.fclonal_min_alt, 
                vaf_threshold=0.001,  # Very low threshold to avoid depth bias
                excluded_sites=excluded_sites
            )
            avg_f_clonal = np.mean(list(f_clonal_regions.values()))
            print(f"Average f-clonal = {avg_f_clonal:.3f}")
            
            print("Fitting per-gene GLM...")
            per_gene_results = estimate_rates_glm_per_gene(
                counts_files, genomic_regions, f_clonal_regions, 
                method=args.glm_method, min_alt=args.min_alt, excluded_sites=excluded_sites, alpha_for_glm=alpha,
                min_alt_c=args.min_alt_c, min_alt_g=args.min_alt_g
            )
            
            # Add per-gene GLM results to dataframe
            per_gene_df = pd.DataFrame(per_gene_results)
            df = df.merge(per_gene_df[['sample', 'per_gene_glm_rate', 'per_gene_glm_mutations', 'per_gene_glm_depth', 'num_regions']], 
                         on='sample', how='left')
        else:
            print("Skipping per-gene GLM (no GFF file provided)")
    else:
        print("Skipping GLM analysis (--glm not specified)")

    # Save results table
    # Step 4: Create output directory
    output_dir = args.output_prefix
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Step 5: Save results to file
    out_table = os.path.join(output_dir, "simple_rates.tsv")
    df.to_csv(out_table, sep="\t", index=False)
    print(f"Simple rates saved to {out_table}")

    # Step 6: Create ranking plots with enhanced data
    df_ranked = create_ranking_plots(df, os.path.join(output_dir, "simple_rates"))
    
    # Save individual simple rate plots immediately
    save_individual_simple_plots(df_ranked, os.path.join(output_dir, "simple_rates"))
    
    # Create GLM plots if GLM analysis was run
    if args.glm:
        # Save GLM results table
        glm_table = os.path.join(output_dir, "glm_rates.tsv")
        df.to_csv(glm_table, sep="\t", index=False)
        print(f"GLM rates saved to {glm_table}")
        
        # Create GLM plots
        create_glm_plots(df, os.path.join(output_dir, "glm_analysis"))
        
        # Save individual GLM plots immediately
        save_individual_glm_plots(df, os.path.join(output_dir, "glm_analysis"))
        
        # Create separate depth GLM plots if depth analysis was run
        if not args.no_depth_glm and 'depth_glm_rate' in df.columns:
            create_depth_glm_plots(df, os.path.join(output_dir, "depth_glm_analysis"))
            # Save individual depth GLM plots immediately
            save_individual_depth_glm_plots(df, os.path.join(output_dir, "depth_glm_analysis"))
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total samples analyzed: {len(df)}")
    print(f"Controls: {df['is_control'].sum()}")
    print(f"Samples: {(~df['is_control']).sum()}")
    
    # Check for depth bias
    print("\n=== DEPTH BIAS ANALYSIS ===")
    if 'mean_coverage' in df.columns and 'high_rate' in df.columns:
        # Calculate correlation between depth and mutation rate
        valid_mask = df['mean_coverage'].notna() & df['high_rate'].notna()
        if valid_mask.sum() > 3:
            correlation, p_value = stats.pearsonr(df.loc[valid_mask, 'mean_coverage'], 
                                                df.loc[valid_mask, 'high_rate'])
            print(f"Depth vs High Rate correlation: {correlation:.3f} (p={p_value:.3f})")
            
            if abs(correlation) > 0.3 and p_value < 0.05:
                print("⚠️  WARNING: Significant depth bias detected!")
                print("   Consider using depth-based GLM or alpha correction")
            else:
                print("✅ No significant depth bias detected")
    
    # Compare GLM methods if available
    if 'glm_rate' in df.columns and 'depth_glm_rate' in df.columns:
        valid_mask = df['glm_rate'].notna() & df['depth_glm_rate'].notna()
        if valid_mask.sum() > 3:
            correlation, p_value = stats.pearsonr(df.loc[valid_mask, 'glm_rate'], 
                                                df.loc[valid_mask, 'depth_glm_rate'])
            print(f"Per-sample vs Depth-based GLM correlation: {correlation:.3f} (p={p_value:.3f})")
            
            if correlation < 0.8:
                print("⚠️  WARNING: Large difference between GLM methods!")
                print("   This suggests depth bias is affecting the estimates")
            else:
                print("✅ GLM methods are consistent")
    
    if alpha > 0:
        print(f"Alpha correction applied: {alpha:.2e}")
    
    if args.glm:
        print(f"GLM analysis: {args.glm_method} method (per-sample + depth-based + per-gene)")
        if args.gff_file:
            print(f"Per-gene GLM: enabled (GFF: {args.gff_file})")
        else:
            print("Per-gene GLM: disabled (no GFF file)")
    else:
        print("GLM analysis: disabled")
    
    if not df.empty:
        print(f"\nHigh Rate Range: {df['high_rate'].min():.2e} - {df['high_rate'].max():.2e}")
        print(f"Low Rate Range: {df['low_rate'].min():.2e} - {df['low_rate'].max():.2e}")
        
        if 'high_rate_alpha_corrected' in df.columns:
            print(f"Alpha-Corrected High Rate Range: {df['high_rate_alpha_corrected'].min():.2e} - {df['high_rate_alpha_corrected'].max():.2e}")
        
        if 'glm_rate' in df.columns and df['glm_rate'].notna().any():
            print(f"Per-Sample GLM Rate Range: {df['glm_rate'].min():.2e} - {df['glm_rate'].max():.2e}")
        
        if 'depth_glm_rate' in df.columns and df['depth_glm_rate'].notna().any():
            print(f"Depth-Based GLM Rate Range: {df['depth_glm_rate'].min():.2e} - {df['depth_glm_rate'].max():.2e}")
        
        if 'per_gene_glm_rate' in df.columns and df['per_gene_glm_rate'].notna().any():
            print(f"Per-Gene GLM Rate Range: {df['per_gene_glm_rate'].min():.2e} - {df['per_gene_glm_rate'].max():.2e}")
        
        # Top 5 samples by high rate
        top_samples = df.nlargest(5, 'high_rate')[['label', 'high_rate', 'low_rate']]
        print(f"\nTop 5 samples by high rate:")
        for _, row in top_samples.iterrows():
            print(f"  {row['label']}: High={row['high_rate']:.2e}, Low={row['low_rate']:.2e}")
            
        # Top 5 samples by alpha-corrected rate (if available)
        if 'high_rate_alpha_corrected' in df.columns:
            top_corrected = df.nlargest(5, 'high_rate_alpha_corrected')[['label', 'high_rate_alpha_corrected']]
            print(f"\nTop 5 samples by alpha-corrected high rate:")
            for _, row in top_corrected.iterrows():
                print(f"  {row['label']}: Alpha-corrected={row['high_rate_alpha_corrected']:.2e}")


if __name__ == "__main__":
    main()
