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


def estimate_rates_glm_genome_wide(df, method='poisson', use_f_clonal=False, f_clonal_regions=None, no_f_clonal=False, use_treatment_covariate=True):
    """Fit GLM to estimate mutation rate with better uncertainty quantification.
    
    If use_treatment_covariate=True (default), fits a pooled model across all samples
    with treatment covariate. Otherwise uses per-sample models with alpha subtraction.
    """
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
    df_valid = df[valid_mask].copy()
    
    # Calculate detection probability if using f-clonal (and not disabled)
    if use_f_clonal and f_clonal_regions and not no_f_clonal:
        # For genome-wide GLM, use average f-clonal across all regions
        avg_f_clonal = np.mean(list(f_clonal_regions.values()))
        detection_probs = np.array([calculate_detection_probability(d, avg_f_clonal) for d in exposure])
        # Adjust exposure by detection probability
        exposure = exposure * detection_probs
    
    if use_treatment_covariate:
        # Treatment covariate approach: fit pooled model with treatment covariate
        # Create treatment indicator (0 for controls, 1 for treated)
        treatment = (~df_valid['is_control']).astype(int).values
        
        # Prepare design matrix with treatment covariate
        X = pd.DataFrame({'treatment': treatment})
        design = sm.add_constant(X, has_constant='add')
        offset = np.log(exposure)
        
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if method == 'poisson':
                    fam = sm.families.Poisson()
                elif method == 'negative_binomial':
                    fam = sm.families.NegativeBinomial(alpha=getattr(df, "_nb_alpha", None) or 1.0)
                
                model = sm.GLM(y, design, family=fam, offset=offset)
                result = model.fit()
                
                # Extract coefficients
                beta0 = result.params.get('const', np.nan)
                beta_treatment = result.params.get('treatment', np.nan)
                
                # Calculate rates for each sample
                for i, (idx, row) in enumerate(df_valid.iterrows()):
                    is_control = row['is_control']
                    if is_control:
                        # Control rate: exp(beta0)
                        rate = np.exp(beta0) if np.isfinite(beta0) else np.nan
                        # CI for intercept
                        ci_beta = result.conf_int().loc['const'].values if 'const' in result.conf_int().index else [np.nan, np.nan]
                        ci = np.exp(ci_beta)
                    else:
                        # Treated rate: exp(beta0 + beta_treatment)
                        if np.isfinite(beta0) and np.isfinite(beta_treatment):
                            rate = np.exp(beta0 + beta_treatment)
                            # CI for treated rate using delta method
                            # Var(beta0 + beta_treatment) = Var(beta0) + Var(beta_treatment) + 2*Cov(beta0, beta_treatment)
                            try:
                                cov_matrix = result.cov_params()
                                var_const = cov_matrix.loc['const', 'const'] if 'const' in cov_matrix.index else np.nan
                                var_treat = cov_matrix.loc['treatment', 'treatment'] if 'treatment' in cov_matrix.index else np.nan
                                cov_const_treat = cov_matrix.loc['const', 'treatment'] if 'const' in cov_matrix.index and 'treatment' in cov_matrix.columns else 0.0
                                
                                if np.isfinite(var_const) and np.isfinite(var_treat):
                                    var_sum = var_const + var_treat + 2 * cov_const_treat
                                    se_sum = np.sqrt(var_sum)
                                    # 95% CI on log scale
                                    log_rate = beta0 + beta_treatment
                                    ci_low_log = log_rate - 1.96 * se_sum
                                    ci_high_log = log_rate + 1.96 * se_sum
                                    ci = [np.exp(ci_low_log), np.exp(ci_high_log)]
                                else:
                                    # Fallback: use individual CIs (less accurate)
                                    ci_const = result.conf_int().loc['const'].values if 'const' in result.conf_int().index else [np.nan, np.nan]
                                    ci_treat = result.conf_int().loc['treatment'].values if 'treatment' in result.conf_int().index else [np.nan, np.nan]
                                    ci_low = np.exp(ci_const[0] + ci_treat[0])
                                    ci_high = np.exp(ci_const[1] + ci_treat[1])
                                    ci = [ci_low, ci_high]
                            except Exception:
                                # Fallback: use individual CIs
                                ci_const = result.conf_int().loc['const'].values if 'const' in result.conf_int().index else [np.nan, np.nan]
                                ci_treat = result.conf_int().loc['treatment'].values if 'treatment' in result.conf_int().index else [np.nan, np.nan]
                                ci_low = np.exp(ci_const[0] + ci_treat[0])
                                ci_high = np.exp(ci_const[1] + ci_treat[1])
                                ci = [ci_low, ci_high]
                        else:
                            rate = np.nan
                            ci = [np.nan, np.nan]
                    
                    df.loc[idx, 'glm_rate'] = rate
                    df.loc[idx, 'glm_CI_low'] = ci[0]
                    df.loc[idx, 'glm_CI_high'] = ci[1]
        
        except Exception as e:
            print(f"Warning: Pooled GLM with treatment covariate failed: {e}")
            # Fall back to per-sample approach
            use_treatment_covariate = False
    
    if not use_treatment_covariate:
        # Legacy approach: per-sample models with alpha subtraction
        alpha_for_glm = float(df['alpha_used'].iloc[0]) if 'alpha_used' in df.columns and len(df['alpha_used']) > 0 else 0.0
        
        # Fit individual GLM for each sample
        for i, (idx, row) in enumerate(df_valid.iterrows()):
            sample_mutations = y[i]
            sample_exposure = exposure[i]
            
            if sample_exposure > 0:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        # Subtract background alpha at counts level
                        sample_mutations_adj = max(0, sample_mutations - alpha_for_glm * sample_exposure)
                        if method == 'poisson':
                            model = sm.GLM([sample_mutations_adj], [1], family=sm.families.Poisson(), 
                                          offset=[np.log(sample_exposure)])
                            result = model.fit()
                            rate = np.exp(result.params[0])
                            ci = np.exp(result.conf_int()[0])
                            
                        elif method == 'negative_binomial':
                            fam = sm.families.NegativeBinomial(alpha=getattr(df, "_nb_alpha", None) or 1.0)
                            model = sm.GLM([sample_mutations_adj], [1], family=fam, 
                                           offset=[np.log(sample_exposure)])
                            result = model.fit()
                            rate = np.exp(result.params[0])
                            ci = np.exp(result.conf_int()[0])
                    
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
                        fam = sm.families.NegativeBinomial(alpha=getattr(result, "_nb_alpha", None) or 1.0)
                        model = sm.GLM([mutations_adj], [1], family=fam, 
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
                            fam = sm.families.NegativeBinomial(alpha=alpha_for_glm if hasattr(sm.families, 'NegativeBinomial') else 1.0)
                            model = sm.GLM([mutations_adj], [1], family=fam, 
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


def create_site_level_glm_plots(df, output_prefix):
    """Create plots for per-sample site-level GLM rates."""
    if 'site_glm_rate' not in df.columns or df['site_glm_rate'].notna().sum() == 0:
        print("No per-sample site-level GLM data available for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Ranking plot
    df_sorted = df.sort_values('site_glm_rate', ascending=False)
    colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
    ax1.barh(range(len(df_sorted)), df_sorted['site_glm_rate'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['label'], fontsize=12)
    ax1.set_xlabel('Per-base mutation probability (site-level GLM)', fontsize=14)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax1.xaxis.set_major_formatter(fmt)
    ax1.set_title('Site-Level GLM Rate Ranking', fontsize=16)
    ax1.invert_yaxis()

    # Error bars if available
    if 'site_glm_CI_low' in df_sorted.columns and 'site_glm_CI_high' in df_sorted.columns:
        y_pos = range(len(df_sorted))
        xerr_low = df_sorted['site_glm_rate'] - df_sorted['site_glm_CI_low']
        xerr_high = df_sorted['site_glm_CI_high'] - df_sorted['site_glm_rate']
        ax1.errorbar(df_sorted['site_glm_rate'], y_pos,
                     xerr=[xerr_low, xerr_high], fmt='none', color='black', capsize=2, alpha=0.6)

    # Comparison vs per-sample GLM if available
    if 'glm_rate' in df.columns and df['glm_rate'].notna().any():
        control_mask = df['is_control']
        sample_mask = ~control_mask
        if control_mask.any():
            ax2.scatter(df.loc[control_mask, 'glm_rate'], df.loc[control_mask, 'site_glm_rate'],
                        color='#1F77B4', s=100, alpha=0.7, label='Controls')
        if sample_mask.any():
            ax2.scatter(df.loc[sample_mask, 'glm_rate'], df.loc[sample_mask, 'site_glm_rate'],
                        color='#FF7F0E', s=100, alpha=0.7, label='Samples')
        ax2.set_xlabel('Per-base probability (per-sample GLM)', fontsize=14)
        ax2.set_ylabel('Per-base probability (site-level GLM)', fontsize=14)
        ax2.xaxis.set_major_formatter(fmt)
        ax2.yaxis.set_major_formatter(fmt)
        ax2.set_title('Per-Sample GLM vs Site-Level GLM', fontsize=16)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Per-sample GLM not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Site-Level GLM Analysis', fontsize=16)

    plt.tight_layout()
    plot_path = f"{output_prefix}_site_glm_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Site-level GLM analysis plots saved to {plot_path}")


def save_individual_site_level_glm_plots(df, output_prefix):
    """Save individual ranking plot for per-sample site-level GLM rates."""
    if 'site_glm_rate' not in df.columns or df['site_glm_rate'].notna().sum() == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    df_sorted = df.sort_values('site_glm_rate', ascending=False)
    colors = ['#FF7F0E' if not is_ctrl else '#1F77B4' for is_ctrl in df_sorted['is_control']]
    ax.barh(range(len(df_sorted)), df_sorted['site_glm_rate'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['label'], fontsize=10)
    ax.set_xlabel('Per-base mutation probability (site-level GLM)')
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(fmt)
    ax.set_title('Site-Level GLM Rate Ranking')
    ax.invert_yaxis()
    if 'site_glm_CI_low' in df_sorted.columns and 'site_glm_CI_high' in df_sorted.columns:
        y_pos = range(len(df_sorted))
        xerr_low = df_sorted['site_glm_rate'] - df_sorted['site_glm_CI_low']
        xerr_high = df_sorted['site_glm_CI_high'] - df_sorted['site_glm_rate']
        ax.errorbar(df_sorted['site_glm_rate'], y_pos,
                    xerr=[xerr_low, xerr_high], fmt='none', color='black', capsize=2, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_site_glm_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Site-level GLM ranking saved to {output_prefix}_site_glm_ranking.png")

def load_site_level_data_no_context(counts_dir: str, exclusion_mask: str | None = None):
    """Load site-level rows across all samples without sequence-context features.

    Returns a DataFrame with columns: chrom, pos, treatment (0 control, 1 treated),
    depth, ems_count, sample.
    """
    excluded_sites = None
    if exclusion_mask and os.path.exists(exclusion_mask):
        excluded_sites = load_exclusion_mask(exclusion_mask)

    rows = []
    counts_files = sorted(glob.glob(os.path.join(counts_dir, "*.counts")))
    for path in counts_files:
        sample_name = os.path.basename(path).replace(".counts", "")
        is_control = ("NT" in sample_name)
        treatment = 0 if is_control else 1
        with open(path) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                try:
                    A_i = int(A_count)
                    C_i = int(C_count)
                    G_i = int(G_count)
                    T_i = int(T_count)
                    depth_i = int(depth)
                except Exception:
                    continue
                if depth_i <= 0:
                    continue
                if ref not in {"G", "C"}:
                    continue
                # Strict EMS-only counting
                ems_count = T_i if ref == "C" else A_i
                rows.append({
                    "chrom": chrom,
                    "pos": pos,
                    "treatment": treatment,
                    "depth": depth_i,
                    "ems_count": ems_count,
                    "sample": sample_name,
                })
    return pd.DataFrame(rows)


def fit_site_level_glm_no_context(df: pd.DataFrame, method: str = "poisson", alpha: float = 0.0):
    """Fit site-level GLM with Treatment covariate and log(depth) offset.

    Optionally subtract background alpha at counts level per site (max with 0).
    Returns the fitted model result object and a small summary dict.
    """
    if df.empty:
        return None, {}

    work = df.copy()
    work["log_depth"] = np.log(work["depth"].values)

    # Apply alpha subtraction at counts level if provided
    if alpha and alpha > 0:
        work["ems_adj"] = np.maximum(0.0, work["ems_count"].values - alpha * work["depth"].values)
        y = work["ems_adj"].values
    else:
        y = work["ems_count"].values

    X = work[["treatment"]]
    design = sm.add_constant(X, has_constant='add')

    if method == "poisson":
        fam = sm.families.Poisson()
    else:
        fam = sm.families.NegativeBinomial(alpha=getattr(df, "_nb_alpha", None) or 1.0)

    model = sm.GLM(y, design, family=fam, offset=work["log_depth"].values)
    result = model.fit()

    # Interpret coefficients as per-base rates for control (intercept) and treated
    beta0 = result.params.get("const", np.nan)
    beta_t = result.params.get("treatment", np.nan)
    rate_control = float(np.exp(beta0)) if np.isfinite(beta0) else np.nan
    rate_treated = float(np.exp(beta0 + beta_t)) if np.isfinite(beta0) and np.isfinite(beta_t) else np.nan

    summary = {
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)) if hasattr(result, "bic") else np.nan,
        "llf": float(getattr(result, "llf", np.nan)),
        "rate_control": rate_control,
        "rate_treated": rate_treated,
        "coef_const": float(beta0) if np.isfinite(beta0) else np.nan,
        "coef_treatment": float(beta_t) if np.isfinite(beta_t) else np.nan,
    }

    return result, summary


def fit_site_level_glm_per_sample(site_df: pd.DataFrame, method: str = "poisson", alpha: float = 0.0):
    """Fit intercept-only site-level GLM separately for each sample.

    Returns a list of dicts with sample-level rate estimates and intervals.
    """
    if site_df is None or site_df.empty:
        return []

    results = []
    samples = sorted(site_df["sample"].unique())

    for sample_name in samples:
        df = site_df[site_df["sample"] == sample_name].copy()
        if df.empty:
            continue

        df["log_depth"] = np.log(df["depth"].values)

        if alpha and alpha > 0:
            df["ems_adj"] = np.maximum(0.0, df["ems_count"].values - alpha * df["depth"].values)
            y = df["ems_adj"].values
        else:
            y = df["ems_count"].values

        # Intercept-only design
        design = sm.add_constant(pd.DataFrame(index=df.index), has_constant='add')

        if method == "poisson":
            fam = sm.families.Poisson()
        else:
            fam = sm.families.NegativeBinomial()

        try:
            model = sm.GLM(y, design, family=fam, offset=df["log_depth"].values)
            result = model.fit()

            beta0 = result.params.get("const", np.nan)
            rate = float(np.exp(beta0)) if np.isfinite(beta0) else np.nan

            # Confidence interval for intercept, transformed
            try:
                ci_beta = result.conf_int().loc["const"].values
                ci_low = float(np.exp(ci_beta[0]))
                ci_high = float(np.exp(ci_beta[1]))
            except Exception:
                ci_low = np.nan
                ci_high = np.nan

            results.append({
                "sample": sample_name,
                "site_glm_rate": rate,
                "site_glm_CI_low": ci_low,
                "site_glm_CI_high": ci_high,
                "site_glm_aic": float(getattr(result, "aic", np.nan)),
                "site_glm_llf": float(getattr(result, "llf", np.nan)),
                "site_glm_rows": int(len(df)),
                "site_glm_depth": int(df["depth"].sum()),
                "site_glm_mutations": float(df["ems_count"].sum()),
            })
        except Exception as e:
            print(f"Warning: Site-level GLM failed for sample {sample_name}: {e}")
            results.append({
                "sample": sample_name,
                "site_glm_rate": np.nan,
                "site_glm_CI_low": np.nan,
                "site_glm_CI_high": np.nan,
                "site_glm_aic": np.nan,
                "site_glm_llf": np.nan,
                "site_glm_rows": int(len(df)),
                "site_glm_depth": int(df["depth"].sum()),
                "site_glm_mutations": float(df["ems_count"].sum()),
            })

    return results


def load_site_level_with_ref(counts_dir: str, exclusion_mask: str | None = None) -> pd.DataFrame:
    """Load site-level rows with reference base retained for min-alt logic.

    Columns: chrom, pos(int), ref, depth, ems_count, sample, is_control(bool)
    """
    excluded_sites = None
    if exclusion_mask and os.path.exists(exclusion_mask):
        excluded_sites = load_exclusion_mask(exclusion_mask)

    rows = []
    counts_files = sorted(glob.glob(os.path.join(counts_dir, "*.counts")))
    for path in counts_files:
        sample = os.path.basename(path).replace(".counts", "")
        is_ctrl = ("NT" in sample)
        with open(path) as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader, None)
            for row in reader:
                if len(row) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = row
                if ref not in {"C", "G"}:
                    continue
                if excluded_sites and (chrom, pos) in excluded_sites:
                    continue
                try:
                    A_i = int(A_count); C_i = int(C_count); G_i = int(G_count); T_i = int(T_count)
                    depth_i = int(depth); pos_i = int(pos)
                except Exception:
                    continue
                if depth_i <= 0:
                    continue
                ems = T_i if ref == "C" else A_i
                rows.append({
                    "chrom": chrom,
                    "pos": pos_i,
                    "ref": ref,
                    "depth": depth_i,
                    "ems_count": ems,
                    "sample": sample,
                    "is_control": is_ctrl,
                })
    return pd.DataFrame(rows)


def fit_site_level_glm_intercept_only(df_sites: pd.DataFrame, method: str = "poisson", alpha: float = 0.0) -> tuple[float, float, float]:
    """Fit intercept-only site-level GLM on provided site rows for a single sample.

    Expects columns: depth, y (counts to model). Applies alpha subtraction at counts level.
    Returns (rate, ci_low, ci_high) in per-base units.
    """
    if df_sites is None or df_sites.empty:
        return (float("nan"), float("nan"), float("nan"))

    work = df_sites.copy()
    work["log_depth"] = np.log(work["depth"].values)

    if alpha and alpha > 0:
        y = np.maximum(0.0, work["y"].values - alpha * work["depth"].values)
    else:
        y = work["y"].values

    design = sm.add_constant(pd.DataFrame(index=work.index), has_constant='add')
    fam = sm.families.Poisson() if method == "poisson" else sm.families.NegativeBinomial(alpha=getattr(df_sites, "_nb_alpha", None) or 1.0)

    try:
        model = sm.GLM(y, design, family=fam, offset=work["log_depth"].values)
        res = model.fit()
        b0 = res.params.get("const", np.nan)
        rate = float(np.exp(b0)) if np.isfinite(b0) else np.nan
        try:
            ci_b = res.conf_int().loc["const"].values
            ci_low = float(np.exp(ci_b[0])); ci_high = float(np.exp(ci_b[1]))
        except Exception:
            ci_low = np.nan; ci_high = np.nan
        return (rate, ci_low, ci_high)
    except Exception:
        return (float("nan"), float("nan"), float("nan"))

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
                        help="Skip alpha correction for simple rates (GLM methods use treatment covariate by default)")
    parser.add_argument("--use-alpha-correction", action="store_true",
                        help="Use alpha correction approach instead of treatment covariate for GLM methods (legacy)")
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
    parser.add_argument("--site-glm", action="store_true",
                        help="Run site-level GLM without sequence-context covariates")
    parser.add_argument("--site-glm-method", type=str, default="poisson",
                        choices=["poisson", "negative_binomial"],
                        help="GLM family for site-level model (default: poisson)")
    parser.add_argument("--glm-use-nb-alpha-from-controls", action="store_true",
                        help="For negative_binomial GLMs, estimate NB alpha from NT controls (site-level) and use it")
    parser.add_argument("--site-glm-use-nb-alpha-from-controls", action="store_true",
                        help="For site-level GLM negative_binomial, estimate NB alpha from NT controls (site-level) and use it")

    # Sweep options (site-level GLM)
    parser.add_argument("--sweep", action="store_true",
                        help="Run a min-alt sweep using site-level per-sample GLM; writes TSVs and plots similar to the sweep script")
    parser.add_argument("--min-alt-start", type=int, default=2,
                        help="Sweep start threshold (inclusive) for min-alt")
    parser.add_argument("--min-alt-end", type=int, default=5,
                        help="Sweep end threshold (inclusive) for min-alt")
    parser.add_argument("--sweep-target", type=str, default="C", choices=["C", "G"],
                        help="Which base to sweep (C or G); the other base fixed at --other-min-alt")
    parser.add_argument("--other-min-alt", type=int, default=1,
                        help="Fixed min-alt to apply to the non-swept base during sweep (default 1)")
    parser.add_argument("--use-split-alpha", action="store_true",
                        help="During sweep, attempt separate alpha for genic vs intergenic (requires --gff-file); otherwise use overall alpha")
    
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

    # Step 2: Determine approach for handling controls
    # Default: Treatment covariate approach (controls as covariate in GLM)
    # Legacy: Alpha correction approach (subtract background rate from counts)
    alpha = 0.0
    nb_alpha_controls = None
    use_treatment_covariate = not args.use_alpha_correction  # Default to True unless user requests alpha correction
    
    if df['is_control'].any():
        if use_treatment_covariate:
            print("Using treatment covariate approach (controls as covariate in GLM)")
            print("  This fits a pooled model: log(E[Y]) = log(depth) + β₀ + β₁·treatment")
            print("  Control rate = exp(β₀), Treated rate = exp(β₀ + β₁)")
        else:
            print("Using alpha correction approach (legacy method)")
            print("  This subtracts background rate from mutation counts before fitting")
        
        # Estimate alpha for reference or for simple rate alpha correction
        control_files = [f for f in counts_files if "NT" in os.path.basename(f)]
        if control_files:
            alpha = estimate_alpha_from_control_sites(control_files, min_alt=args.min_alt, excluded_sites=excluded_sites)
            print(f"Estimated alpha from pooled control sites (background rate): {alpha:.2e}")
        else:
            alpha = estimate_alpha_from_controls(df)
            print(f"Estimated alpha from weighted control rates (background rate): {alpha:.2e}")
    else:
        print("No control samples found, using intercept-only models")
        use_treatment_covariate = False

    # Step 3: Apply alpha correction to simple rates (if not skipped)
    # Note: GLM methods use treatment covariate approach by default
    if alpha > 0 and not args.no_alpha_correction:
        df = apply_alpha_correction(df, alpha)
        print("Applied alpha correction to simple rate estimates")

    # Optional: Estimate NB alpha from controls at site-level for NB GLMs
    if (args.glm and args.glm_method == 'negative_binomial' and args.glm_use_nb_alpha_from_controls) or (args.site_glm and args.site_glm_method == 'negative_binomial' and args.site_glm_use_nb_alpha_from_controls):
        try:
            site_controls = load_site_level_with_ref(args.counts_dir, args.exclusion_mask)
            site_controls = site_controls[site_controls['is_control']].copy()
            if not site_controls.empty:
                site_controls['log_depth'] = np.log(site_controls['depth'].values)
                y = site_controls['ems_count'].values
                design = sm.add_constant(pd.DataFrame(index=site_controls.index), has_constant='add')
                pois = sm.GLM(y, design, family=sm.families.Poisson(), offset=site_controls['log_depth'].values).fit()
                mu = pois.fittedvalues
                var_y = np.var(y)
                mean_mu = np.mean(mu)
                if var_y > mean_mu and mean_mu > 0:
                    nb_alpha_controls = float((mean_mu ** 2) / max(var_y - mean_mu, 1e-12))
                    print(f"Estimated NB alpha from controls (site-level): {nb_alpha_controls:.4f}")
                else:
                    print("Controls suggest Poisson-like dispersion; skipping NB alpha override")
            else:
                print("No control site-level rows found for NB alpha estimation")
        except Exception as e:
            print(f"Warning: NB alpha estimation from controls failed: {e}")

    # Step 4: GLM analysis (if requested)
    if args.glm:
        print(f"Running GLM analysis (method={args.glm_method})")
        
        # 1) Per-sample GLM (genome-wide, no f-clonal)
        # Uses treatment covariate approach by default (pooled model with treatment covariate)
        print("Fitting genome-wide GLM with treatment covariate...")
        if nb_alpha_controls is not None:
            df._nb_alpha = nb_alpha_controls
        df = estimate_rates_glm_genome_wide(df, method=args.glm_method, use_f_clonal=False, f_clonal_regions=None, no_f_clonal=True, use_treatment_covariate=use_treatment_covariate)
        
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

    # Optional: Site-level GLM without sequence-context covariates
    if args.site_glm:
        print(f"Running site-level GLM (method={args.site_glm_method}) without sequence-context covariates")
        site_df = load_site_level_data_no_context(args.counts_dir, args.exclusion_mask)
        if site_df is None or site_df.empty:
            print("No site-level data loaded; skipping site-level GLM")
        else:
            site_result, site_summary = fit_site_level_glm_no_context(site_df, method=args.site_glm_method, alpha=alpha)

            # Save a concise report
            site_dir = args.output_prefix
            os.makedirs(site_dir, exist_ok=True)
            report_path = os.path.join(site_dir, "site_level_glm_report.txt")
            with open(report_path, "w") as fh:
                fh.write("=== Site-Level GLM (no context covariates) ===\n\n")
                fh.write(f"Method: {args.site_glm_method}\n")
                fh.write(f"Rows: {len(site_df)}\n")
                fh.write(f"Alpha subtracted: {alpha:.2e}\n\n")
                if site_result is not None:
                    fh.write(f"AIC: {site_summary.get('aic', float('nan'))}\n")
                    fh.write(f"BIC: {site_summary.get('bic', float('nan'))}\n")
                    fh.write(f"LogLik: {site_summary.get('llf', float('nan'))}\n\n")
                    fh.write("Coefficients (log-rate scale):\n")
                    fh.write(f"  const: {site_summary.get('coef_const', float('nan'))}\n")
                    fh.write(f"  treatment: {site_summary.get('coef_treatment', float('nan'))}\n\n")
                    fh.write("Implied per-base rates:\n")
                    fh.write(f"  control: {site_summary.get('rate_control', float('nan')):.6e}\n")
                    fh.write(f"  treated: {site_summary.get('rate_treated', float('nan')):.6e}\n")
                else:
                    fh.write("Model fitting failed.\n")
            print(f"Site-level GLM report saved to {report_path}")

            # Per-sample site-level GLM (intercept-only per sample)
            print("Fitting per-sample site-level GLMs (intercept-only)...")
            per_sample_rows = fit_site_level_glm_per_sample(site_df, method=args.site_glm_method, alpha=alpha)
            per_sample_df = pd.DataFrame(per_sample_rows)
            tsv_path = os.path.join(site_dir, "site_level_glm_per_sample.tsv")
            per_sample_df.to_csv(tsv_path, sep="\t", index=False)
            print(f"Per-sample site-level GLM rates saved to {tsv_path}")

            # Merge per-sample site GLM rates into main results table
            if not per_sample_df.empty:
                df = df.merge(per_sample_df[["sample", "site_glm_rate", "site_glm_CI_low", "site_glm_CI_high"]], on="sample", how="left")

            # Create per-sample site-level GLM plots
            create_site_level_glm_plots(df, os.path.join(site_dir, "site_glm"))
            save_individual_site_level_glm_plots(df, os.path.join(site_dir, "site_glm"))

    # Optional: Site-level GLM sweep (overall + optional genic/intergenic)
    if args.sweep:
        print(f"Running site-level GLM sweep: {args.sweep_target} min_alt {args.min_alt_start}..{args.min_alt_end}, other={args.other_min_alt}")
        sweep_dir = os.path.join(args.output_prefix, "sweep")
        os.makedirs(sweep_dir, exist_ok=True)

        # Load site-level rows with reference base
        site_all = load_site_level_with_ref(args.counts_dir, args.exclusion_mask)
        if site_all is None or site_all.empty:
            print("No site-level rows found; skipping sweep")
        else:
            thresholds = list(range(args.min_alt_start, args.min_alt_end + 1))

            # Prepare genic regions (optional)
            gene_regions = None
            if args.gff_file and os.path.exists(args.gff_file):
                print("Parsing GFF for genic/intergenic split...")
                gene_regions = parse_gff(args.gff_file)

            # Helper to test genic membership (simple scan over gene list by chrom)
            chrom_to_genes = {}
            if gene_regions:
                for chrom, start, end, gid in gene_regions:
                    chrom_to_genes.setdefault(chrom, []).append((start, end))
                for chrom in chrom_to_genes:
                    chrom_to_genes[chrom].sort()

            def is_genic(chrom: str, pos: int) -> bool:
                if not chrom_to_genes:
                    return False
                genes = chrom_to_genes.get(chrom)
                if not genes:
                    return False
                # Linear scan (acceptable for moderate sizes); could be optimized with bisect
                for s, e in genes:
                    if pos < s:
                        break
                    if s <= pos <= e:
                        return True
                return False

            # Alpha per threshold (overall); split alpha optional if genic available
            control_files = [f for f in counts_files if "NT" in os.path.basename(f)]

            overall_wide_frames = []
            genic_wide_frames = []
            inter_wide_frames = []
            trends_rows = []

            for thr in thresholds:
                thr_c = thr if args.sweep_target == "C" else args.other_min_alt
                thr_g = thr if args.sweep_target == "G" else args.other_min_alt

                # Estimate alpha overall for this threshold from controls
                alpha_thr = 0.0
                if control_files and not args.no_alpha_correction:
                    alpha_thr = estimate_alpha_from_control_sites(control_files, min_alt=min(thr_c, thr_g), excluded_sites=excluded_sites)

                # Prepare thresholded counts y
                work = site_all.copy()
                mask_c = (work["ref"] == "C") & (work["ems_count"] >= thr_c)
                mask_g = (work["ref"] == "G") & (work["ems_count"] >= thr_g)
                work["y"] = np.where(mask_c | mask_g, work["ems_count"], 0)

                # Per-sample overall site-level GLM
                samples = sorted(work["sample"].unique())
                overall_rows = []
                genic_rows = []
                inter_rows = []
                for s in samples:
                    sub = work[work["sample"] == s]
                    is_ctrl = bool(("NT" in s))

                    r_o, lo_o, hi_o = fit_site_level_glm_intercept_only(sub, method=args.site_glm_method, alpha=alpha_thr)
                    overall_rows.append({"sample": s, "rate": r_o, "lo": lo_o, "hi": hi_o})

                    # Genic/intergenic split if GFF available
                    if gene_regions:
                        mask_gx = sub.apply(lambda r: is_genic(r["chrom"], int(r["pos"])), axis=1)
                        sub_g = sub[mask_gx]
                        sub_i = sub[~mask_gx]

                        # If requested, we could compute split alphas; for now use overall alpha_thr
                        r_g, lo_g, hi_g = fit_site_level_glm_intercept_only(sub_g, method=args.site_glm_method, alpha=alpha_thr)
                        r_i, lo_i, hi_i = fit_site_level_glm_intercept_only(sub_i, method=args.site_glm_method, alpha=alpha_thr)
                        genic_rows.append({"sample": s, "rate": r_g, "lo": lo_g, "hi": hi_g})
                        inter_rows.append({"sample": s, "rate": r_i, "lo": lo_i, "hi": hi_i})

                    # Trends row
                    trends_rows.append({
                        "sample": s,
                        "min_alt": thr,
                        "rate": r_o,
                        "is_control": is_ctrl,
                    })

                label = f"{args.sweep_target}_minAlt{thr}"
                over_df = pd.DataFrame(overall_rows)
                overall_wide_frames.append(over_df[["sample", "rate"]].rename(columns={"rate": f"rate_{label}"}))

                if gene_regions:
                    gdf = pd.DataFrame(genic_rows)
                    idf = pd.DataFrame(inter_rows)
                    genic_wide_frames.append(gdf[["sample", "rate", "lo", "hi"]].rename(columns={
                        "rate": f"genic_rate_{label}", "lo": f"genic_CI_low_{label}", "hi": f"genic_CI_high_{label}"
                    }))
                    inter_wide_frames.append(idf[["sample", "rate", "lo", "hi"]].rename(columns={
                        "rate": f"intergenic_rate_{label}", "lo": f"intergenic_CI_low_{label}", "hi": f"intergenic_CI_high_{label}"
                    }))

            # Merge and save TSVs
            if overall_wide_frames:
                rates_wide = overall_wide_frames[0]
                for f in overall_wide_frames[1:]:
                    rates_wide = rates_wide.merge(f, on="sample", how="outer")
                rates_wide_path = os.path.join(sweep_dir, "site_level_overall_rates_wide.tsv")
                rates_wide.to_csv(rates_wide_path, sep="\t", index=False)
                print(f"Saved sweep overall wide TSV to {rates_wide_path}")

                rates_long = rates_wide.melt(id_vars=["sample"], var_name="threshold", value_name="rate")
                rates_long["min_alt"] = rates_long["threshold"].str.extract(r"(\d+)").astype(int)
                rates_long_path = os.path.join(sweep_dir, "site_level_overall_rates_long.tsv")
                rates_long.sort_values(["sample", "min_alt"]).to_csv(rates_long_path, sep="\t", index=False)
                print(f"Saved sweep overall long TSV to {rates_long_path}")

                # Trend plot
                trend_df = pd.DataFrame(trends_rows)
                if not trend_df.empty:
                    plt.figure(figsize=(10, 7))
                    control_color = "#1f77b4"; treated_color = "#ff7f0e"
                    for s, sub in trend_df.groupby("sample"):
                        sub = sub.sort_values("min_alt")
                        color = control_color if bool(("NT" in s)) else treated_color
                        plt.plot(sub["min_alt"], sub["rate"], marker="o", alpha=0.55, linewidth=1, color=color)
                    plt.yscale("log")
                    plt.xlabel("min_alt"); plt.ylabel("Per-base mutation probability (site-level GLM)")
                    plt.title("Per-sample site-level GLM rate trends vs min_alt")
                    legend_elements = [Line2D([0],[0], color=treated_color, lw=2, label="Treated"), Line2D([0],[0], color=control_color, lw=2, label="Control")]
                    plt.legend(handles=legend_elements, loc="best")
                    trend_path = os.path.join(sweep_dir, "site_level_trends_overall.png")
                    plt.tight_layout(); plt.savefig(trend_path, dpi=300); plt.close()
                    print(f"Saved sweep overall trend plot to {trend_path}")

            if gene_regions and genic_wide_frames and inter_wide_frames:
                genic_wide = genic_wide_frames[0]
                for f in genic_wide_frames[1:]:
                    genic_wide = genic_wide.merge(f, on="sample", how="outer")
                inter_wide = inter_wide_frames[0]
                for f in inter_wide_frames[1:]:
                    inter_wide = inter_wide.merge(f, on="sample", how="outer")
                genic_wide_path = os.path.join(sweep_dir, "site_level_genic_rates_wide.tsv")
                inter_wide_path = os.path.join(sweep_dir, "site_level_intergenic_rates_wide.tsv")
                genic_wide.to_csv(genic_wide_path, sep="\t", index=False)
                inter_wide.to_csv(inter_wide_path, sep="\t", index=False)
                print(f"Saved sweep genic/intergenic wide TSVs to {sweep_dir}")

                # Genic vs intergenic scatter per threshold
                for thr in thresholds:
                    label = f"{args.sweep_target}_minAlt{thr}"
                    gcol = f"genic_rate_{label}"; icol = f"intergenic_rate_{label}"
                    g_lo = f"genic_CI_low_{label}"; g_hi = f"genic_CI_high_{label}"
                    i_lo = f"intergenic_CI_low_{label}"; i_hi = f"intergenic_CI_high_{label}"
                    if gcol not in genic_wide.columns or icol not in inter_wide.columns:
                        continue
                    join_df = genic_wide[["sample", gcol, g_lo, g_hi]].merge(inter_wide[["sample", icol, i_lo, i_hi]], on="sample", how="inner")
                    if join_df.empty:
                        continue
                    join_df["g_err_low"] = join_df[gcol] - join_df[g_lo]
                    join_df["g_err_high"] = join_df[g_hi] - join_df[gcol]
                    join_df["i_err_low"] = join_df[icol] - join_df[i_lo]
                    join_df["i_err_high"] = join_df[i_hi] - join_df[icol]
                    plt.figure(figsize=(7,7))
                    # Control vs treated coloring
                    mask_ctrl = join_df["sample"].str.contains("NT")
                    treated_df = join_df[~mask_ctrl]; control_df = join_df[mask_ctrl]
                    if not treated_df.empty:
                        plt.errorbar(treated_df[gcol], treated_df[icol], xerr=[treated_df["g_err_low"], treated_df["g_err_high"]], yerr=[treated_df["i_err_low"], treated_df["i_err_high"]], fmt='o', color="#ff7f0e", alpha=0.6, elinewidth=1, capsize=2, label="Treated")
                    if not control_df.empty:
                        plt.errorbar(control_df[gcol], control_df[icol], xerr=[control_df["g_err_low"], control_df["g_err_high"]], yerr=[control_df["i_err_low"], control_df["i_err_high"]], fmt='o', color="#1f77b4", alpha=0.6, elinewidth=1, capsize=2, label="Control")
                    plt.xscale("log"); plt.yscale("log")
                    plt.xlabel(f"Genic per-base mutation probability (min_alt={thr})")
                    plt.ylabel(f"Intergenic per-base mutation probability (min_alt={thr})")
                    plt.title("Site-level GLM: Genic vs Intergenic rates (95% CI)")
                    # Diagonal
                    vals = pd.concat([join_df[gcol], join_df[icol]]).replace([np.inf, -np.inf], np.nan).dropna()
                    if not vals.empty:
                        lo_v, hi_v = vals.min(), vals.max()
                        if lo_v > 0 and np.isfinite(hi_v):
                            plt.plot([lo_v, hi_v], [lo_v, hi_v], linestyle="--", color="grey", alpha=0.6)
                    plt.legend(loc="best"); plt.grid(alpha=0.2)
                    scatter_path = os.path.join(sweep_dir, f"site_level_genic_vs_intergenic_minAlt{thr}.png")
                    plt.tight_layout(); plt.savefig(scatter_path, dpi=300); plt.close()
                    print(f"Saved sweep genic vs intergenic scatter for min_alt={thr} to {scatter_path}")

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
