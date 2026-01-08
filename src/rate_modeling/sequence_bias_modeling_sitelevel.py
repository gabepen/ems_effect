#!/usr/bin/env python3
"""
Site-level sequence-bias modeling for EMS mutations using depth-aware GLMs.

Goal:
- Model per-site mutation counts as rates (Poisson with log(depth) offset)
- Compare sequence-context encodings: positional (15), 3mer (64), 5mer (one-hot), and positional-3mer (left/right trimers)
- Select best model by AIC/BIC

Inputs:
- --counts-dir: directory of .counts files (from collect_mutation_counts.py)
- --genome-fasta: reference fasta to extract k-mers
- --exclusion-mask: optional TSV (chrom\tpos) to exclude sites
- Control vs EMS: files with 'NT' in name are treated as controls; others as EMS

Notes:
- 5mer features use canonicalization to center C; G-centered sites are reverse-complemented
- Poisson GLM: y = ems_count, offset = log(depth); includes Treatment covariate (0/1)
- Strict canonical EMS counting: at C sites count T only; at G sites count A only
"""

# Set thread limits BEFORE importing numpy/statsmodels
import os
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32' 
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['BLAS_NUM_THREADS'] = '32'
os.environ['LAPACK_NUM_THREADS'] = '32'

import argparse
import glob
import gzip
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle


def canonical_kmer(kmer: str) -> str:
    if len(kmer) % 2 == 0:
        return kmer
    center = len(kmer) // 2
    if kmer[center] == 'C':
        return kmer
    if kmer[center] == 'G':
        return str(Seq(kmer).reverse_complement())
    return kmer


def load_exclusion_mask_tsv(mask_file: str):
    excluded = set()
    if not mask_file:
        return excluded
    with open(mask_file) as f:
        header = next(f, None)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            excluded.add((parts[0], parts[1]))
    return excluded


def load_genome_sequences(genome_fasta: str):
    seqs = {}
    current = None
    def _read(handle):
        nonlocal current
        for raw in handle:
            line = raw.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                current = line[1:].split()[0]
                seqs[current] = []
            else:
                if current is not None:
                    seqs[current].append(line.upper())
    if genome_fasta.endswith('.gz'):
        with gzip.open(genome_fasta, 'rt') as fh:
            _read(fh)
    else:
        with open(genome_fasta, 'r') as fh:
            _read(fh)
    for k in list(seqs.keys()):
        seqs[k] = ''.join(seqs[k])
    return seqs


def extract_kmer(seqs, chrom: str, pos_1based: int, k: int):
    if chrom not in seqs:
        return None
    s = seqs[chrom]
    i0 = int(pos_1based) - 1
    half = (k - 1) // 2
    start = i0 - half
    end = i0 + half + 1
    if start < 0 or end > len(s):
        return None
    kmer = s[start:end].upper()
    if len(kmer) != k or 'N' in kmer:
        return None
    return kmer


def load_site_level_data(counts_dir: str, genome_fasta: str, exclusion_mask: str = None):
    mask = load_exclusion_mask_tsv(exclusion_mask)
    seqs = load_genome_sequences(genome_fasta)

    rows = []
    files = glob.glob(os.path.join(counts_dir, '*.counts'))
    for path in files:
        sample_name = os.path.basename(path)
        is_control = ('NT' in sample_name)
        treatment = 0 if is_control else 1
        with open(path) as f:
            header = next(f, None)
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 9:
                    continue
                chrom, pos, ref, ref_count, A_count, C_count, G_count, T_count, depth = parts[:9]
                if (chrom, pos) in mask:
                    continue
                if ref not in {'G', 'C'}:
                    continue
                try:
                    ref_i = int(ref_count)
                    a_i = int(A_count)
                    c_i = int(C_count)
                    g_i = int(G_count)
                    t_i = int(T_count)
                    depth_i = int(depth)
                except Exception:
                    continue
                if depth_i <= 0:
                    continue
                # Strict canonical EMS counting
                if ref == 'C':
                    ems_count = t_i  # C->T
                else:  # ref == 'G'
                    ems_count = a_i  # G->A
                # extract 5mer and 7mer
                k5 = extract_kmer(seqs, chrom, int(pos), 5)
                if not k5:
                    continue
                if k5[2] not in ['C', 'G']:
                    continue
                k5 = canonical_kmer(k5)
                k7 = extract_kmer(seqs, chrom, int(pos), 7)
                k7 = canonical_kmer(k7) if k7 and k7[3] in ['C', 'G'] else None
                rows.append({
                    'chrom': chrom,
                    'pos': pos,
                    'treatment': treatment,
                    'depth': depth_i,
                    'ems_count': ems_count,
                    'kmer5': k5,
                    'kmer7': k7,
                })
    df = pd.DataFrame(rows)
    return df


def add_positional_features(df: pd.DataFrame) -> list:
    cols = []
    for pos in range(5):
        for base in ['T', 'G', 'C']:
            col = f'pos{pos}_{base}'
            df[col] = df['kmer5'].apply(lambda k, p=pos, b=base: 1 if k and len(k) == 5 and k[p] == b else 0)
            cols.append(col)
    return cols


def add_3mer_features(df: pd.DataFrame) -> list:
    bases = ['A', 'T', 'G', 'C']
    # 3mer: positions 1, 2, 3 of 5mer (positions -1, 0, +1 relative to mutation)
    # Position 2 (center) is always C, so only 4×1×4 = 16 possible 3mers
    all3 = [a + 'C' + b for a in bases for b in bases]
    cols = []
    for t in all3:
        col = f'3mer_{t}'
        df[col] = df['kmer5'].apply(lambda k, t=t: 1 if k and len(k) == 5 and k[1:4] == t else 0)
        cols.append(col)
    return cols


def add_pos3mer_features(df: pd.DataFrame) -> list:
    bases = ['A', 'T', 'G', 'C']
    # Left 3mer: positions 0, 1, 2 of 5mer (positions -2, -1, 0 relative to mutation)
    # Position 2 (center) is always C, so only 4×4×1 = 16 possible left 3mers
    left_3mers = [a + b + 'C' for a in bases for b in bases]
    # Right 3mer: positions 2, 3, 4 of 5mer (positions 0, +1, +2 relative to mutation)
    # Position 2 (center) is always C, so only 1×4×4 = 16 possible right 3mers
    right_3mers = ['C' + a + b for a in bases for b in bases]
    cols = []
    for t in left_3mers:
        col = f'l3_{t}'
        df[col] = df['kmer5'].apply(lambda k, t=t: 1 if k and len(k) == 5 and k[0:3] == t else 0)
        cols.append(col)
    for t in right_3mers:
        col = f'r3_{t}'
        df[col] = df['kmer5'].apply(lambda k, t=t: 1 if k and len(k) == 5 and k[2:5] == t else 0)
        cols.append(col)
    return cols


def add_5mer_features(df: pd.DataFrame) -> list:
    # Build all 5mer one-hot columns at once to avoid fragmentation
    dummies = pd.get_dummies(df['kmer5'], prefix='5mer', dtype='uint8')
    df[dummies.columns] = dummies
    return list(dummies.columns)


def _split_list_round_robin(items: list, n_splits: int) -> list:
    groups = [[] for _ in range(max(1, n_splits))]
    for i, it in enumerate(sorted(items)):
        groups[i % len(groups)].append(it)
    return groups


def fit_7mer_model_splits(df: pd.DataFrame, glm_family: str, nb_alpha: float | None, n_splits: int = 8, return_models: bool = False):
    # Use only rows with canonical 7mer available
    if 'kmer7' not in df.columns:
        return None
    df7 = df[df['kmer7'].notna()].copy()
    if df7.empty:
        return None
    if 'log_depth' not in df7.columns:
        df7['log_depth'] = np.log(df7['depth'])

    unique7 = sorted(df7['kmer7'].dropna().unique())
    if not unique7:
        return None

    groups = _split_list_round_robin(unique7, n_splits)

    total_llf = 0.0
    total_params = 0
    total_n = 0
    split_models = []  # Store models if requested

    for group in groups:
        if not group:
            continue
        sub = df7[df7['kmer7'].isin(group)].copy()
        if sub.empty:
            continue
        # Build split-local one-hots for 7mer in one shot
        dummies = pd.get_dummies(sub['kmer7'], prefix='7mer', dtype='uint8')
        # Keep only current group's columns (prefix matches)
        want_cols = [f'7mer_{k}' for k in group]
        missing = [c for c in want_cols if c not in dummies.columns]
        if missing:
            # Add missing columns as zeros to keep design consistent
            for c in missing:
                dummies[c] = 0
        dummies = dummies[want_cols]
        sub[dummies.columns] = dummies
        X = sub[['treatment'] + list(dummies.columns)]
        y = sub['ems_count'].values
        offset = sub['log_depth'].values

        try:
            result = fit_glm(y, X, offset, family=glm_family, nb_alpha=nb_alpha)
        except Exception:
            continue

        llf = float(getattr(result, 'llf', np.nan))
        k_params = int(len(result.params)) if hasattr(result, 'params') else (1 + dummies.shape[1])
        n_obs = int(len(sub))
        if np.isfinite(llf):
            total_llf += llf
            total_params += k_params
            total_n += n_obs
            if return_models:
                # Store model with metadata about which kmers it covers
                split_models.append({
                    'model': result,
                    'group': group,
                    'want_cols': want_cols,
                    'glm_family': glm_family,
                    'nb_alpha': nb_alpha
                })

    if total_n == 0:
        return None

    aic = 2 * total_params - 2 * total_llf
    bic = total_params * np.log(total_n) - 2 * total_llf

    result_dict = {
        'aic': aic,
        'bic': bic,
        'llf': total_llf,
        'n_params': total_params,
        'n_obs': total_n,
        'n_splits': n_splits,
    }
    
    if return_models:
        result_dict['split_models'] = split_models
    
    return result_dict

def fit_glm(y, X, offset, family: str = 'poisson', nb_alpha: float | None = None):
    design = sm.add_constant(X, has_constant='add')
    if family == 'negative_binomial':
        # Poisson initializer for speed and stability
        pois_model = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset)
        pois_res = pois_model.fit()

        # Build start vector aligned to design columns
        try:
            design_cols = list(design.columns)
            start_beta = pois_res.params.reindex(design_cols, fill_value=0.0).values
        except Exception:
            # Fallback: ensure length matches number of columns
            p = design.shape[1]
            start_beta = np.zeros(p, dtype=float)
            pr = np.asarray(pois_res.params)
            start_beta[:min(len(pr), p)] = pr[:min(len(pr), p)]

        # If user provides alpha, use GLM NB with that alpha; else estimate alpha via discrete NB
        if nb_alpha is not None:
            fam = sm.families.NegativeBinomial(alpha=nb_alpha)
            model = sm.GLM(y, design, family=fam, offset=offset)
            result = model.fit(start_params=start_beta)
            return result
        else:
            exposure = np.exp(offset)
            # Discrete NB estimates dispersion (alpha) from data; warm-start with Poisson params
            model = NBDiscrete(y, design, exposure=exposure)
            # Pad with ancillary dispersion start (at end) so len >= exog cols
            start_params_nb = np.r_[start_beta, 0.1]
            result = model.fit(start_params=start_params_nb, disp=False)
            return result
    else:
        fam = sm.families.Poisson()
        model = sm.GLM(y, design, family=fam, offset=offset).fit()
        return model


def validate_model_metrics(model, model_name: str, n_obs: int, feature_cols: list = None):
    """
    Validate that AIC and BIC are calculated correctly.
    
    AIC = 2*k - 2*llf
    BIC = k*log(n) - 2*llf
    where k = number of parameters, n = number of observations, llf = log-likelihood
    """
    k = len(model.params)
    llf = model.llf
    aic_reported = model.aic
    bic_reported = model.bic
    
    # Calculate expected AIC and BIC
    aic_expected = 2 * k - 2 * llf
    bic_expected = k * np.log(n_obs) - 2 * llf
    
    # Check if they match (within numerical precision)
    aic_match = np.isclose(aic_reported, aic_expected, rtol=1e-10)
    bic_match = np.isclose(bic_reported, bic_expected, rtol=1e-10)
    
    if not aic_match or not bic_match:
        print(f"  WARNING: {model_name} metric validation failed!")
        if not aic_match:
            print(f"    AIC: reported={aic_reported:.6f}, expected={aic_expected:.6f}, diff={abs(aic_reported - aic_expected):.2e}")
        if not bic_match:
            print(f"    BIC: reported={bic_reported:.6f}, expected={bic_expected:.6f}, diff={abs(bic_reported - bic_expected):.2e}")
    else:
        # Build status message
        status_msg = f"  ✓ {model_name}: {k} parameters, AIC={aic_reported:.2f}, BIC={bic_reported:.2f}"
        if feature_cols is not None:
            expected_params = 1 + 1 + len(feature_cols)  # intercept + treatment + features
            if k == expected_params:
                status_msg += f" (1 intercept + 1 treatment + {len(feature_cols)} features)"
            else:
                status_msg += f" (expected {expected_params} = 1 intercept + 1 treatment + {len(feature_cols)} features, got {k})"
        print(status_msg)
    
    return {
        'n_params': k,
        'aic': aic_reported,
        'bic': bic_reported,
        'llf': llf,
        'aic_valid': aic_match,
        'bic_valid': bic_match
    }


def compare_models_sitelevel(df: pd.DataFrame, output_dir: str, glm_family: str = 'poisson', nb_alpha: float | None = None, sevenmer_splits: int = 8, force_refit: bool = False):
    # Common pieces
    df = df[df['depth'] > 0].copy()
    df['log_depth'] = np.log(df['depth'])

    results = {}
    models = {}
    
    # Check for existing models and summary
    models_dir = os.path.join(output_dir, 'fitted_models')
    summary_path = os.path.join(output_dir, 'sequence_bias_sitelevel_summary.json')
    existing_summary = None
    
    if os.path.exists(summary_path) and not force_refit:
        try:
            with open(summary_path, 'r') as f:
                existing_summary = json.load(f)
            print(f"Found existing summary JSON at {summary_path}")
        except Exception as e:
            print(f"Warning: Could not load existing summary: {e}")
    
    # Try to load existing models
    if os.path.exists(models_dir) and not force_refit:
        try:
            from load_saved_models import load_saved_models
            existing_models, existing_metadata = load_saved_models(models_dir)
            if existing_models:
                print(f"Found {len(existing_models)} existing model(s) in {models_dir}")
        except Exception as e:
            print(f"Warning: Could not load existing models: {e}")
            existing_models = {}
            existing_metadata = {}
    else:
        existing_models = {}
        existing_metadata = {}

    # Base covariate: treatment
    base_cov = ['treatment']

    # Positional model
    if 'positional' in existing_models and not force_refit:
        print(f"  Using existing positional model")
        models['positional'] = existing_models['positional']
        if existing_summary and 'results' in existing_summary and 'positional' in existing_summary['results']:
            results['positional'] = existing_summary['results']['positional']
        else:
            # Compute metrics from loaded model (may be slow but ensures accuracy)
            pos_cols = add_positional_features(df)
            X_pos = df[base_cov + pos_cols]
            # Validate metrics for loaded model
            validate_model_metrics(models['positional'], 'positional (loaded)', len(df), pos_cols)
            results['positional'] = {
                'aic': models['positional'].aic,
                'bic': models['positional'].bic,
                'llf': models['positional'].llf,
                'n_params': len(models['positional'].params)
            }
    else:
        print(f"  Fitting positional model...")
        pos_cols = add_positional_features(df)
        X_pos = df[base_cov + pos_cols]
        model_pos = fit_glm(df['ems_count'].values, X_pos, df['log_depth'].values, family=glm_family, nb_alpha=nb_alpha)
        # Validate metrics
        validate_model_metrics(model_pos, 'positional', len(df), pos_cols)
        results['positional'] = {'aic': model_pos.aic, 'bic': model_pos.bic, 'llf': model_pos.llf, 'n_params': len(model_pos.params)}
        models['positional'] = model_pos

    # 3mer model
    if '3mer' in existing_models and not force_refit:
        print(f"  Using existing 3mer model")
        models['3mer'] = existing_models['3mer']
        # Always compute feature columns for metadata (even if model was loaded)
        tri_cols = add_3mer_features(df)
        if existing_summary and 'results' in existing_summary and '3mer' in existing_summary['results']:
            results['3mer'] = existing_summary['results']['3mer']
        else:
            # Validate metrics for loaded model
            validate_model_metrics(models['3mer'], '3mer (loaded)', len(df), tri_cols)
            results['3mer'] = {
                'aic': models['3mer'].aic,
                'bic': models['3mer'].bic,
                'llf': models['3mer'].llf,
                'n_params': len(models['3mer'].params)
            }
    else:
        print(f"  Fitting 3mer model...")
        tri_cols = add_3mer_features(df)
        X_tri = df[base_cov + tri_cols]
        model_tri = fit_glm(df['ems_count'].values, X_tri, df['log_depth'].values, family=glm_family, nb_alpha=nb_alpha)
        # Validate metrics
        validate_model_metrics(model_tri, '3mer', len(df), tri_cols)
        results['3mer'] = {'aic': model_tri.aic, 'bic': model_tri.bic, 'llf': model_tri.llf, 'n_params': len(model_tri.params)}
        models['3mer'] = model_tri

    # 5mer model
    if '5mer' in existing_models and not force_refit:
        print(f"  Using existing 5mer model")
        models['5mer'] = existing_models['5mer']
        # Always compute feature columns for metadata (even if model was loaded)
        k5_cols = add_5mer_features(df)
        if existing_summary and 'results' in existing_summary and '5mer' in existing_summary['results']:
            results['5mer'] = existing_summary['results']['5mer']
        else:
            # Validate metrics for loaded model
            validate_model_metrics(models['5mer'], '5mer (loaded)', len(df), k5_cols)
            results['5mer'] = {
                'aic': models['5mer'].aic,
                'bic': models['5mer'].bic,
                'llf': models['5mer'].llf,
                'n_params': len(models['5mer'].params)
            }
    else:
        print(f"  Fitting 5mer model...")
        k5_cols = add_5mer_features(df)
        X_k5 = df[base_cov + k5_cols]
        model_k5 = fit_glm(df['ems_count'].values, X_k5, df['log_depth'].values, family=glm_family, nb_alpha=nb_alpha)
        # Validate metrics
        validate_model_metrics(model_k5, '5mer', len(df), k5_cols)
        results['5mer'] = {'aic': model_k5.aic, 'bic': model_k5.bic, 'llf': model_k5.llf, 'n_params': len(model_k5.params)}
        models['5mer'] = model_k5

    # positional 3mer
    if 'pos3mer' in existing_models and not force_refit:
        print(f"  Using existing pos3mer model")
        models['pos3mer'] = existing_models['pos3mer']
        # Always compute feature columns for metadata (even if model was loaded)
        p3_cols = add_pos3mer_features(df)
        if existing_summary and 'results' in existing_summary and 'pos3mer' in existing_summary['results']:
            results['pos3mer'] = existing_summary['results']['pos3mer']
        else:
            # Validate metrics for loaded model
            validate_model_metrics(models['pos3mer'], 'pos3mer (loaded)', len(df), p3_cols)
            results['pos3mer'] = {
                'aic': models['pos3mer'].aic,
                'bic': models['pos3mer'].bic,
                'llf': models['pos3mer'].llf,
                'n_params': len(models['pos3mer'].params)
            }
    else:
        print(f"  Fitting pos3mer model...")
        p3_cols = add_pos3mer_features(df)
        X_p3 = df[base_cov + p3_cols]
        model_p3 = fit_glm(df['ems_count'].values, X_p3, df['log_depth'].values, family=glm_family, nb_alpha=nb_alpha)
        # Validate metrics
        validate_model_metrics(model_p3, 'pos3mer', len(df), p3_cols)
        results['pos3mer'] = {'aic': model_p3.aic, 'bic': model_p3.bic, 'llf': model_p3.llf, 'n_params': len(model_p3.params)}
        models['pos3mer'] = model_p3

    # 7mer model via split-fitting to control memory
    if '7mer_split' in existing_models and not force_refit:
        print(f"  Using existing 7mer_split model")
        models['7mer_split'] = existing_models['7mer_split']
        if existing_summary and 'results' in existing_summary and '7mer_split' in existing_summary['results']:
            results['7mer_split'] = existing_summary['results']['7mer_split']
        else:
            # Can't easily recompute 7mer metrics from loaded models without refitting
            # So just use summary if available, otherwise warn
            print(f"    Warning: Could not load 7mer_split metrics from summary. Metrics may be missing.")
            results['7mer_split'] = {'aic': np.nan, 'bic': np.nan, 'llf': np.nan, 'n_params': np.nan}
    else:
        print(f"  Fitting 7mer_split model...")
        res7 = fit_7mer_model_splits(df, glm_family, nb_alpha, n_splits=sevenmer_splits, return_models=True)
        if res7 is not None:
            results['7mer_split'] = {k: v for k, v in res7.items() if k != 'split_models'}
            # Store split models as a list for saving
            if 'split_models' in res7:
                models['7mer_split'] = res7['split_models']
        else:
            print(f"    Warning: 7mer_split model fitting failed (fit_7mer_model_splits returned None)")
            results['7mer_split'] = {'aic': np.nan, 'bic': np.nan, 'llf': np.nan, 'n_params': 0}

    # Ensure we have positional features for plotting (even if model was loaded)
    if 'positional' in models:
        if 'positional' not in existing_models or force_refit:
            # Features already added during fitting
            pos_cols = add_positional_features(df)
        else:
            # Need to add features for plotting
            pos_cols = add_positional_features(df)
    
    # Plots (retain originals for compatibility) and compute improved arrays for reuse
    names = list(results.keys())
    # Filter out models with NaN AIC (e.g., loaded 7mer without summary)
    valid_names = [n for n in names if not np.isnan(results[n].get('aic', np.nan))]
    if not valid_names:
        print("Warning: No valid models with metrics found for plotting")
        valid_names = names  # Use all names anyway
    
    aic = [results[n].get('aic', np.nan) for n in valid_names]
    bic = [results[n].get('bic', np.nan) for n in valid_names]
    npar = [results[n].get('n_params', 0) for n in valid_names]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Site-level Sequence Bias Model Comparison (Poisson GLM with log(depth) offset)', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.bar(valid_names, aic)
    ax1.set_title('AIC (lower is better)')
    ax1.set_ylabel('AIC')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[0, 1]
    ax2.bar(valid_names, bic)
    ax2.set_title('BIC (lower is better)')
    ax2.set_ylabel('BIC')
    ax2.tick_params(axis='x', rotation=45)

    ax3 = axes[1, 0]
    ax3.bar(valid_names, npar)
    ax3.set_title('Number of Parameters')
    ax3.set_yscale('log')
    ax3.tick_params(axis='x', rotation=45)

    valid_aic = [a for a in aic if not np.isnan(a)]
    if valid_aic:
        min_aic = min(valid_aic)
        delta = [v - min_aic if not np.isnan(v) else np.nan for v in aic]
    else:
        delta = [np.nan] * len(aic)
    ax4 = axes[1, 1]
    ax4.bar(valid_names, delta)
    ax4.set_title('ΔAIC (relative to best)')
    ax4.set_ylabel('ΔAIC')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sequence_bias_model_comparison_sitelevel.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sequence_bias_model_comparison_sitelevel.pdf'), bbox_inches='tight')
    plt.close()

    # Positional heatmap (original)
    if 'positional' in models:
        coef_series = models['positional'].params
        pos_coefs = {k: v for k, v in coef_series.items() if isinstance(k, str) and k.startswith('pos')}
        mat = np.zeros((5, 4))
        for p in range(5):
            for bi, base in enumerate(['A', 'T', 'G', 'C']):
                if base == 'A':
                    continue
                name = f'pos{p}_{base}'
                if name in pos_coefs:
                    mat[p, bi] = pos_coefs[name]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(mat, annot=True, fmt='.4f', cmap='RdBu_r', center=0,
                    xticklabels=['A', 'T', 'G', 'C'], yticklabels=[f'Pos {i}' for i in range(5)],
                    cbar_kws={'label': 'Log enrichment (vs A at pos)'}, ax=ax)
        ax.set_title('Positional parameters (site-level, Poisson offset log(depth))')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sequence_bias_positional_parameters_sitelevel.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'sequence_bias_positional_parameters_sitelevel.pdf'), bbox_inches='tight')
        plt.close()

    # Report
    report = os.path.join(output_dir, 'sequence_bias_modeling_sitelevel_report.txt')
    with open(report, 'w') as f:
        f.write('=== Site-level Sequence Bias Modeling Report (Poisson GLM with log(depth) offset) ===\n\n')
        f.write(f"Total sites: {len(df)}\n")
        f.write(f"Treated rows: {(df['treatment']==1).sum()}, Control rows: {(df['treatment']==0).sum()}\n\n")
        f.write(f"{'Model':<12} {'AIC':<12} {'BIC':<12} {'LogLik':<12} {'Params':<8}\n")
        f.write('-'*70+'\n')
        for n in names:
            r = results[n]
            aic_val = r.get('aic', np.nan)
            bic_val = r.get('bic', np.nan)
            llf_val = r.get('llf', np.nan)
            n_params_val = r.get('n_params', 0)
            if np.isnan(aic_val):
                f.write(f"{n:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {n_params_val:<8}\n")
            else:
                f.write(f"{n:<12} {aic_val:<12.2f} {bic_val:<12.2f} {llf_val:<12.2f} {n_params_val:<8}\n")
        valid_results = {n: r for n, r in results.items() if not np.isnan(r.get('aic', np.nan))}
        if valid_results:
            best = min(valid_results.keys(), key=lambda x: valid_results[x]['aic'])
            f.write(f"\nBest by AIC: {best} (AIC={valid_results[best]['aic']:.2f})\n")

    # Save machine-readable summary for downstream plotting without refitting
    summary = {
        'df_len': int(len(df)),
        'glm_family': glm_family,
        'results': {}
    }
    
    # Add results, handling NaN values
    for k, v in results.items():
        aic_val = v.get('aic', np.nan)
        bic_val = v.get('bic', np.nan)
        llf_val = v.get('llf', np.nan)
        n_params_val = v.get('n_params', 0)
        summary['results'][k] = {
            'aic': float(aic_val) if not np.isnan(aic_val) else None,
            'bic': float(bic_val) if not np.isnan(bic_val) else None,
            'llf': float(llf_val) if not np.isnan(llf_val) else None,
            'n_params': int(n_params_val) if not np.isnan(n_params_val) else 0
        }
    
    # Add positional params if positional model exists
    if 'positional' in models:
        coef_series = models['positional'].params
        summary['positional_params'] = {str(k): float(val) for k, val in coef_series.items() if isinstance(k, str)}
    
    with open(os.path.join(output_dir, 'sequence_bias_sitelevel_summary.json'), 'w') as fh:
        json.dump(summary, fh)
    
    # Save fitted models for downstream evaluation (residual analysis, prediction metrics)
    models_dir = os.path.join(output_dir, 'fitted_models')
    os.makedirs(models_dir, exist_ok=True)
    for model_name, model_obj in models.items():
        if model_obj is not None:
            model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
            try:
                # Special handling for 7mer_split which is a list of split models
                if model_name == '7mer_split' and isinstance(model_obj, list):
                    # Save as a list of split models
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_obj, f)
                    print(f"Saved {model_name} model ({len(model_obj)} split models)")
                else:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_obj, f)
            except Exception as e:
                print(f"Warning: Could not save {model_name} model: {e}")
    
    # Save model metadata (which features were used)
    model_metadata = {}
    if 'positional' in models:
        model_metadata['positional'] = {'feature_cols': pos_cols}
    if '3mer' in models:
        model_metadata['3mer'] = {'feature_cols': tri_cols}
    if '5mer' in models:
        model_metadata['5mer'] = {'feature_cols': k5_cols}
    if 'pos3mer' in models:
        model_metadata['pos3mer'] = {'feature_cols': p3_cols}
    if '7mer_split' in models:
        # Store info about 7mer split structure
        if isinstance(models['7mer_split'], list) and len(models['7mer_split']) > 0:
            all_kmers = []
            for split_model in models['7mer_split']:
                if 'group' in split_model:
                    all_kmers.extend(split_model['group'])
            model_metadata['7mer_split'] = {
                'n_splits': len(models['7mer_split']),
                'total_kmers': len(set(all_kmers)),
                'glm_family': models['7mer_split'][0].get('glm_family', glm_family),
                'nb_alpha': models['7mer_split'][0].get('nb_alpha', None)
            }
    
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)

    return results, models


def plot_model_comparison_better(results: dict, df_len: int, output_dir: str):
    names = list(results.keys())
    aic = np.array([results[n]['aic'] for n in names], dtype=float)
    bic = np.array([results[n]['bic'] for n in names], dtype=float)
    npar = np.array([results[n]['n_params'] for n in names], dtype=float)

    aic_per_site = aic / max(1, df_len)
    min_aic = np.nanmin(aic)
    delta_aic = aic - min_aic

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Site-level Sequence Bias Model Comparison (improved)', fontsize=18, fontweight='bold')

    ax = axes[0, 0]
    axes[0, 0].bar(names, aic, color='#4C78A8')
    ax.set_title('AIC (absolute)')
    ax.set_ylabel('AIC')
    ax.set_xlabel('Model')

    ax = axes[0, 1]
    axes[0, 1].bar(names, bic, color='#F58518')
    ax.set_title('BIC (absolute)')
    ax.set_ylabel('BIC')
    ax.set_xlabel('Model')

    ax = axes[1, 0]
    axes[1, 0].bar(names, npar, color='#54A24B')
    ax.set_title('Number of parameters (log scale)')
    ax.set_yscale('log')
    ax.set_ylabel('Parameters')
    ax.set_xlabel('Model')

    ax = axes[1, 1]
    x = np.arange(len(names))
    width = 0.45
    ax.bar(x - width/2, delta_aic, width=width, label='ΔAIC', color='#4C78A8')
    ax.bar(x + width/2, aic_per_site, width=width, label='AIC per site', color='#F58518')
    ax.set_title('ΔAIC (vs best) and AIC per site')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Score')
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sequence_bias_model_comparison_improved.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'sequence_bias_model_comparison_improved.pdf'))
    plt.close()


def plot_positional_heatmap_better(positional_model_or_params, output_dir: str):
    # Accept either a statsmodels result (with .params) or a dict of params
    if hasattr(positional_model_or_params, 'params'):
        coef_series = positional_model_or_params.params
        raw_items = coef_series.items()
    else:
        raw_items = positional_model_or_params.items()
    pos_coefs = {k: v for k, v in raw_items if isinstance(k, str) and k.startswith('pos')}
    positions = 5
    bases = ['A', 'T', 'G', 'C']
    mat = np.zeros((positions, len(bases)))
    for p in range(positions):
        for bi, base in enumerate(bases):
            if base == 'A':
                mat[p, bi] = 0.0
                continue
            name = f'pos{p}_{base}'
            if name in pos_coefs:
                mat[p, bi] = pos_coefs[name]

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        mat,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Log enrichment vs A'},
        ax=ax
    )
    ax.set_xticklabels(bases, rotation=0)
    ax.set_yticklabels([f'Pos {i}' for i in range(positions)], rotation=0)
    ax.set_title('Positional parameters (log-scale, A as baseline)', fontsize=16)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sequence_bias_positional_parameters_improved.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'sequence_bias_positional_parameters_improved.pdf'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Site-level sequence bias modeling (rate-based)')
    parser.add_argument('--counts-dir', required=True, help='Directory with .counts files')
    parser.add_argument('--genome-fasta', required=True, help='Reference genome FASTA (supports .gz)')
    parser.add_argument('--exclusion-mask', help='TSV with chrom\tpos to exclude')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--glm-family', choices=['poisson', 'negative_binomial'], default='poisson',
                        help='GLM family to use for site-level models (default: poisson)')
    parser.add_argument('--nb-alpha', type=float, default=None,
                        help='If set with --glm-family negative_binomial, use this fixed dispersion alpha; otherwise alpha is estimated')
    parser.add_argument('--nb-alpha-from-controls', action='store_true',
                        help='Estimate NB alpha from NT controls (intercept-only Poisson warm start) and use it')
    parser.add_argument('--sevenmer-splits', type=int, default=8,
                        help='Number of groups to split 7mer categories for memory control (default: 8)')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation scripts (residual analysis, prediction metrics)')
    parser.add_argument('--run-cross-validation', action='store_true',
                        help='Run cross-validation evaluation (takes longer, refits models)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--force-refit', action='store_true',
                        help='Force refitting all models even if .pkl files exist (default: load existing models)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print('Loading site-level data...')
    df = load_site_level_data(args.counts_dir, args.genome_fasta, args.exclusion_mask)
    if df is None or len(df) == 0:
        print('No data found')
        return
    print(f'Loaded {len(df):,} site rows')

    # Optional: estimate NB alpha from controls
    nb_alpha_use = args.nb_alpha
    if args.glm_family == 'negative_binomial' and args.nb_alpha is None and args.nb_alpha_from_controls:
        df_ctrl = df[df['treatment'] == 0].copy()
        if not df_ctrl.empty:
            try:
                if 'log_depth' not in df_ctrl.columns:
                    df_ctrl['log_depth'] = np.log(df_ctrl['depth'])
                y = df_ctrl['ems_count'].values
                design = sm.add_constant(pd.DataFrame(index=df_ctrl.index), has_constant='add')
                pois = sm.GLM(y, design, family=sm.families.Poisson(), offset=df_ctrl['log_depth'].values).fit()
                mu = pois.fittedvalues
                var_y = np.var(y); mean_mu = np.mean(mu)
                if var_y > mean_mu and mean_mu > 0:
                    nb_alpha_use = float((mean_mu ** 2) / max(var_y - mean_mu, 1e-12))
                    print(f"Estimated NB alpha from controls: {nb_alpha_use:.4f}")
            except Exception as e:
                print(f"Warning: NB alpha estimation from controls failed: {e}")

    print(f"Fitting and comparing models (glm_family={args.glm_family}, nb_alpha={nb_alpha_use})...")
    if args.force_refit:
        print("  Force refitting enabled - will refit all models even if .pkl files exist")
    results, models = compare_models_sitelevel(df, args.output_dir, glm_family=args.glm_family, nb_alpha=nb_alpha_use, sevenmer_splits=args.sevenmer_splits, force_refit=args.force_refit)
    print('Model fitting complete.')
    
    # Run evaluation scripts
    if not args.skip_evaluation:
        print("\n" + "="*80)
        print("RUNNING EVALUATION SCRIPTS")
        print("="*80)
        
        # 1. Residual analysis
        print("\n[1/3] Running residual analysis...")
        try:
            from residual_analysis import analyze_all_models
            from load_saved_models import load_models_and_prepare_data
            
            # Load models and metadata from just-saved files
            result = load_models_and_prepare_data(args.output_dir, df)
            eval_models = result['models']
            eval_metadata = result['metadata']
            
            residual_results = analyze_all_models(
                eval_models, df, args.output_dir, 
                glm_family=args.glm_family, 
                metadata=eval_metadata
            )
            print("✓ Residual analysis complete")
        except Exception as e:
            print(f"⚠ Warning: Residual analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Prediction accuracy metrics (on full dataset)
        print("\n[2/3] Computing prediction accuracy metrics...")
        try:
            from prediction_accuracy_metrics import compare_models_predictions, print_metrics_summary
            
            # Use all data for evaluation (models already fit)
            eval_dir = os.path.join(args.output_dir, 'prediction_evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            
            # Check for CV results if cross-validation was run
            cv_results_path = None
            if args.run_cross_validation:
                cv_results_path = os.path.join(args.output_dir, 'cross_validation', 'cv_results_summary.json')
                if not os.path.exists(cv_results_path):
                    cv_results_path = None
            
            test_metrics = compare_models_predictions(
                eval_models, df, eval_dir, eval_metadata, cv_results_path=cv_results_path
            )
            
            if test_metrics is not None:
                print_metrics_summary(test_metrics)
                print("✓ Prediction accuracy metrics complete")
            else:
                print("⚠ Warning: No prediction metrics computed")
        except Exception as e:
            print(f"⚠ Warning: Prediction accuracy metrics failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. Cross-validation (optional, refits models)
        if args.run_cross_validation:
            print("\n[3/3] Running cross-validation (this will take longer)...")
            try:
                from cross_validation_evaluation import cross_validate_all_models
                
                cv_dir = os.path.join(args.output_dir, 'cross_validation')
                os.makedirs(cv_dir, exist_ok=True)
                
                cv_results = cross_validate_all_models(
                    df, args.glm_family, nb_alpha_use, 
                    args.cv_folds, cv_dir, sevenmer_splits=args.sevenmer_splits
                )
                
                print("\nCross-Validation Results Summary:")
                print("-" * 60)
                for model_name, metrics in cv_results.items():
                    if metrics:
                        print(f"\n{model_name}:")
                        print(f"  MSE: {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}")
                        print(f"  MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
                        print(f"  Pearson r: {metrics['pearson_r_mean']:.4f} ± {metrics['pearson_r_std']:.4f}")
                print("✓ Cross-validation complete")
                
                # Regenerate combined 8-panel plot now that CV results exist
                cv_results_path = os.path.join(cv_dir, 'cv_results_summary.json')
                if os.path.exists(cv_results_path):
                    try:
                        from prediction_accuracy_metrics import plot_combined_model_evaluation
                        import pandas as pd
                        
                        # Load prediction metrics
                        eval_dir = os.path.join(args.output_dir, 'prediction_evaluation')
                        metrics_tsv = os.path.join(eval_dir, 'prediction_accuracy_metrics.tsv')
                        if os.path.exists(metrics_tsv):
                            print("\nRegenerating 8-panel combined plot with CV results...")
                            metrics_df = pd.read_csv(metrics_tsv, sep='\t')
                            
                            # Save to final_figs directory if it exists, otherwise to eval_dir
                            final_figs_dir = os.path.join(args.output_dir, 'final_figs')
                            if os.path.exists(final_figs_dir):
                                plot_output_dir = final_figs_dir
                            else:
                                os.makedirs(final_figs_dir, exist_ok=True)
                                plot_output_dir = final_figs_dir
                            
                            plot_combined_model_evaluation(metrics_df, cv_results_path, plot_output_dir)
                            print(f"✓ Combined plot saved to {plot_output_dir}/combined_model_evaluation.png")
                    except Exception as e2:
                        print(f"⚠ Warning: Failed to regenerate combined plot: {e2}")
            except Exception as e:
                print(f"⚠ Warning: Cross-validation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n[3/3] Skipping cross-validation (use --run-cross-validation to enable)")
        
        print("\n" + "="*80)
        print("ALL EVALUATIONS COMPLETE")
        print("="*80)
        print(f"\nOutput directory: {args.output_dir}")
        print("  - Fitted models: fitted_models/")
        print("  - Residual diagnostics: residual_diagnostics_*.png")
        print("  - Prediction metrics: prediction_evaluation/")
        if args.run_cross_validation:
            print("  - Cross-validation results: cross_validation/")
    
    print('\nDone.')


if __name__ == '__main__':
    main()


