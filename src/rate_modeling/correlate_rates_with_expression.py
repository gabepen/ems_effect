#!/usr/bin/env python3
"""
Correlation analysis between gene mutation rates and TPM expression values.

Performs linear regression analysis to test the hypothesis that higher expression
leads to higher mutation rates (transcription-coupled mutations).
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def extract_gene_id(gene_id_str):
    """
    Extract WD_RS ID from gene_id column.
    Handles formats like "gene-WD_RS00015" -> "WD_RS00015"
    """
    if pd.isna(gene_id_str):
        return None
    gene_id_str = str(gene_id_str)
    if gene_id_str.startswith('gene-'):
        return gene_id_str[5:]  # Remove "gene-" prefix
    return gene_id_str


def load_data(mutation_rates_file, expression_file, expression_metric='log_tpm'):
    """
    Load and merge mutation rates and expression data.
    
    Parameters:
    -----------
    expression_metric : str
        Which expression metric to use:
        - 'tpm': Raw TPM values
        - 'log_tpm': log(TPM + 1) - recommended for skewed data
        - 'counts': Raw estimated counts
        - 'log_counts': log(counts + 1)
    """
    print("Loading mutation rates data...")
    rates_df = pd.read_csv(mutation_rates_file, sep='\t')
    print(f"  Loaded {len(rates_df)} genes from mutation rates file")
    
    print("Loading expression data...")
    expr_df = pd.read_csv(expression_file, sep='\t')
    print(f"  Loaded {len(expr_df)} genes from expression file")
    
    # Extract gene IDs for matching
    rates_df['gene_id_clean'] = rates_df['gene_id'].apply(extract_gene_id)
    expr_df['gene_id_clean'] = expr_df['gene_id'].copy()
    
    # Calculate expression metric based on choice
    if expression_metric == 'tpm':
        expr_df['expression_value'] = expr_df['mean_tpm']
        expr_label = 'TPM'
    elif expression_metric == 'log_tpm':
        expr_df['expression_value'] = np.log1p(expr_df['mean_tpm'])  # log(TPM + 1) handles zeros
        expr_label = 'log(TPM + 1)'
    elif expression_metric == 'counts':
        expr_df['expression_value'] = expr_df['total_est_counts']
        expr_label = 'Estimated Counts'
    elif expression_metric == 'log_counts':
        expr_df['expression_value'] = np.log1p(expr_df['total_est_counts'])
        expr_label = 'log(Counts + 1)'
    else:
        raise ValueError(f"Unknown expression_metric: {expression_metric}")
    
    print(f"  Using expression metric: {expr_label}")
    
    # Merge on clean gene ID
    merged_df = rates_df.merge(
        expr_df[['gene_id_clean', 'expression_value', 'mean_tpm', 'total_est_counts']],
        on='gene_id_clean',
        how='inner',
        suffixes=('_rates', '_expr')
    )
    
    print(f"  Matched {len(merged_df)} genes between datasets")
    
    # Check for missing matches
    rates_only = rates_df[~rates_df['gene_id_clean'].isin(expr_df['gene_id_clean'])]
    expr_only = expr_df[~expr_df['gene_id_clean'].isin(rates_df['gene_id_clean'])]
    
    if len(rates_only) > 0:
        print(f"  Warning: {len(rates_only)} genes in rates file without expression data")
    if len(expr_only) > 0:
        print(f"  Warning: {len(expr_only)} genes in expression file without rates data")
    
    return merged_df, expr_label


def perform_correlations(x, y, rate_type):
    """
    Perform multiple correlation analyses: linear, log-log, and Spearman.
    
    Parameters:
    -----------
    x : array-like
        Independent variable (TPM)
    y : array-like
        Dependent variable (mutation rate)
    rate_type : str
        Type of rate ('control' or 'treated')
    
    Returns:
    --------
    dict with correlation statistics
    """
    # Remove any NaN or infinite values
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        print(f"  Warning: Insufficient data points for {rate_type} analysis")
        return None
    
    n = len(x_clean)
    
    # 1. Linear regression (original scale)
    slope_linear, intercept_linear, r_linear, p_linear, std_err_linear = stats.linregress(x_clean, y_clean)
    
    # 2. Log-log regression (power law relationship)
    log_x = np.log(x_clean)
    log_y = np.log(y_clean)
    slope_log, intercept_log, r_log, p_log, std_err_log = stats.linregress(log_x, log_y)
    
    # 3. Spearman correlation (non-parametric, rank-based)
    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
    
    # 4. Kendall's tau (another non-parametric measure)
    kendall_tau, kendall_p = stats.kendalltau(x_clean, y_clean)
    
    # Build linear model for residuals/outlier detection
    X_linear = sm.add_constant(x_clean)
    model_linear = sm.OLS(y_clean, X_linear).fit()
    conf_int_linear = model_linear.conf_int(alpha=0.05)
    ci_intercept_linear = conf_int_linear[0]
    ci_slope_linear = conf_int_linear[1]
    
    y_pred_linear = model_linear.predict(X_linear)
    residuals_linear = y_clean - y_pred_linear
    
    # Build log-log model
    X_log = sm.add_constant(log_x)
    model_log = sm.OLS(log_y, X_log).fit()
    conf_int_log = model_log.conf_int(alpha=0.05)
    ci_intercept_log = conf_int_log[0]
    ci_slope_log = conf_int_log[1]
    
    y_pred_log = model_log.predict(X_log)
    residuals_log = log_y - y_pred_log
    
    # Outlier detection on linear model
    influence = OLSInfluence(model_linear)
    cook_distance = influence.cooks_distance[0]
    standardized_residuals = residuals_linear / np.std(residuals_linear)
    outlier_threshold_cook = 4.0 / n
    is_outlier = (cook_distance > outlier_threshold_cook) | (np.abs(standardized_residuals) > 3)
    
    # Normality tests
    if len(residuals_linear) > 8:
        _, p_normality_linear = normaltest(residuals_linear)
        _, p_normality_log = normaltest(residuals_log)
    else:
        p_normality_linear = np.nan
        p_normality_log = np.nan
    
    results = {
        'rate_type': rate_type,
        'n_genes': n,
        'x_clean': x_clean,
        'y_clean': y_clean,
        'log_x': log_x,
        'log_y': log_y,
        # Linear regression results
        'linear': {
            'slope': slope_linear,
            'intercept': intercept_linear,
            'r_squared': r_linear ** 2,
            'pearson_r': r_linear,
            'p_value': p_linear,
            'std_err_slope': std_err_linear,
            'slope_ci_low': ci_slope_linear[0],
            'slope_ci_high': ci_slope_linear[1],
            'intercept_ci_low': ci_intercept_linear[0],
            'intercept_ci_high': ci_intercept_linear[1],
            'y_pred': y_pred_linear,
            'residuals': residuals_linear,
            'p_normality': p_normality_linear,
            'model': model_linear
        },
        # Log-log regression results
        'log_log': {
            'slope': slope_log,
            'intercept': intercept_log,
            'r_squared': r_log ** 2,
            'pearson_r': r_log,
            'p_value': p_log,
            'std_err_slope': std_err_log,
            'slope_ci_low': ci_slope_log[0],
            'slope_ci_high': ci_slope_log[1],
            'intercept_ci_low': ci_intercept_log[0],
            'intercept_ci_high': ci_intercept_log[1],
            'y_pred': np.exp(y_pred_log),  # Transform back to original scale
            'residuals': residuals_log,
            'p_normality': p_normality_log,
            'model': model_log
        },
        # Non-parametric correlations
        'spearman': {
            'r': spearman_r,
            'p_value': spearman_p
        },
        'kendall': {
            'tau': kendall_tau,
            'p_value': kendall_p
        },
        # Outlier information
        'cook_distance': cook_distance,
        'standardized_residuals': standardized_residuals,
        'is_outlier': is_outlier
    }
    
    return results


def create_plots(control_results, treated_results, output_path, expr_label='TPM'):
    """
    Create comprehensive multi-panel visualization showing all correlation methods.
    """
    # Set up the figure with subplots - larger to accommodate more plots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    # Define colors
    control_color = '#2E86AB'
    treated_color = '#A23B72'
    
    # Row 1: Linear scale plots with hexbin
    ax1 = fig.add_subplot(gs[0, 0])
    plot_scatter_hexbin(ax1, control_results, control_color, 'Control: Linear Scale', use_log=False, expr_label=expr_label)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_scatter_hexbin(ax2, treated_results, treated_color, 'Treated: Linear Scale', use_log=False, expr_label=expr_label)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_scatter_hexbin(ax3, control_results, control_color, 'Control: Log-Log Scale', use_log=True, expr_label=expr_label)
    
    ax4 = fig.add_subplot(gs[0, 3])
    plot_scatter_hexbin(ax4, treated_results, treated_color, 'Treated: Log-Log Scale', use_log=True, expr_label=expr_label)
    
    # Row 2: Residual plots
    ax5 = fig.add_subplot(gs[1, 0])
    plot_residuals(ax5, control_results, control_color, 'Control: Linear Residuals', use_log=False)
    
    ax6 = fig.add_subplot(gs[1, 1])
    plot_residuals(ax6, treated_results, treated_color, 'Treated: Linear Residuals', use_log=False)
    
    ax7 = fig.add_subplot(gs[1, 2])
    plot_residuals(ax7, control_results, control_color, 'Control: Log-Log Residuals', use_log=True)
    
    ax8 = fig.add_subplot(gs[1, 3])
    plot_residuals(ax8, treated_results, treated_color, 'Treated: Log-Log Residuals', use_log=True)
    
    # Row 3: Q-Q plots and distribution
    ax9 = fig.add_subplot(gs[2, 0])
    plot_qq(ax9, control_results, control_color, 'Control: Linear Q-Q', use_log=False)
    
    ax10 = fig.add_subplot(gs[2, 1])
    plot_qq(ax10, treated_results, treated_color, 'Treated: Linear Q-Q', use_log=False)
    
    ax11 = fig.add_subplot(gs[2, 2])
    plot_data_distribution(ax11, control_results, treated_results)
    
    ax12 = fig.add_subplot(gs[2, 3])
    plot_cooks_distance(ax12, control_results, control_color, "Control Cook's Distance")
    
    # Row 4: Summary and comparison
    ax13 = fig.add_subplot(gs[3, :2])
    plot_both_rates_comparison(ax13, control_results, treated_results, expr_label)
    
    ax14 = fig.add_subplot(gs[3, 2:])
    ax14.axis('off')
    plot_summary_text(ax14, control_results, treated_results, expr_label)
    
    plt.suptitle('Mutation Rate vs TPM Expression: Comprehensive Correlation Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")


def create_key_results_plot(control_results, treated_results, output_path, expr_label='TPM', use_log_scale=False):
    """
    Create a focused figure showing only the key correlation results.
    Optimized for publication with larger fonts and appropriate figure size.
    
    Args:
        use_log_scale: If True, use log-log scale (useful for wide ranges but can show negative values).
                      If False, use linear scale (clearer, no negative values).
    """
    # Publication-friendly figure size: wider to prevent overlap, suitable for full-width or double column
    # Width increased to give more space between panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Add spacing between subplots to prevent overlap
    plt.subplots_adjust(wspace=0.35)
    
    # Define colors
    control_color = '#2E86AB'
    treated_color = '#A23B72'
    
    # Control plot
    ax1 = axes[0]
    plot_key_scatter(ax1, control_results, control_color, 'Control', expr_label, use_log=use_log_scale)
    
    # Treated plot
    ax2 = axes[1]
    plot_key_scatter(ax2, treated_results, treated_color, 'Treated', expr_label, use_log=use_log_scale)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved key results plot to: {output_path}")


def plot_key_scatter(ax, results, color, title, expr_label, use_log=True):
    """Plot clean scatter plot with regression line for key results figure.
    Optimized for publication with larger fonts (minimum 10pt, typically 12-14pt).
    """
    if use_log:
        x = results['log_x']
        y = results['log_y']
        method = 'log_log'
        xlabel = f'log({expr_label})'
        ylabel = 'log(Mutation Rate)'
    else:
        x = results['x_clean']
        y = results['y_clean']
        method = 'linear'
        xlabel = expr_label
        ylabel = 'Mutation Rate'
    
    method_results = results[method]
    
    # Scatter plot with transparency - slightly larger markers for publication
    ax.scatter(x, y, alpha=0.4, s=40, color=color, edgecolors='none')
    
    # Regression line - slightly thicker for visibility
    x_sorted = np.sort(x)
    X_sorted = sm.add_constant(x_sorted)
    y_pred_sorted = method_results['model'].predict(X_sorted)
    
    ax.plot(x_sorted, y_pred_sorted, '--', color='black', linewidth=3.0, zorder=5)
    
    # Add confidence bands
    pred_ci = method_results['model'].get_prediction(X_sorted).conf_int(alpha=0.05)
    ax.fill_between(x_sorted, pred_ci[:, 0], pred_ci[:, 1], 
                    alpha=0.2, color=color, zorder=1)
    
    # Formatting - increased font sizes for publication (minimum 10pt, target 12-14pt)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(f'{title} Rate vs {expr_label}', fontsize=15, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    # Add statistics text box - only R² value
    stats_text = f"R² = {method_results['r_squared']:.4f}"
    
    # Position text box in upper right
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=12, family='monospace',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                     edgecolor='gray', alpha=0.9, linewidth=1.5))


def plot_scatter_hexbin(ax, results, color, title, use_log=False, expr_label='TPM'):
    """Plot scatter with hexbin density and regression line."""
    if use_log:
        x = results['log_x']
        y = results['log_y']
        method = 'log_log'
        xlabel = f'log({expr_label})'
        ylabel = 'log(Mutation Rate)'
    else:
        x = results['x_clean']
        y = results['y_clean']
        method = 'linear'
        xlabel = expr_label
        ylabel = 'Mutation Rate'
    
    is_outlier = results['is_outlier']
    method_results = results[method]
    
    # Use hexbin for density visualization
    hb = ax.hexbin(x[~is_outlier], y[~is_outlier], gridsize=30, cmap='Blues', 
                   mincnt=1, alpha=0.6)
    
    # Highlight outliers
    if np.any(is_outlier):
        ax.scatter(x[is_outlier], y[is_outlier], alpha=0.8, s=50, 
                  color='red', marker='x', label='Outliers', linewidths=2, zorder=5)
    
    # Plot regression line
    x_sorted = np.sort(x)
    if use_log:
        # For log-log, predict on log scale
        X_sorted = sm.add_constant(x_sorted)
        y_pred_sorted = method_results['model'].predict(X_sorted)
    else:
        y_pred_sorted = method_results['y_pred'][np.argsort(x)]
    
    ax.plot(x_sorted, y_pred_sorted, '--', color='red', linewidth=2, 
           label='Regression', zorder=4)
    
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f"R² = {method_results['r_squared']:.4f}\n"
                  f"r = {method_results['pearson_r']:.4f}\n"
                  f"p = {method_results['p_value']:.2e}\n"
                  f"n = {results['n_genes']}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))


def plot_residuals(ax, results, color, title, use_log=False):
    """Plot residuals vs fitted values."""
    method = 'log_log' if use_log else 'linear'
    method_results = results[method]
    
    y_pred = method_results['y_pred']
    residuals = method_results['residuals']
    is_outlier = results['is_outlier']
    
    ax.scatter(y_pred[~is_outlier], residuals[~is_outlier], 
              alpha=0.5, s=30, color=color)
    if np.any(is_outlier):
        ax.scatter(y_pred[is_outlier], residuals[is_outlier],
                  alpha=0.8, s=50, color='red', marker='x', linewidths=2)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Predicted Mutation Rate', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_qq(ax, results, color, title, use_log=False):
    """Q-Q plot for residual normality."""
    method = 'log_log' if use_log else 'linear'
    method_results = results[method]
    
    residuals = method_results['residuals']
    standardized = residuals / np.std(residuals)
    
    stats.probplot(standardized, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_markeredgecolor(color)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('black')
    ax.get_lines()[1].set_linewidth(2)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add normality test p-value
    p_norm = method_results['p_normality']
    if not np.isnan(p_norm):
        ax.text(0.05, 0.95, f"Normality p = {p_norm:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_cooks_distance(ax, results, color, title):
    """Plot Cook's distance for outlier detection."""
    cook_dist = results['cook_distance']
    is_outlier = results['is_outlier']
    n = len(cook_dist)
    threshold = 4.0 / n
    
    indices = np.arange(len(cook_dist))
    ax.scatter(indices[~is_outlier], cook_dist[~is_outlier],
              alpha=0.5, s=30, color=color)
    if np.any(is_outlier):
        ax.scatter(indices[is_outlier], cook_dist[is_outlier],
                  alpha=0.8, s=50, color='red', marker='x', linewidths=2)
    
    ax.axhline(y=threshold, color='red', linestyle='--', 
              linewidth=1, label=f'Threshold ({threshold:.4f})')
    ax.set_xlabel('Gene Index', fontsize=10)
    ax.set_ylabel("Cook's Distance", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
def plot_data_distribution(ax, control_results, treated_results):
    """Plot distribution of TPM and mutation rates."""
    ax.hist(control_results['x_clean'], bins=50, alpha=0.5, 
           color='#2E86AB', label='Control TPM', density=True)
    ax.hist(treated_results['x_clean'], bins=50, alpha=0.5,
           color='#A23B72', label='Treated TPM', density=True)
    ax.set_xlabel('TPM', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('TPM Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
def plot_both_rates_comparison(ax, control_results, treated_results, expr_label='TPM'):
    """Plot both rates with all correlation methods."""
    x_control = control_results['x_clean']
    y_control = control_results['y_clean']
    x_treated = treated_results['x_clean']
    y_treated = treated_results['y_clean']
    
    # Scatter plots
    ax.scatter(x_control, y_control, alpha=0.3, s=20, 
              color='#2E86AB', label='Control', marker='o')
    ax.scatter(x_treated, y_treated, alpha=0.3, s=20,
              color='#A23B72', label='Treated', marker='s')
    
    # Regression lines
    x_sorted_control = np.sort(x_control)
    y_pred_control = control_results['linear']['y_pred'][np.argsort(x_control)]
    ax.plot(x_sorted_control, y_pred_control, '--', 
           color='#2E86AB', linewidth=2, label='Control linear')
    
    x_sorted_treated = np.sort(x_treated)
    y_pred_treated = treated_results['linear']['y_pred'][np.argsort(x_treated)]
    ax.plot(x_sorted_treated, y_pred_treated, '--',
           color='#A23B72', linewidth=2, label='Treated linear')
    
    ax.set_xlabel(f'{expr_label}', fontsize=11)
    ax.set_ylabel('Mutation Rate', fontsize=11)
    ax.set_title(f'Both Rates vs {expr_label} (Linear Scale)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)




def plot_summary_text(ax, control_results, treated_results, expr_label='TPM'):
    """Display summary statistics as text."""
    c_lin = control_results['linear']
    c_log = control_results['log_log']
    t_lin = treated_results['linear']
    t_log = treated_results['log_log']
    
    text_lines = [
        "CORRELATION SUMMARY",
        "=" * 60,
        "",
        f"CONTROL RATE vs {expr_label.upper()}:",
        "  Linear Regression:",
        f"    R² = {c_lin['r_squared']:.4f}, r = {c_lin['pearson_r']:.4f}, p = {c_lin['p_value']:.2e}",
        "  Log-Log Regression:",
        f"    R² = {c_log['r_squared']:.4f}, r = {c_log['pearson_r']:.4f}, p = {c_log['p_value']:.2e}",
        "  Spearman (rank):",
        f"    ρ = {control_results['spearman']['r']:.4f}, p = {control_results['spearman']['p_value']:.2e}",
        "  Kendall's τ:",
        f"    τ = {control_results['kendall']['tau']:.4f}, p = {control_results['kendall']['p_value']:.2e}",
        "",
        f"TREATED RATE vs {expr_label.upper()}:",
        "  Linear Regression:",
        f"    R² = {t_lin['r_squared']:.4f}, r = {t_lin['pearson_r']:.4f}, p = {t_lin['p_value']:.2e}",
        "  Log-Log Regression:",
        f"    R² = {t_log['r_squared']:.4f}, r = {t_log['pearson_r']:.4f}, p = {t_log['p_value']:.2e}",
        "  Spearman (rank):",
        f"    ρ = {treated_results['spearman']['r']:.4f}, p = {treated_results['spearman']['p_value']:.2e}",
        "  Kendall's τ:",
        f"    τ = {treated_results['kendall']['tau']:.4f}, p = {treated_results['kendall']['p_value']:.2e}",
        "",
        f"n genes: {control_results['n_genes']} (control), {treated_results['n_genes']} (treated)",
        f"Outliers: {np.sum(control_results['is_outlier'])} (control), {np.sum(treated_results['is_outlier'])} (treated)",
    ]
    
    text = '\n'.join(text_lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
           fontsize=9, family='monospace',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


def save_results(merged_df, control_results, treated_results, output_dir):
    """Save summary and detailed results to TSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics - comprehensive with all methods
    summary_rows = []
    for rate_type, results in [('control', control_results), ('treated', treated_results)]:
        lin = results['linear']
        log = results['log_log']
        sp = results['spearman']
        kd = results['kendall']
        
        summary_rows.append({
            'rate_type': rate_type,
            'method': 'linear',
            'r_squared': lin['r_squared'],
            'pearson_r': lin['pearson_r'],
            'p_value': lin['p_value'],
            'slope': lin['slope'],
            'intercept': lin['intercept'],
            'n_genes': results['n_genes'],
            'p_normality': lin['p_normality']
        })
        summary_rows.append({
            'rate_type': rate_type,
            'method': 'log_log',
            'r_squared': log['r_squared'],
            'pearson_r': log['pearson_r'],
            'p_value': log['p_value'],
            'slope': log['slope'],
            'intercept': log['intercept'],
            'n_genes': results['n_genes'],
            'p_normality': log['p_normality']
        })
        summary_rows.append({
            'rate_type': rate_type,
            'method': 'spearman',
            'r_squared': np.nan,
            'pearson_r': sp['r'],
            'p_value': sp['p_value'],
            'slope': np.nan,
            'intercept': np.nan,
            'n_genes': results['n_genes'],
            'p_normality': np.nan
        })
        summary_rows.append({
            'rate_type': rate_type,
            'method': 'kendall',
            'r_squared': np.nan,
            'pearson_r': kd['tau'],
            'p_value': kd['p_value'],
            'slope': np.nan,
            'intercept': np.nan,
            'n_genes': results['n_genes'],
            'p_normality': np.nan
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / 'mutation_rate_tpm_correlation_summary.tsv'
    summary_df.to_csv(summary_path, sep='\t', index=False)
    print(f"  Saved summary statistics to: {summary_path}")
    
    # Detailed results
    control_mask = (np.isfinite(merged_df['expression_value']) & 
                   np.isfinite(merged_df['rate_control']) & 
                   (merged_df['expression_value'] > 0) & 
                   (merged_df['rate_control'] > 0))
    treated_mask = (np.isfinite(merged_df['expression_value']) & 
                   np.isfinite(merged_df['rate_treated']) & 
                   (merged_df['expression_value'] > 0) & 
                   (merged_df['rate_treated'] > 0))
    
    detailed_df = merged_df[['gene_id_clean', 'expression_value', 'mean_tpm', 'total_est_counts', 'rate_control', 'rate_treated']].copy()
    detailed_df.columns = ['gene_id', 'expression_value', 'tpm', 'counts', 'rate_control', 'rate_treated']
    
    # Add residuals and other metrics
    detailed_df['residual_control_linear'] = np.nan
    detailed_df['residual_control_log'] = np.nan
    detailed_df['residual_treated_linear'] = np.nan
    detailed_df['residual_treated_log'] = np.nan
    detailed_df['cook_distance_control'] = np.nan
    detailed_df['cook_distance_treated'] = np.nan
    detailed_df['is_outlier_control'] = False
    detailed_df['is_outlier_treated'] = False
    
    # Fill in control results
    control_indices = np.where(control_mask)[0]
    if len(control_indices) == len(control_results['linear']['residuals']):
        detailed_df.loc[control_mask, 'residual_control_linear'] = control_results['linear']['residuals']
        detailed_df.loc[control_mask, 'residual_control_log'] = control_results['log_log']['residuals']
        detailed_df.loc[control_mask, 'cook_distance_control'] = control_results['cook_distance']
        detailed_df.loc[control_mask, 'is_outlier_control'] = control_results['is_outlier']
    
    # Fill in treated results
    treated_indices = np.where(treated_mask)[0]
    if len(treated_indices) == len(treated_results['linear']['residuals']):
        detailed_df.loc[treated_mask, 'residual_treated_linear'] = treated_results['linear']['residuals']
        detailed_df.loc[treated_mask, 'residual_treated_log'] = treated_results['log_log']['residuals']
        detailed_df.loc[treated_mask, 'cook_distance_treated'] = treated_results['cook_distance']
        detailed_df.loc[treated_mask, 'is_outlier_treated'] = treated_results['is_outlier']
    
    detailed_path = output_dir / 'mutation_rate_tpm_correlation_detailed.tsv'
    detailed_df.to_csv(detailed_path, sep='\t', index=False)
    print(f"  Saved detailed results to: {detailed_path}")


def main():
    """Main analysis function."""
    # Default file paths
    parser = argparse.ArgumentParser(description='Correlate mutation rates with expression levels')
    parser.add_argument('-m','--mutation-rates-file', type=str, required=True,
                        help='Path to the mutation rates file')
    parser.add_argument('-e','--expression-file', type=str, required=True,
                        help='Path to the expression file')
    parser.add_argument('-o','--output-dir', type=str, required=True,
                        help='Path to the output directory')
    parser.add_argument('--expression-metric', type=str, default='log_tpm',
                        choices=['tpm', 'log_tpm', 'counts', 'log_counts'],
                        help='Expression metric to use (default: log_tpm)')
    parser.add_argument('--use-linear-scale', action='store_true',
                        help='Use linear scale for plots (default: log scale). Note: log scale can show negative values for small positive rates.')
    args = parser.parse_args()
    mutation_rates_file = args.mutation_rates_file
    expression_file = args.expression_file
    output_dir = args.output_dir
    expression_metric = args.expression_metric

    print("=" * 70)
    print("Mutation Rate vs TPM Expression Correlation Analysis")
    print("=" * 70)
    print(f"Mutation rates file: {mutation_rates_file}")
    print(f"Expression file: {expression_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Check input files
    if not os.path.exists(mutation_rates_file):
        print(f"Error: Mutation rates file not found: {mutation_rates_file}")
        sys.exit(1)
    if not os.path.exists(expression_file):
        print(f"Error: Expression file not found: {expression_file}")
        sys.exit(1)
    
    # Load and merge data
    merged_df, expr_label = load_data(mutation_rates_file, expression_file, expression_metric)
    
    if len(merged_df) == 0:
        print("Error: No genes matched between datasets")
        sys.exit(1)
    
    # Perform correlation analyses
    print("\nPerforming correlation analysis...")
    print(f"  Control rate vs {expr_label}...")
    control_results = perform_correlations(
        merged_df['expression_value'].values,
        merged_df['rate_control'].values,
        'control'
    )
    
    print(f"  Treated rate vs {expr_label}...")
    treated_results = perform_correlations(
        merged_df['expression_value'].values,
        merged_df['rate_treated'].values,
        'treated'
    )
    
    if control_results is None or treated_results is None:
        print("Error: Correlation analysis failed")
        sys.exit(1)
    
    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)
    print(f"\nCONTROL RATE vs {expr_label.upper()}:")
    print(f"  Linear:     R² = {control_results['linear']['r_squared']:.4f}, r = {control_results['linear']['pearson_r']:.4f}, p = {control_results['linear']['p_value']:.2e}")
    print(f"  Log-Log:    R² = {control_results['log_log']['r_squared']:.4f}, r = {control_results['log_log']['pearson_r']:.4f}, p = {control_results['log_log']['p_value']:.2e}")
    print(f"  Spearman:   ρ = {control_results['spearman']['r']:.4f}, p = {control_results['spearman']['p_value']:.2e}")
    print(f"  Kendall:    τ = {control_results['kendall']['tau']:.4f}, p = {control_results['kendall']['p_value']:.2e}")
    print(f"  n genes = {control_results['n_genes']}")
    
    print(f"\nTREATED RATE vs {expr_label.upper()}:")
    print(f"  Linear:     R² = {treated_results['linear']['r_squared']:.4f}, r = {treated_results['linear']['pearson_r']:.4f}, p = {treated_results['linear']['p_value']:.2e}")
    print(f"  Log-Log:    R² = {treated_results['log_log']['r_squared']:.4f}, r = {treated_results['log_log']['pearson_r']:.4f}, p = {treated_results['log_log']['p_value']:.2e}")
    print(f"  Spearman:   ρ = {treated_results['spearman']['r']:.4f}, p = {treated_results['spearman']['p_value']:.2e}")
    print(f"  Kendall:    τ = {treated_results['kendall']['tau']:.4f}, p = {treated_results['kendall']['p_value']:.2e}")
    print(f"  n genes = {treated_results['n_genes']}")
    
    print(f"\nOUTLIERS:")
    print(f"  Control: {np.sum(control_results['is_outlier'])} genes")
    print(f"  Treated: {np.sum(treated_results['is_outlier'])} genes")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print("\nSaving results...")
    save_results(merged_df, control_results, treated_results, output_dir)
    
    # Create plots
    print("\nGenerating plots...")
    plot_path = output_dir / 'mutation_rate_tpm_correlation_plots.png'
    create_plots(control_results, treated_results, plot_path, expr_label)
    
    # Create key results plot (simplified) - using log scale by default (can be overridden with --use-linear-scale)
    key_plot_path = output_dir / 'mutation_rate_tpm_correlation_key_results.png'
    use_log = not args.use_linear_scale  # Default to log scale unless user requests linear
    create_key_results_plot(control_results, treated_results, key_plot_path, expr_label, use_log_scale=use_log)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

