#!/usr/bin/env python3
"""
Script to regenerate publication-ready plots from estimate_rates.py output TSV files.

Creates a single multiplot figure with:
1. GLM rates per sample (bar plot with error bars)
2. Mutation category significances (intergenic vs genic[nonsyn/syn])
3. Rates per treatment time group
"""

import argparse
import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter, LogLocator, MaxNLocator, LogFormatterSciNotation
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
rcParams['font.size'] = 28
rcParams['axes.labelsize'] = 32
rcParams['axes.titlesize'] = 36
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 24
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['axes.linewidth'] = 2.0
rcParams['grid.linewidth'] = 1.2
rcParams['grid.alpha'] = 0.3

# Color scheme matching plot_ems_spectra: grey for controls, orange-to-red gradient for treated
NT_GREY = '#808080'  # Grey for controls (matching plot_ems_spectra)
# Orange-to-red gradient colors for treated samples (matching plot_ems_spectra)
TREATED_COLORS = ['#FFB366', '#FF6B35', '#C1121F']  # Light orange, orange-red, dark red
# Blue-to-purple gradient colors for category panel (middle panel) - complements orange-to-red
CATEGORY_COLORS = ['#6BA3D8', '#4A7BC8', '#3D5AA6']  # Light blue, medium blue, dark blue-purple

COLORS = {
    'control': NT_GREY,      # Grey
    'treated': TREATED_COLORS[1],  # Medium orange-red (default treated color)
    'intergenic': TREATED_COLORS[0],  # Light orange
    'synonymous': TREATED_COLORS[1],  # Orange-red
    'non_synonymous': TREATED_COLORS[2],  # Dark red
    '3d': TREATED_COLORS[0],  # Light orange
    '7d': TREATED_COLORS[1],  # Orange-red
    'no_label': TREATED_COLORS[2],  # Dark red
}

def derive_sample_label(sample_name, treated_str='EMS', control_str='NT'):
    """Derive cleaner sample labels using the logic from estimate_rates.py"""
    s_str = str(sample_name)
    base = None
    
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


def format_pvalue(p):
    """Format p-value for display."""
    if p < 0.001:
        return '***', f'p<0.001'
    elif p < 0.01:
        return '**', f'p={p:.3f}'
    elif p < 0.05:
        return '*', f'p={p:.3f}'
    else:
        return 'ns', f'p={p:.3f}'


def load_data(output_dir):
    """Load all TSV files from the output directory."""
    data = {}
    
    # Load per-sample GLM rates
    glm_path = os.path.join(output_dir, 'site_level_glm_per_sample.tsv')
    if os.path.exists(glm_path):
        data['glm_per_sample'] = pd.read_csv(glm_path, sep='\t')
        # Extract treatment status from sample name
        data['glm_per_sample']['is_control'] = data['glm_per_sample']['sample'].str.contains('NT', case=False, na=False)
        data['glm_per_sample']['is_control'] = ~data['glm_per_sample']['sample'].str.contains('EMS', case=False, na=False)
    else:
        print(f"Warning: {glm_path} not found")
        data['glm_per_sample'] = None
    
    # Load category rates
    # Prefer category_rates_summary.tsv which contains the corrected rates with 5mer normalization
    # This file is now generated with proper 5mer handling (only applied to treated samples)
    category_path = os.path.join(output_dir, 'category_comparison', 'category_rates_summary.tsv')
    if not os.path.exists(category_path):
        # Fall back to nb_glm file (older format without 5mer normalization)
        category_path = os.path.join(output_dir, 'category_comparison', 'category_rates_nb_glm.tsv')
    if os.path.exists(category_path):
        data['category_rates'] = pd.read_csv(category_path, sep='\t')
        # Check for invalid rates (negative or near-zero controls)
        if 'rate' in data['category_rates'].columns and 'treatment' in data['category_rates'].columns:
            control_rates = data['category_rates'][data['category_rates']['treatment'] == 'control']['rate']
            treated_rates = data['category_rates'][data['category_rates']['treatment'] == 'treated']['rate']
            # If control rates are suspiciously low (< 1e-8), data may be corrupted
            if len(control_rates) > 0 and (control_rates < 1e-8).any():
                print(f"Warning: {category_path} has suspiciously low control rates (< 1e-8), may indicate a bug")
                print(f"  Control rate range: {control_rates.min():.2e} to {control_rates.max():.2e}")
            # If treated rates are negative, data is on wrong scale
            if len(treated_rates) > 0 and (treated_rates < 0).any():
                print(f"Warning: {category_path} has negative rates, data may be on wrong scale")
        print(f"Loaded category rates from: {category_path}")
    else:
        print(f"Warning: Category rates file not found in {output_dir}/category_comparison/")
        data['category_rates'] = None
    
    # Load category statistical tests
    category_tests_path = os.path.join(output_dir, 'category_comparison', 'category_statistical_tests.tsv')
    if os.path.exists(category_tests_path):
        data['category_tests'] = pd.read_csv(category_tests_path, sep='\t')
    else:
        print(f"Warning: Category tests file not found")
        data['category_tests'] = None
    
    # Load treatment day rates
    treatment_day_path = os.path.join(output_dir, 'treatment_day_comparison', 'treatment_day_rates_summary.tsv')
    if os.path.exists(treatment_day_path):
        data['treatment_day_rates'] = pd.read_csv(treatment_day_path, sep='\t')
    else:
        print(f"Warning: Treatment day rates file not found in {output_dir}/treatment_day_comparison/")
        data['treatment_day_rates'] = None
    
    # Load per-sample treatment day rates
    treatment_day_per_sample_path = os.path.join(output_dir, 'treatment_day_comparison', 'treatment_day_per_sample_rates.tsv')
    if os.path.exists(treatment_day_per_sample_path):
        data['treatment_day_per_sample'] = pd.read_csv(treatment_day_per_sample_path, sep='\t')
    else:
        print(f"Warning: Per-sample treatment day rates file not found")
        data['treatment_day_per_sample'] = None
    
    # Load treatment day statistical tests
    treatment_day_tests_path = os.path.join(output_dir, 'treatment_day_comparison', 'treatment_day_statistical_tests.tsv')
    if os.path.exists(treatment_day_tests_path):
        data['treatment_day_tests'] = pd.read_csv(treatment_day_tests_path, sep='\t')
    else:
        print(f"Warning: Treatment day tests file not found")
        data['treatment_day_tests'] = None
    
    # Load 5mer normalized window rates if available
    kmer5_window_path = os.path.join(output_dir, '5mer_normalized_windows', 'window_rates.tsv')
    if os.path.exists(kmer5_window_path):
        data['kmer5_windows'] = pd.read_csv(kmer5_window_path, sep='\t')
        print(f"Loaded 5mer normalized window rates from {kmer5_window_path}")
    else:
        data['kmer5_windows'] = None
    
    return data


def plot_glm_per_sample(ax, df):
    """Plot per-sample GLM rates with error bars.
    
    Returns:
        Dictionary mapping sample names to their colors
    """
    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No per-sample GLM data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=24)
        return {}
    
    # Sort by rate, separating controls and treated
    df = df.copy()
    df = df.sort_values('site_glm_rate', ascending=True)
    
    # Create color array - use plot_ems_spectra color scheme with gradient for treated samples
    # Separate controls and treated
    control_mask = df['is_control'].values
    treated_samples = df[~control_mask]
    n_treated = len(treated_samples)
    
    # Create colors array
    colors = []
    for is_ctrl in df['is_control']:
        if is_ctrl:
            colors.append(NT_GREY)  # Grey for controls
        else:
            # Use gradient for treated samples (same as plot_ems_spectra)
            if n_treated == 1:
                colors.append('#FF6B35')  # Single orange-red color
            else:
                # Create gradient from light orange to dark red (same palette as plot_ems_spectra)
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'orange_red', ['#FFB366', '#FF8C42', '#FF6B35', '#E63946', '#C1121F']
                )
                # Get index of this treated sample in the sorted treated list
                treated_idx = len([c for c in colors if c != NT_GREY])
                treated_color = mcolors.rgb2hex(cmap(treated_idx / (n_treated - 1)))
                colors.append(treated_color)
    
    # Create y positions
    y_pos = np.arange(len(df))
    
    # Plot bars
    bars = ax.barh(y_pos, df['site_glm_rate'], color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5, height=0.8)
    
    # Add error bars
    if 'site_glm_CI_low' in df.columns and 'site_glm_CI_high' in df.columns:
        xerr_low = df['site_glm_rate'] - df['site_glm_CI_low']
        xerr_high = df['site_glm_CI_high'] - df['site_glm_rate']
        ax.errorbar(df['site_glm_rate'], y_pos,
                   xerr=[xerr_low, xerr_high], fmt='none', 
                   color='black', capsize=2, capthick=1, elinewidth=1, alpha=0.7)
    
    # Format sample labels using proper function
    labels = df['sample'].apply(derive_sample_label)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=14)
    
    # Format x-axis with more ticks for better readability
    # Reduced labelpad to create space for panel 3 label
    ax.set_xlabel('Mutation Rate (per base)', fontsize=32, fontweight='bold', labelpad=5)
    ax.set_xscale('log')
    
    # Set explicit ticks covering the typical range of mutation rates
    ax.set_xticks([3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4])
    
    # Format with scientific notation
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(fmt)
    
    # Add minor ticks for finer resolution
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
    
    ax.tick_params(axis='x', labelsize=28, pad=15, which='major')
    ax.tick_params(axis='y', labelsize=14, pad=8)
    
    # Add legend - use plot_ems_spectra colors
    control_patch = mpatches.Patch(color=NT_GREY, label='Control', alpha=0.8)
    treated_patch = mpatches.Patch(color=TREATED_COLORS[1], label='Treated', alpha=0.8)
    ax.legend(handles=[control_patch, treated_patch], loc='lower right', frameon=True, 
              fancybox=True, shadow=True, fontsize=28, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
    
    # Return sample color mapping for use in other plots
    sample_color_dict = {}
    for i, (sample, color) in enumerate(zip(df['sample'], colors)):
        sample_color_dict[sample] = color
    
    return sample_color_dict


def plot_control_baseline_model_comparison(fig, category_df, tests_df):
    """Create multi-panel plot comparing control-baseline vs original model approach.
    
    Shows:
    1. Control rates by category (baseline)
    2. Treatment effects (constant vs category-specific)
    3. Predicted rates comparison
    """
    if category_df is None or category_df.empty:
        fig.text(0.5, 0.5, 'No category data available', 
                ha='center', va='center', transform=fig.transFigure, fontsize=24)
        return
    
    # Get control and treated rates
    control_df = category_df[category_df['treatment'] == 'control'].copy()
    treated_df = category_df[category_df['treatment'] == 'treated'].copy()
    
    if control_df.empty or treated_df.empty:
        fig.text(0.5, 0.5, 'Missing control or treated data', 
                ha='center', va='center', transform=fig.transFigure, fontsize=24)
        return
    
    # Filter out coding category for main plots
    control_df = control_df[control_df['category'] != 'coding'].copy()
    treated_df = treated_df[treated_df['category'] != 'coding'].copy()
    
    # Create 3-panel layout
    gs = fig.add_gridspec(1, 3, hspace=0.4, wspace=0.5, 
                         left=0.08, right=0.95, top=0.90, bottom=0.15,
                         width_ratios=[1.0, 1.0, 1.0])
    
    # Panel 1: Control rates (baseline)
    ax1 = fig.add_subplot(gs[0, 0])
    control_df = control_df.sort_values('rate', ascending=True)
    x_pos1 = np.arange(len(control_df))
    colors1 = [TREATED_COLORS[i] for i in np.linspace(0, len(TREATED_COLORS)-1, len(control_df), dtype=int)]
    
    bars1 = ax1.bar(x_pos1, control_df['rate'], color=colors1, alpha=0.7,
                   edgecolor='black', linewidth=1, width=0.6)
    
    if 'CI_low' in control_df.columns and 'CI_high' in control_df.columns:
        yerr_low = control_df['rate'] - control_df['CI_low']
        yerr_high = control_df['CI_high'] - control_df['rate']
        ax1.errorbar(x_pos1, control_df['rate'],
                    yerr=[yerr_low, yerr_high], fmt='none',
                    color='black', capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8)
    
    category_labels = {'intergenic': 'Intergenic', 'synonymous': 'Synonymous', 'non_synonymous': 'Non-syn'}
    labels1 = [category_labels.get(cat, cat) for cat in control_df['category']]
    ax1.set_xticks(x_pos1)
    ax1.set_xticklabels(labels1, fontsize=18, fontweight='bold', rotation=15, ha='right')
    ax1.set_ylabel('Mutation Rate (per base)', fontsize=22, fontweight='bold', labelpad=20)
    ax1.set_yscale('log')
    ax1.set_title('Control Rates\n(Baseline)', fontsize=24, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax1.yaxis.set_major_formatter(fmt)
    ax1.tick_params(axis='y', labelsize=18, pad=10)
    
    # Panel 2: Absolute treatment effects - simple, let data speak
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate absolute effects
    absolute_effects = []
    categories_ordered = []
    
    for cat in control_df['category']:
        control_rate = control_df[control_df['category'] == cat]['rate'].values[0]
        treated_rate = treated_df[treated_df['category'] == cat]['rate'].values[0]
        if control_rate > 0:
            absolute_effect = treated_rate - control_rate
            absolute_effects.append(absolute_effect)
            categories_ordered.append(cat)
    
    x_pos2 = np.arange(len(absolute_effects))
    labels2 = [category_labels.get(cat, cat) for cat in categories_ordered]
    
    # Plot absolute effects on log scale
    bars2 = ax2.bar(x_pos2, absolute_effects, color=TREATED_COLORS[1], alpha=0.8,
                   edgecolor='black', linewidth=1.5, width=0.6)
    
    # Set log scale
    ax2.set_yscale('log')
    
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(labels2, fontsize=18, fontweight='bold', rotation=15, ha='right')
    ax2.set_ylabel('Absolute Treatment Effect\n(Treated - Control)', fontsize=22, fontweight='bold', labelpad=20)
    ax2.set_title('Treatment Effect by Category', fontsize=24, fontweight='bold', pad=15)
    
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y', which='both')
    ax2.tick_params(axis='y', labelsize=18, pad=10)
    
    # Format y-axis with scientific notation
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax2.yaxis.set_major_formatter(fmt)
    
    # Add minor ticks
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
    
    # Panel 3: Statistical test results - show if treatment effect is constant
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Get test results
    abs_test = None
    abs_model_test = None
    control_baseline_test = None
    
    if tests_df is not None and not tests_df.empty:
        # Prioritize: Absolute effect model > Absolute treatment effect > Control-baseline
        abs_model_test = tests_df[tests_df['test'].str.contains('Absolute effect model', case=False, na=False)]
        abs_test = tests_df[tests_df['test'].str.contains('Absolute treatment effect.*coding vs intergenic', case=False, na=False)]
        control_baseline_test = tests_df[tests_df['test'].str.contains('Control-baseline', case=False, na=False)]
    
    # Calculate absolute effects for visualization
    absolute_effects_viz = {}
    fold_changes_viz = {}
    for cat in control_df['category']:
        control_rate = control_df[control_df['category'] == cat]['rate'].values[0]
        treated_rate = treated_df[treated_df['category'] == cat]['rate'].values[0]
        if control_rate > 0:
            absolute_effects_viz[cat] = treated_rate - control_rate
            fold_changes_viz[cat] = treated_rate / control_rate
    
    # Prioritize showing the results more clearly
    has_test_result = False
    
    # Show absolute effect MODEL test if available (NEW: tests constant absolute mutations/base)
    if abs_model_test is not None and not abs_model_test.empty:
        row = abs_model_test.iloc[0]
        p_val = row['p_value']
        abs_effect_intergenic = row.get('absolute_effect_intergenic', np.nan)
        abs_effect_syn = row.get('absolute_effect_synonymous', np.nan)
        abs_effect_nonsyn = row.get('absolute_effect_nonsyn', np.nan)
        
        has_test_result = True
        
        # Create bar plot showing absolute effects by category
        if not np.isnan(abs_effect_intergenic) and not np.isnan(abs_effect_syn) and not np.isnan(abs_effect_nonsyn):
            effects = [abs_effect_intergenic, abs_effect_syn, abs_effect_nonsyn]
            labels = ['Intergenic', 'Synonymous', 'Non-synonymous']
            x_pos = np.arange(len(labels))
            
            colors = ['#FF7F0E', '#2CA02C', '#1F77B4']
            bars = ax3.bar(x_pos, effects, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=2, width=0.6)
            
            # Add value labels on bars
            for bar, val in zip(bars, effects):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2e}', ha='center', va='bottom', 
                        fontsize=16, fontweight='bold')
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(labels, fontsize=18, fontweight='bold')
            ax3.set_ylabel('Absolute Treatment Effect\n(mutations per base added by EMS)', 
                          fontsize=18, fontweight='bold', labelpad=15)
            
            # Add p-value and interpretation
            p_text = f'LRT p = {p_val:.2e}' if p_val < 0.001 else f'LRT p = {p_val:.3f}'
            if p_val > 0.05:
                interp = 'Constant across categories'
            else:
                interp = 'Differs by category'
            
            ax3.set_title(f"Absolute Effect Model\n{p_text} ({interp})", 
                         fontsize=20, fontweight='bold', pad=15)
            
            ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            ax3.tick_params(axis='both', labelsize=16)
            
            # Format y-axis
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((-3, 3))
            ax3.yaxis.set_major_formatter(fmt)
    
    # Fallback: Show absolute effect test if available (OLD: compares coding vs intergenic)
    elif abs_test is not None and not abs_test.empty:
        row = abs_test.iloc[0]
        p_val = row['p_value']
        abs_effect_intergenic = row.get('absolute_effect_intergenic', np.nan)
        abs_effect_coding = row.get('absolute_effect_coding', np.nan)
        
        has_test_result = True
        
        # Create visualization showing the difference
        if not np.isnan(abs_effect_intergenic) and not np.isnan(abs_effect_coding):
            # Bar plot showing absolute effects
            effects = [abs_effect_intergenic, abs_effect_coding]
            labels = ['Intergenic', 'Coding\n(Syn+Nonsyn)']
            x_pos = np.arange(len(labels))
            
            colors = ['#FF7F0E', '#2CA02C']
            bars = ax3.bar(x_pos, effects, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=2, width=0.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, effects):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2e}', ha='center', va='bottom', 
                        fontsize=18, fontweight='bold')
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(labels, fontsize=20, fontweight='bold')
            ax3.set_ylabel('Absolute Treatment Effect\n(Treated - Control Rate)', 
                          fontsize=20, fontweight='bold', labelpad=15)
            
            # Add p-value
            p_text = f'p = {p_val:.2e}' if p_val < 0.001 else f'p = {p_val:.3f}'
            ax3.set_title(f"Absolute Treatment Effect by Category\n{p_text}", 
                         fontsize=22, fontweight='bold', pad=15)
            
            ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            ax3.tick_params(axis='both', labelsize=18)
            
            # Format y-axis
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((-3, 3))
            ax3.yaxis.set_major_formatter(fmt)
    
    # Fallback to control-baseline LRT if absolute test not available
    elif control_baseline_test is not None and not control_baseline_test.empty:
        row = control_baseline_test.iloc[0]
        p_val = row['p_value']
        
        has_test_result = True
        
        # Show fold-changes with interpretation
        cats_list = list(fold_changes_viz.keys())
        fcs = [fold_changes_viz[cat] for cat in cats_list]
        labels_viz = [category_labels.get(cat, cat) for cat in cats_list]
        x_pos = np.arange(len(fcs))
        
        bars = ax3.bar(x_pos, fcs, color=TREATED_COLORS[1], alpha=0.7,
                      edgecolor='black', linewidth=2, width=0.6)
        
        ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        
        for bar, fc in zip(bars, fcs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{fc:.2f}x', ha='center', va='bottom', 
                    fontsize=16, fontweight='bold')
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels_viz, fontsize=18, fontweight='bold', rotation=15, ha='right')
        ax3.set_ylabel('Fold-change\n(Treated / Control)', fontsize=20, fontweight='bold', labelpad=15)
        
        # Add p-value
        p_text = f'p = {p_val:.2e}' if p_val < 0.001 else f'p = {p_val:.3f}'
        ax3.set_title(f"Treatment Effect by Category\n(Control-baseline LRT: {p_text})", 
                     fontsize=22, fontweight='bold', pad=15)
        
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax3.tick_params(axis='both', labelsize=18)
    
    if not has_test_result:
        ax3.text(0.5, 0.5, 'No statistical test results available', 
                ha='center', va='center', transform=ax3.transAxes, 
                fontsize=20, fontweight='bold')


def plot_control_baseline_comparison(ax, tests_df):
    """Plot comparison of control-baseline approach vs original approach."""
    if tests_df is None or tests_df.empty:
        ax.text(0.5, 0.5, 'No test data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=24)
        return
    
    # Find control-baseline test
    control_baseline_test = tests_df[tests_df['test'].str.contains('Control-baseline', case=False, na=False)]
    
    # Find original treatment differential effect test
    original_test = tests_df[tests_df['test'].str.contains('Treatment differential effect.*coding vs intergenic', case=False, na=False)]
    
    # If we only have one test, still show it but explain what's missing
    if control_baseline_test.empty and original_test.empty:
        ax.text(0.5, 0.5, 'No comparison data available\n(Run estimate_rates.py with use_control_baseline=True)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=20)
        return
    
    # Prepare data for comparison - always show both if available
    comparison_data = []
    
    if not control_baseline_test.empty:
        row = control_baseline_test.iloc[0]
        comparison_data.append({
            'Approach': 'Control-Baseline',
            'Test': 'Treatment effect constant\nacross categories?',
            'p_value': row['p_value'],
            'rate_ratio': row.get('rate_ratio', np.nan),
            'interpretation': 'p > 0.05: category does NOT matter'
        })
    else:
        # Add placeholder if missing
        comparison_data.append({
            'Approach': 'Control-Baseline',
            'Test': 'Not available',
            'p_value': np.nan,
            'rate_ratio': np.nan,
            'interpretation': 'Run with use_control_baseline=True'
        })
    
    if not original_test.empty:
        row = original_test.iloc[0]
        comparison_data.append({
            'Approach': 'Original\n(with interactions)',
            'Test': 'Treatment differential effect:\ncoding vs intergenic',
            'p_value': row['p_value'],
            'rate_ratio': row.get('rate_ratio', np.nan),
            'interpretation': 'Tests if treatment affects\ncoding differently than intergenic'
        })
    else:
        # Add placeholder if missing
        comparison_data.append({
            'Approach': 'Original\n(with interactions)',
            'Test': 'Not available',
            'p_value': np.nan,
            'rate_ratio': np.nan,
            'interpretation': 'Missing from test results'
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create bar plot comparing p-values
    x_pos = np.arange(len(comp_df))
    colors = []
    for i, approach in enumerate(comp_df['Approach']):
        if 'Control-Baseline' in approach:
            colors.append(TREATED_COLORS[0])  # Light orange
        else:
            colors.append(TREATED_COLORS[2])  # Dark red
    
    # Plot p-values (use -log10 for better visualization)
    p_values = comp_df['p_value'].values
    # Handle NaN values
    valid_p = [p if not pd.isna(p) else 1e-10 for p in p_values]
    neg_log10_p = -np.log10(np.maximum(valid_p, 1e-10))  # Avoid log(0)
    
    bars = ax.bar(x_pos, neg_log10_p, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add significance threshold line at p=0.05
    threshold = -np.log10(0.05)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
               label='p=0.05 threshold', alpha=0.7)
    
    # Add text annotations on bars
    for i, (bar, p_val, rate_ratio, approach) in enumerate(zip(bars, p_values, comp_df['rate_ratio'], comp_df['Approach'])):
        height = bar.get_height()
        
        if not pd.isna(p_val):
            # Add p-value text
            p_text = f'p={p_val:.3e}' if p_val < 0.001 else f'p={p_val:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   p_text, ha='center', va='bottom', fontsize=18, fontweight='bold')
            
            # Add rate ratio if available
            if not pd.isna(rate_ratio):
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'Rate ratio:\n{rate_ratio:.3f}', ha='center', va='center', 
                       fontsize=16, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            # Show "N/A" for missing data
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   'N/A', ha='center', va='center', 
                   fontsize=18, fontweight='bold', style='italic',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    
    # Format x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comp_df['Approach'], fontsize=20, fontweight='bold', rotation=0, ha='center')
    
    # Format y-axis
    ax.set_ylabel('-log₁₀(p-value)', fontsize=26, fontweight='bold', labelpad=20)
    if any(~pd.isna(p_values)):
        max_p = max([p for p in p_values if not pd.isna(p)])
        max_neg_log = -np.log10(max_p) if max_p > 0 else 1
        ax.set_ylim(bottom=0, top=max_neg_log * 1.4)
    else:
        ax.set_ylim(bottom=0, top=5)
    ax.tick_params(axis='y', labelsize=20, pad=12)
    ax.tick_params(axis='x', labelsize=20, pad=14)
    
    # Add interpretation text for control-baseline if available
    if not control_baseline_test.empty:
        row = control_baseline_test.iloc[0]
        p_val = row['p_value']
        if p_val > 0.05:
            interpretation = "✓ Category does NOT matter\nfor treatment effect"
            text_color = 'green'
        else:
            interpretation = "✗ Category DOES matter\nfor treatment effect"
            text_color = 'red'
        
        ax.text(0.5, 0.95, interpretation, transform=ax.transAxes,
               ha='center', va='top', fontsize=22, fontweight='bold', color=text_color,
               bbox=dict(boxstyle='round,pad=1.0', facecolor='white', alpha=0.9, 
                        edgecolor=text_color, linewidth=2))
    
    # Add legend
    ax.legend(loc='upper right', fontsize=18, frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_title('Model Comparison:\nControl-Baseline vs Original', 
                fontsize=28, fontweight='bold', pad=20)


def plot_category_significances(ax, category_df, tests_df):
    """Plot absolute EMS effects by category (from clean absolute analysis).
    
    Shows:
    - Bar plot of absolute EMS effects (mutations/base added by EMS) by category
    - This directly answers: Does EMS add the same absolute mutations/base to intergenic vs genic?
    """
    if category_df is None or category_df.empty:
        ax.text(0.5, 0.5, 'No category data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=24)
        return
    
    # Check if we have absolute effect data from clean absolute analysis
    has_absolute_effects = 'absolute_effect' in category_df.columns and category_df['clean_absolute_analysis'].any() if 'clean_absolute_analysis' in category_df.columns else False
    
    # Get treated and control data
    treated_df = category_df[category_df['treatment'] == 'treated'].copy()
    control_df = category_df[category_df['treatment'] == 'control'].copy()
    
    if treated_df.empty:
        ax.text(0.5, 0.5, 'No treated category data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=24)
        return
    
    # Remove coding category from the plot
    treated_df = treated_df[treated_df['category'] != 'coding'].copy()
    control_df = control_df[control_df['category'] != 'coding'].copy()
    
    if treated_df.empty:
        ax.text(0.5, 0.5, 'No category data available (after filtering)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=24)
        return
    
    # Sort both dataframes by category to ensure alignment
    category_order = ['intergenic', 'synonymous', 'non_synonymous']
    treated_df['cat_order'] = treated_df['category'].apply(lambda x: category_order.index(x) if x in category_order else 99)
    control_df['cat_order'] = control_df['category'].apply(lambda x: category_order.index(x) if x in category_order else 99)
    treated_df = treated_df.sort_values('cat_order')
    control_df = control_df.sort_values('cat_order')
    
    # Get categories present
    categories = treated_df['category'].tolist()
    n_categories = len(categories)
    
    # Set up grouped bar positions
    bar_width = 0.35
    group_spacing = 1.0
    x_pos = np.arange(n_categories) * group_spacing
    
    # Colors
    control_color = NT_GREY
    treated_color = TREATED_COLORS[1]  # Orange-red
    
    # Plot control bars (left side of each group)
    control_rates = []
    for cat in categories:
        cat_data = control_df[control_df['category'] == cat]
        if not cat_data.empty:
            control_rates.append(cat_data['rate'].values[0])
        else:
            control_rates.append(np.nan)
    
    control_rates = np.array(control_rates)
    bars_ctrl = ax.bar(x_pos - bar_width/2, control_rates, bar_width, 
                      color=control_color, alpha=0.8,
                      edgecolor='black', linewidth=1.5, label='Control')
    
    # Add error bars for control if available
    if 'CI_low' in control_df.columns and 'CI_high' in control_df.columns:
        control_ci_low = []
        control_ci_high = []
        for cat in categories:
            cat_data = control_df[control_df['category'] == cat]
            if not cat_data.empty:
                control_ci_low.append(cat_data['CI_low'].values[0])
                control_ci_high.append(cat_data['CI_high'].values[0])
            else:
                control_ci_low.append(np.nan)
                control_ci_high.append(np.nan)
        
        if not all(np.isnan(control_ci_low)):
            yerr_low = control_rates - np.array(control_ci_low)
            yerr_high = np.array(control_ci_high) - control_rates
            ax.errorbar(x_pos - bar_width/2, control_rates,
                       yerr=[yerr_low, yerr_high], fmt='none',
                       color='black', capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8)
    
    # Plot treated bars (right side of each group)
    treated_rates = treated_df['rate'].values
    bars_treated = ax.bar(x_pos + bar_width/2, treated_rates, bar_width,
                         color=treated_color, alpha=0.8,
                         edgecolor='black', linewidth=1.5, label='Treated')
    
    # Add error bars for treated if available
    if 'CI_low' in treated_df.columns and 'CI_high' in treated_df.columns:
        yerr_low = treated_rates - treated_df['CI_low'].values
        yerr_high = treated_df['CI_high'].values - treated_rates
        ax.errorbar(x_pos + bar_width/2, treated_rates,
                   yerr=[yerr_low, yerr_high], fmt='none',
                   color='black', capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8)
    
    # Set y-axis range to clearly show controls (~10^-5) and treated (~10^-4)
    all_rates = list(treated_rates)
    all_rates.extend([r for r in control_rates if not np.isnan(r)])
    min_rate = min(all_rates)
    max_rate = max(all_rates)
    
    # Format y-axis (linear scale) - shows controls are small, treated are larger
    # The gaps between them (absolute effects) will look similar across categories
    ax.set_ylabel('Mutation Rate (per base)', fontsize=32, fontweight='bold', labelpad=25)
    
    # Set y-axis limits to span from 0 to above treated rates
    y_min = 0
    y_max = 3e-4  # Round number above max treated rate
    ax.set_ylim(y_min, y_max)
    
    # Set explicit ticks at nice round numbers
    tick_values = [0, 5e-5, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4]
    ax.set_yticks(tick_values)
    
    # Use ScalarFormatter for scientific notation
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    
    # Add legend in top right
    ax.legend(loc='upper right', fontsize=28, frameon=True, fancybox=True, 
              shadow=True, framealpha=0.95)
    
    # Format category labels (works for both absolute effects and rates)
    category_labels = {
        'intergenic': 'Intergenic',
        'synonymous': 'Synonymous',
        'non_synonymous': 'Non-syn',
        'coding': 'Coding'
    }
    labels = [category_labels.get(cat, cat) for cat in categories]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=28, fontweight='bold', rotation=0, ha='center')
    
    # Set x-axis limits
    if len(x_pos) > 0:
        padding = 0.5
        ax.set_xlim(left=x_pos[0] - padding, right=x_pos[-1] + padding)
    
    ax.tick_params(axis='y', labelsize=28, pad=12)
    ax.tick_params(axis='x', labelsize=28, pad=14)
    
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # No title for panel 2


def plot_treatment_day_rates(ax, summary_df, per_sample_df, tests_df, sample_color_dict=None):
    """Plot rates per treatment time group with scatter of individual samples and significance markers.
    
    Args:
        ax: Matplotlib axis
        summary_df: DataFrame with aggregate rates per treatment day
        per_sample_df: DataFrame with per-sample rates
        tests_df: DataFrame with statistical tests
        sample_color_dict: Optional dict mapping sample names to colors (from first plot)
    """
    if summary_df is None or summary_df.empty:
        ax.text(0.5, 0.5, 'No treatment day data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=24)
        return
    
    # Sort by rate (lowest to highest)
    summary_df = summary_df.copy()
    summary_df = summary_df[summary_df['treatment_day'].notna()]
    summary_df = summary_df.sort_values('rate', ascending=True)
    
    # Normalize treatment_day values for consistent ordering
    summary_df['treatment_day_normalized'] = summary_df['treatment_day'].str.lower().str.replace('_', '')
    
    # Create x positions for each treatment day
    day_order = summary_df['treatment_day_normalized'].tolist()
    x_positions = {day: i for i, day in enumerate(day_order)}
    
    # Map days to colors - use plot_ems_spectra color scheme
    # For aggregate diamonds, use orange-to-red gradient matching plot_ems_spectra
    color_map = {
        'control': NT_GREY,  # Grey for controls
        'nolabel': TREATED_COLORS[2],  # Dark red
        '3d': TREATED_COLORS[0],  # Light orange
        '7d': TREATED_COLORS[1],  # Orange-red
    }
    
    # Plot individual samples as scatter if available
    if per_sample_df is not None and not per_sample_df.empty:
        per_sample_df = per_sample_df.copy()
        per_sample_df['treatment_day_normalized'] = per_sample_df['treatment_day'].str.lower().str.replace('_', '')
        
        # Add jitter to x positions for scatter
        np.random.seed(42)  # For reproducibility
        for day in day_order:
            day_samples = per_sample_df[per_sample_df['treatment_day_normalized'] == day]
            if not day_samples.empty:
                x_base = x_positions[day]
                
                # Use sample-specific colors if provided, otherwise use day color
                for idx, row in day_samples.iterrows():
                    sample_name = str(row.get('sample', ''))
                    if sample_color_dict and sample_name in sample_color_dict:
                        color = sample_color_dict[sample_name]
                    else:
                        color = color_map.get(day, NT_GREY)  # Use NT_GREY as fallback
                    
                    # Add small random jitter for each point
                    x_jitter = x_base + np.random.normal(0, 0.08, 1)[0]
                    ax.scatter(x_jitter, row['rate'], 
                              color=color, alpha=0.5, s=40, edgecolors='black', linewidth=0.5, zorder=2)
    
    # Plot aggregate rates as large points with error bars
    for idx, row in summary_df.iterrows():
        day = row['treatment_day_normalized']
        x_pos = x_positions[day]
        rate = row['rate']
        color = color_map.get(day, '#808080')
        
        # Plot aggregate as large point
        ax.scatter(x_pos, rate, color=color, s=200, edgecolors='black', 
                  linewidth=2, zorder=3, marker='D')  # Diamond marker for aggregate
        
        # Add error bars if available
        if 'CI_low' in row and 'CI_high' in row:
            yerr_low = rate - row['CI_low']
            yerr_high = row['CI_high'] - rate
            ax.errorbar(x_pos, rate,
                       yerr=[[yerr_low], [yerr_high]], fmt='none',
                       color='black', capsize=6, capthick=2, elinewidth=2, alpha=0.8, zorder=3)
    
    # Format labels first (before getting y limits)
    day_labels = {
        'control': 'Control',
        'nolabel': '5 Days',
        '3d': '3 Days',
        '7d': '7 Days'
    }
    labels = [day_labels.get(day, summary_df.loc[idx, 'treatment_day']) 
              for idx, day in summary_df['treatment_day_normalized'].items()]
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(labels, fontsize=20, fontweight='bold', rotation=0, ha='center')
    
    # Ensure x-axis limits have padding so there's never just one tick label visible
    if len(x_positions) > 0:
        x_ticks = list(x_positions.values())
        x_min = min(x_ticks)
        x_max = max(x_ticks)
        # Add padding: at least 0.5 units on each side, or scale with range
        padding = max(0.5, (x_max - x_min) * 0.3) if x_max > x_min else 0.5
        ax.set_xlim(left=x_min - padding, right=x_max + padding)
    
    # Format y-axis with more ticks
    ax.set_ylabel('Mutation Rate (per base)', fontsize=32, fontweight='bold', labelpad=30)
    ax.set_yscale('log')
    
    # Set explicit y-axis ticks for log scale
    ax.set_yticks([3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4])
    
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    
    # Add minor ticks
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
    
    ax.tick_params(axis='y', labelsize=28, pad=15)
    ax.tick_params(axis='x', labelsize=28, pad=16)
    
    # Add significance annotations using letter-based system (cleaner for multiple groups)
    if tests_df is not None and not tests_df.empty:
        # Get max rate for positioning
        all_rates = list(summary_df['rate'].values)
        if per_sample_df is not None and not per_sample_df.empty:
            all_rates.extend(per_sample_df['rate'].values)
        max_rate = max([r for r in all_rates if np.isfinite(r) and r > 0]) if all_rates else 1e-6
        
        # Build significance matrix for treated days only
        treated_days = [day for day in x_positions.keys() if day != 'control']
        significance_matrix = {}  # (day1, day2) -> (is_significant, p_value)
        
        for _, test_row in tests_df.iterrows():
            test_name = test_row['test']
            p_val = test_row['p_value']
            
            # Only include treated day comparisons
            if ' vs ' in test_name and 'control' not in test_name.lower():
                parts = test_name.split(' vs ')
                if len(parts) == 2:
                    day1 = parts[0].strip().lower().replace('_', '').replace(' ', '')
                    day2 = parts[1].strip().lower().replace('_', '').replace(' ', '')
                    # Map variations
                    day1 = day1.replace('nolabel', 'nolabel').replace('no_label', 'nolabel')
                    day2 = day2.replace('nolabel', 'nolabel').replace('no_label', 'nolabel')
                    if day1 in x_positions and day2 in x_positions:
                        # Store both directions
                        significance_matrix[(day1, day2)] = (p_val < 0.05, p_val)
                        significance_matrix[(day2, day1)] = (p_val < 0.05, p_val)
        
        # Add compact p-value table in upper right corner (same style as center plot)
        if significance_matrix:
            # Collect all unique comparisons
            comparisons = []
            seen_pairs = set()
            for (day1, day2), (is_sig, p_val) in significance_matrix.items():
                pair = tuple(sorted([day1, day2]))
                if pair not in seen_pairs and day1 in treated_days and day2 in treated_days:
                    seen_pairs.add(pair)
                    symbol, p_str = format_pvalue(p_val)
                    day1_label = {'nolabel': '5 Days', '3d': '3 Days', '7d': '7 Days'}.get(day1, day1)
                    day2_label = {'nolabel': '5 Days', '3d': '3 Days', '7d': '7 Days'}.get(day2, day2)
                    comparisons.append((day1_label, day2_label, symbol, p_str, p_val))
            
            # Sort by p-value
            comparisons.sort(key=lambda x: x[4])
            
            # Display as compact text in bottom right (smaller font and padding)
            table_text = "Comparisons:\n"
            for day1_label, day2_label, symbol, p_str, _ in comparisons[:4]:  # Limit to 4 comparisons
                table_text += f"{day1_label} vs {day2_label}: {p_str}\n"
            
            ax.text(0.98, 0.02, table_text, transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=14,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=1.5),
                   family='monospace')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')


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


def load_gene_info_cache(cache_file):
    """Load gene information from JSON cache file.
    
    Returns:
        dict mapping gene_id (WD_RS format) to dict with 'name' and 'description'
    """
    if not cache_file or not os.path.exists(cache_file):
        return {}
    
    gene_info = {}
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Extract gene info - cache uses UID as key, need to find WD_RS IDs in otheraliases
        for uid, gene_data in cache_data.items():
            name = gene_data.get('name', '')
            description = gene_data.get('description', '')
            otheraliases = gene_data.get('otheraliases', '')
            
            # Extract WD_RS IDs from otheraliases (format: "WD_RS05860, WD1295, WD_1295")
            if otheraliases:
                aliases = [a.strip() for a in otheraliases.split(',')]
                for alias in aliases:
                    if alias.startswith('WD_RS'):
                        # Store info keyed by WD_RS ID
                        gene_info[alias] = {
                            'name': name,
                            'description': description
                        }
                        # Also store with "gene-" prefix for compatibility
                        gene_info[f'gene-{alias}'] = {
                            'name': name,
                            'description': description
                        }
    except Exception as e:
        print(f"Warning: Could not load gene info cache: {e}")
        return {}
    
    return gene_info


def extract_wd_rs_id(gene_id_str):
    """Extract WD_RS ID from gene identifier string.
    Handles formats like "gene-WD_RS00015" -> "WD_RS00015"
    """
    if pd.isna(gene_id_str) or not gene_id_str:
        return None
    gene_id_str = str(gene_id_str)
    if gene_id_str.startswith('gene-'):
        return gene_id_str[5:]  # Remove "gene-" prefix
    return gene_id_str


def analyze_genes_in_rate_windows(windows_df, gff_file, gene_info_cache, rate_col, n_windows=10, highest=True):
    """
    Analyze genes in windows with highest or lowest mutation rates.
    
    Args:
        windows_df: DataFrame with window rates
        gff_file: Path to GFF file with gene annotations
        gene_info_cache: Dict mapping gene IDs to name/description
        rate_col: Column name for rate values
        n_windows: Number of top/bottom windows to analyze
        highest: If True, analyze highest rate windows; if False, analyze lowest
    
    Returns:
        DataFrame with window info and gene details
    """
    if gff_file is None or not os.path.exists(gff_file):
        return pd.DataFrame()
    
    # Load genes from GFF file
    try:
        gene_regions = parse_gff(gff_file)
        print(f"Loaded {len(gene_regions)} genes from GFF file")
    except Exception as e:
        print(f"Error loading GFF file: {e}")
        return pd.DataFrame()
    
    # Get top/bottom windows by rate
    valid_windows = windows_df[windows_df[rate_col].notna() & (windows_df[rate_col] > 0)].copy()
    if len(valid_windows) == 0:
        return pd.DataFrame()
    
    if highest:
        top_windows = valid_windows.nlargest(n_windows, rate_col)
        window_type = 'highest'
    else:
        top_windows = valid_windows.nsmallest(n_windows, rate_col)
        window_type = 'lowest'
    
    print(f"\nAnalyzing genes in {n_windows} {window_type} rate windows...")
    
    # Organize genes by chromosome
    genes_by_chrom = defaultdict(list)
    for chrom, start, end, gene_id in gene_regions:
        genes_by_chrom[str(chrom)].append((start, end, gene_id))
    
    # Analyze each window
    results = []
    for idx, window_row in top_windows.iterrows():
        chrom = str(window_row['chrom'])
        window_start = int(window_row['start'])
        window_end = int(window_row['end'])
        window_rate = window_row[rate_col]
        
        # Find overlapping genes
        overlapping_genes = []
        if chrom in genes_by_chrom:
            for gene_start, gene_end, gene_id in genes_by_chrom[chrom]:
                # Check if gene overlaps with window
                overlap_start = max(window_start, gene_start)
                overlap_end = min(window_end, gene_end)
                if overlap_start <= overlap_end:
                    # Calculate coverage percentage
                    gene_length = gene_end - gene_start + 1
                    overlap_length = overlap_end - overlap_start + 1
                    coverage_pct = (overlap_length / gene_length * 100) if gene_length > 0 else 0
                    
                    # Extract WD_RS ID
                    wd_rs_id = extract_wd_rs_id(gene_id)
                    
                    # Get gene name and description from cache
                    gene_name = ''
                    gene_description = ''
                    if wd_rs_id:
                        # Try direct match first
                        if wd_rs_id in gene_info_cache:
                            gene_name = gene_info_cache[wd_rs_id].get('name', '')
                            gene_description = gene_info_cache[wd_rs_id].get('description', '')
                        # Try with gene- prefix
                        elif f'gene-{wd_rs_id}' in gene_info_cache:
                            gene_name = gene_info_cache[f'gene-{wd_rs_id}'].get('name', '')
                            gene_description = gene_info_cache[f'gene-{wd_rs_id}'].get('description', '')
                    
                    overlapping_genes.append({
                        'gene_id': gene_id,
                        'wd_rs_id': wd_rs_id if wd_rs_id else gene_id,
                        'gene_name': gene_name,
                        'gene_description': gene_description,
                        'gene_start': gene_start,
                        'gene_end': gene_end,
                        'overlap_start': overlap_start,
                        'overlap_end': overlap_end,
                        'coverage_percentage': coverage_pct
                    })
        
        # Create result row
        if overlapping_genes:
            for gene_info in overlapping_genes:
                results.append({
                    'window_index': idx,
                    'chrom': chrom,
                    'window_start': window_start,
                    'window_end': window_end,
                    'window_rate': window_rate,
                    'gene_id': gene_info['gene_id'],
                    'wd_rs_id': gene_info['wd_rs_id'],
                    'gene_name': gene_info['gene_name'],
                    'gene_description': gene_info['gene_description'],
                    'gene_start': gene_info['gene_start'],
                    'gene_end': gene_info['gene_end'],
                    'overlap_start': gene_info['overlap_start'],
                    'overlap_end': gene_info['overlap_end'],
                    'coverage_percentage': gene_info['coverage_percentage']
                })
        else:
            # Window with no genes
            results.append({
                'window_index': idx,
                'chrom': chrom,
                'window_start': window_start,
                'window_end': window_end,
                'window_rate': window_rate,
                'gene_id': 'none',
                'wd_rs_id': 'none',
                'gene_name': '',
                'gene_description': '',
                'gene_start': np.nan,
                'gene_end': np.nan,
                'overlap_start': np.nan,
                'overlap_end': np.nan,
                'coverage_percentage': 0.0
            })
    
    result_df = pd.DataFrame(results)
    print(f"Found {len([r for r in results if r['gene_id'] != 'none'])} gene hits in {window_type} rate windows")
    
    return result_df


def find_origin_of_replication(gff_file, oriC_position=None, oriC_chrom=None):
    """
    Find the origin of replication by locating dnaA gene or using specified position.
    In bacterial genomes, oriC is typically located near the start of dnaA.
    
    Args:
        gff_file: Path to GFF file
        oriC_position: Explicit oriC position (center or start coordinate)
        oriC_chrom: Chromosome name for oriC (required if oriC_position is provided)
    
    Returns:
        dict with 'chrom', 'position' (oriC location), 'dnaA_start', 'dnaA_end', or None if not found
    """
    # If explicit oriC position provided, use that
    if oriC_position is not None:
        if oriC_chrom is None:
            # Try to get first chromosome from GFF if not specified
            if gff_file and os.path.exists(gff_file):
                try:
                    with open(gff_file) as f:
                        for line in f:
                            if not line.startswith('#') and line.strip():
                                fields = line.strip().split('\t')
                                if len(fields) >= 1:
                                    oriC_chrom = fields[0]
                                    break
                except Exception:
                    pass
        
        if oriC_chrom:
            return {
                'chrom': oriC_chrom,
                'position': oriC_position,
                'dnaA_start': None,
                'dnaA_end': None
            }
    
    if not gff_file or not os.path.exists(gff_file):
        return None
    
    try:
        with open(gff_file) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                fields = line.strip().split('\t')
                if len(fields) >= 9 and fields[2] == 'gene':
                    # Check if this is dnaA gene
                    attributes = fields[8]
                    if 'gene=dnaA' in attributes or 'Name=dnaA' in attributes:
                        chrom = fields[0]
                        start = int(fields[3])
                        end = int(fields[4])
                        # Origin is typically just before or at the start of dnaA
                        # For circular genomes, if dnaA is near position 1, oriC is at position 1
                        # Otherwise, oriC is typically ~100-200 bp before dnaA start
                        if start < 1000:  # dnaA near start of chromosome
                            oriC_pos = 1
                        else:
                            oriC_pos = max(1, start - 150)  # ~150 bp before dnaA
                        
                        return {
                            'chrom': chrom,
                            'position': oriC_pos,
                            'dnaA_start': start,
                            'dnaA_end': end
                        }
    except Exception as e:
        print(f"Error finding origin of replication: {e}")
    
    return None


def analyze_genes_in_clusters(clusters, windows_df, gff_file, min_gene_coverage=50.0, report_all_overlaps=False):
    """
    Analyze genes within hotspot clusters.
    
    Args:
        clusters: List of cluster dictionaries (from identify_significant_clusters)
        windows_df: DataFrame with window data
        gff_file: Path to GFF file with gene annotations
        min_gene_coverage: Minimum percentage of gene that must be in cluster (default: 50%)
    
    Returns:
        DataFrame with columns: cluster_id, chrom, cluster_start, cluster_end, 
        n_windows, genes_included, gene_coverage_percentage
    """
    if not clusters or not gff_file or not os.path.exists(gff_file):
        return pd.DataFrame()
    
    print(f"Analyzing genes in {len(clusters)} clusters...")
    
    # Load genes from GFF file
    try:
        gene_regions = parse_gff(gff_file)
        print(f"Loaded {len(gene_regions)} genes from GFF file")
    except Exception as e:
        print(f"Error loading GFF file: {e}")
        return pd.DataFrame()
    
    # Organize genes by chromosome for faster lookup
    # Normalize chromosome names (handle variations like "chr1" vs "1", etc.)
    genes_by_chrom = defaultdict(list)
    chrom_variants = defaultdict(set)  # Track all chromosome name variants
    
    for chrom, start, end, gene_id in gene_regions:
        chrom_str = str(chrom)
        genes_by_chrom[chrom_str].append((start, end, gene_id))
        chrom_variants[chrom_str].add(chrom_str)
        # Also add normalized versions
        chrom_normalized = chrom_str.replace('chr', '').replace('Chr', '').replace('CHR', '')
        if chrom_normalized != chrom_str:
            chrom_variants[chrom_normalized].add(chrom_str)
    
    print(f"Chromosome names in GFF: {sorted(list(genes_by_chrom.keys()))[:10]}...")  # Show first 10
    
    # Analyze each hotspot cluster
    cluster_gene_hits = []
    
    for i, cluster in enumerate(clusters):
        # Only analyze hotspot clusters
        if cluster['type'] != 'hotspot':
            continue
        
        chrom = str(cluster['chrom'])
        cluster_start = cluster['start']
        cluster_end = cluster['end']
        cluster_span = cluster['genomic_span']
        n_windows = cluster['n_windows']
        
        print(f"\nAnalyzing cluster {i+1}: {chrom}:{cluster_start}-{cluster_end} (span={cluster_span}bp, {n_windows} windows)")
        
        # Find overlapping genes - try exact match and normalized versions
        overlapping_genes = []
        coverage_percentages = []
        genes_to_check = []
        
        # Try exact chromosome match first
        if chrom in genes_by_chrom:
            genes_to_check = genes_by_chrom[chrom]
            print(f"  Found {len(genes_to_check)} genes on chromosome {chrom}")
        else:
            # Try normalized version
            chrom_normalized = chrom.replace('chr', '').replace('Chr', '').replace('CHR', '')
            if chrom_normalized in genes_by_chrom:
                genes_to_check = genes_by_chrom[chrom_normalized]
                print(f"  Found {len(genes_to_check)} genes on normalized chromosome {chrom_normalized}")
            else:
                # Try reverse - add 'chr' prefix
                chrom_with_chr = f"chr{chrom}" if not chrom.startswith('chr') else chrom
                if chrom_with_chr in genes_by_chrom:
                    genes_to_check = genes_by_chrom[chrom_with_chr]
                    print(f"  Found {len(genes_to_check)} genes on chromosome {chrom_with_chr}")
                else:
                    print(f"  WARNING: No genes found for chromosome {chrom} (tried: {chrom}, {chrom_normalized}, {chrom_with_chr})")
        
        genes_checked = 0
        genes_overlapping = 0
        
        for gene_start, gene_end, gene_id in genes_to_check:
            genes_checked += 1
            # Calculate overlap between gene and cluster
            overlap_start = max(cluster_start, gene_start)
            overlap_end = min(cluster_end, gene_end)
            overlap_length = max(0, overlap_end - overlap_start + 1)
            
            if overlap_length > 0:
                genes_overlapping += 1
                # Calculate what percentage of the gene is within the cluster
                gene_length = gene_end - gene_start + 1
                if gene_length > 0:
                    gene_coverage_pct = (overlap_length / gene_length) * 100
                    
                    # Include genes based on threshold or if reporting all overlaps
                    if report_all_overlaps or gene_coverage_pct >= min_gene_coverage:
                        overlapping_genes.append(gene_id)
                        coverage_percentages.append(gene_coverage_pct)
                        print(f"    Gene {gene_id}: {gene_start}-{gene_end} ({gene_length}bp) -> {gene_coverage_pct:.1f}% coverage")
        
        print(f"  Checked {genes_checked} genes, {genes_overlapping} overlapping, {len(overlapping_genes)} with >= {min_gene_coverage}% coverage")
        
        # Format genes and coverage percentages
        if overlapping_genes:
            genes_str = ','.join(overlapping_genes)
            coverage_str = ','.join([f'{pct:.2f}' for pct in coverage_percentages])
        else:
            genes_str = 'none'
            coverage_str = '0.00'
        
        cluster_gene_hits.append({
            'cluster_id': f"hotspot_cluster_{i+1}",
            'chrom': chrom,
            'cluster_start': cluster_start,
            'cluster_end': cluster_end,
            'cluster_span_bp': cluster_span,
            'n_windows': n_windows,
            'genes_included': genes_str,
            'gene_coverage_percentage': coverage_str,
            'n_genes': len(overlapping_genes)
        })
    
    result_df = pd.DataFrame(cluster_gene_hits)
    print(f"Gene analysis complete: {len(result_df)} hotspot clusters analyzed")
    if len(result_df) > 0:
        total_genes = result_df['n_genes'].sum()
        print(f"  Found {total_genes} genes with >= {min_gene_coverage}% coverage in clusters")
    
    return result_df


def identify_significant_clusters(
    windows_df: pd.DataFrame,
    chrom_order: list,
    chrom_offsets: dict,
    rate_col: str,
    overall_mean: float,
    min_cluster_size: int = 3,
    max_gap: int = 2,
    cluster_p_threshold: float = 0.05
):
    """
    Identify significant clusters of hotspots or coldspots.
    
    Uses a scan statistic approach:
    1. Find contiguous regions of significant windows
    2. Test each cluster using a binomial test (observed vs expected significant windows)
    3. Return clusters that pass significance threshold
    
    Args:
        windows_df: DataFrame with window data
        chrom_order: Ordered list of chromosomes
        chrom_offsets: Dictionary mapping chromosome to cumulative offset
        rate_col: Column name for rate values
        overall_mean: Overall mean rate for comparison
        min_cluster_size: Minimum number of windows in a cluster
        max_gap: Maximum gap (in windows) allowed within a cluster
        cluster_p_threshold: P-value threshold for cluster significance
    
    Returns:
        List of cluster dictionaries with keys: type, chrom, start, end, n_windows, 
        window_indices, p_value, mean_rate
    """
    clusters = []
    
    # Process each chromosome separately
    for chrom in chrom_order:
        chrom_windows = windows_df[windows_df['chrom'] == chrom].copy()
        if len(chrom_windows) == 0:
            continue
        
        # Sort by position and preserve original index
        chrom_windows = chrom_windows.sort_values('start')
        original_indices = chrom_windows.index.values  # Store original DataFrame indices
        
        # Identify significant windows
        sig_mask = chrom_windows['is_significant'].values
        is_hotspot = (chrom_windows[rate_col] > overall_mean).values
        is_coldspot = (chrom_windows[rate_col] < overall_mean).values
        
        # Find contiguous clusters of significant windows
        current_cluster = None
        for i in range(len(chrom_windows)):
            if sig_mask[i]:
                cluster_type = 'hotspot' if is_hotspot[i] else 'coldspot'
                
                if current_cluster is None:
                    # Start new cluster
                    current_cluster = {
                        'type': cluster_type,
                        'indices': [i],
                        'start_idx': i
                    }
                elif (current_cluster['type'] == cluster_type and 
                      i - current_cluster['indices'][-1] <= max_gap + 1):
                    # Extend cluster (within max_gap)
                    current_cluster['indices'].append(i)
                else:
                    # End current cluster and start new one
                    if len(current_cluster['indices']) >= min_cluster_size:
                        clusters.append(_finalize_cluster(
                            current_cluster, chrom_windows, chrom, rate_col, 
                            len(chrom_windows), sig_mask.sum(), original_indices
                        ))
                    current_cluster = {
                        'type': cluster_type,
                        'indices': [i],
                        'start_idx': i
                    }
            else:
                # Gap in significant windows
                if current_cluster is not None:
                    # Check if gap is too large
                    if i - current_cluster['indices'][-1] > max_gap:
                        # End cluster
                        if len(current_cluster['indices']) >= min_cluster_size:
                            clusters.append(_finalize_cluster(
                                current_cluster, chrom_windows, chrom, rate_col,
                                len(chrom_windows), sig_mask.sum(), original_indices
                            ))
                        current_cluster = None
        
        # Handle cluster at end of chromosome
        if current_cluster is not None and len(current_cluster['indices']) >= min_cluster_size:
            clusters.append(_finalize_cluster(
                current_cluster, chrom_windows, chrom, rate_col,
                len(chrom_windows), sig_mask.sum(), original_indices
            ))
    
    # Filter clusters by significance
    significant_clusters = [c for c in clusters if c['p_value'] < cluster_p_threshold]
    
    print(f"Found {len(clusters)} clusters (min_size={min_cluster_size}), "
          f"{len(significant_clusters)} significant (p<{cluster_p_threshold})")
    
    return significant_clusters


def _finalize_cluster(cluster, chrom_windows, chrom, rate_col, total_windows, total_sig, original_indices):
    """Finalize cluster information and calculate significance."""
    indices = cluster['indices']  # These are positions in the sorted chrom_windows
    cluster_windows = chrom_windows.iloc[indices]  # Use iloc for positional indexing
    # Map to original DataFrame indices
    original_cluster_indices = [original_indices[i] for i in indices]
    
    # Calculate cluster statistics
    n_cluster_sig = len(indices)  # Number of significant windows in cluster
    cluster_start = cluster_windows['start'].min()
    cluster_end = cluster_windows['end'].max()
    mean_rate = cluster_windows[rate_col].mean()
    
    # Count total windows in cluster region (including non-significant windows in gaps)
    # Find all windows that fall within the cluster span
    cluster_region_mask = ((chrom_windows['start'] >= cluster_start) & 
                           (chrom_windows['end'] <= cluster_end))
    n_cluster_total = cluster_region_mask.sum()  # Total windows in cluster region
    
    # Test cluster significance using binomial test
    # Null hypothesis: cluster is random (proportion of significant windows matches genome-wide)
    # Alternative: cluster has more significant windows than expected by chance
    p_sig_genome = total_sig / total_windows if total_windows > 0 else 0
    
    # Binomial test: probability of observing n_cluster_sig significant windows
    # out of n_cluster_total total windows in the cluster region, given genome-wide proportion
    if p_sig_genome > 0 and p_sig_genome < 1 and n_cluster_total > 0 and n_cluster_sig > 0:
        # Use binomtest for newer scipy, fallback to binom_test for older versions
        try:
            from scipy.stats import binomtest
            # binomtest(k, n, p, alternative='greater')
            # k = number of successes (significant windows), n = total trials (total windows in region)
            result = binomtest(n_cluster_sig, n_cluster_total, p_sig_genome, alternative='greater')
            p_value = result.pvalue
        except (ImportError, AttributeError):
            # Fallback for older scipy versions
            try:
                p_value = stats.binom_test(n_cluster_sig, n_cluster_total, p_sig_genome, alternative='greater')
            except (AttributeError, TypeError):
                # Manual calculation if binom_test not available
                from scipy.stats import binom
                p_value = 1 - binom.cdf(n_cluster_sig - 1, n_cluster_total, p_sig_genome)
    else:
        # If all or no windows are significant, or no windows in cluster region, can't test
        p_value = 1.0
    
    return {
        'type': cluster['type'],
        'chrom': chrom,
        'start': int(cluster_start),
        'end': int(cluster_end),
        'n_windows': n_cluster_sig,  # Number of significant windows in cluster
        'window_indices': original_cluster_indices,  # Use original DataFrame indices
        'p_value': p_value,
        'mean_rate': mean_rate,
        'genomic_span': int(cluster_end - cluster_start)
    }


def plot_regional_rates_manhattan(
    windows_rates_df: pd.DataFrame,
    output_path: str,
    use_treatment_covariate: bool = True,
    significance_threshold: float = 0.05,
    gff_file: str = None,
    rate_threshold_multiplier: float = 2.0,
    min_cluster_size: int = 3,
    max_cluster_gap: int = 2,
    cluster_p_threshold: float = 0.05,
    oriC_position: int = None,
    oriC_chrom: str = None,
    gene_info_cache_file: str = None,
    output_dir: str = None
):
    """
    Create Manhattan plot showing mutation rates across the genome.
    
    Windows with significantly different rates are colored differently.
    
    Args:
        windows_rates_df: DataFrame with window rates
        output_path: Path to save plot
        use_treatment_covariate: Whether to show treated rates
        significance_threshold: p-value threshold for significance
        gff_file: Optional GFF file for gene hit analysis
        rate_threshold_multiplier: Multiplier for rate threshold in gene hit analysis
        min_cluster_size: Minimum number of consecutive significant windows in a cluster
        max_cluster_gap: Maximum gap (in windows) allowed within a cluster
        cluster_p_threshold: P-value threshold for cluster significance
        oriC_position: Explicit oriC position (center coordinate)
        oriC_chrom: Chromosome name for oriC
    """
    print(f"Creating Manhattan plot...")
    
    # Calculate genomic position (cumulative across chromosomes)
    chrom_order = sorted(windows_rates_df['chrom'].unique())
    chrom_lengths = {}
    chrom_offsets = {}
    
    cumulative_pos = 0
    for chrom in chrom_order:
        chrom_windows = windows_rates_df[windows_rates_df['chrom'] == chrom]
        chrom_max = chrom_windows['end'].max() if len(chrom_windows) > 0 else 0
        chrom_lengths[chrom] = chrom_max
        chrom_offsets[chrom] = cumulative_pos
        cumulative_pos += chrom_max
    
    # Calculate x positions
    windows_rates_df = windows_rates_df.copy()
    windows_rates_df['x_pos'] = windows_rates_df.apply(
        lambda row: chrom_offsets[row['chrom']] + (row['start'] + row['end']) / 2,
        axis=1
    )
    
    # Filter out windows where control rate > treated rate or where rates are virtually identical (no treatment effect)
    # Save a copy BEFORE filtering for clustering (to avoid breaking up clusters)
    windows_rates_df_before_outlier_filter = None
    if use_treatment_covariate and 'rate_control' in windows_rates_df.columns and 'rate_treated' in windows_rates_df.columns:
        n_before = len(windows_rates_df)
        # Calculate fold-change: treated/control (handle division by zero)
        # Filter out windows where:
        # 1. Control > treated (artifacts)
        # 2. Treated/control < 1.01 (less than 1% increase - virtually identical, no treatment effect)
        # Keep windows where treated >= control AND fold-change >= 1.01 (or either rate is NaN)
        control_positive = windows_rates_df['rate_control'] > 0
        fold_change = np.where(
            control_positive,
            windows_rates_df['rate_treated'] / windows_rates_df['rate_control'],
            np.inf  # If control is 0, fold-change is infinite (keep if treated > 0)
        )
        valid_mask = (
            ((windows_rates_df['rate_treated'] >= windows_rates_df['rate_control']) & (fold_change >= 1.01)) |
            windows_rates_df['rate_treated'].isna() |
            windows_rates_df['rate_control'].isna()
        )
        windows_rates_df = windows_rates_df[valid_mask].copy()
        n_after = len(windows_rates_df)
        n_filtered = n_before - n_after
        if n_filtered > 0:
            print(f"Filtered out {n_filtered} windows where control rate > treated rate or rates are virtually identical (fold-change < 1.01) ({n_after} remaining)")
    
    # Save copy AFTER fold-change filter but BEFORE outlier filter for clustering
    # This way clustering sees windows that pass the fold-change filter (valid treatment effect)
    # but can still span across outlier windows to avoid breaking up clusters
    windows_rates_df_before_outlier_filter = windows_rates_df.copy()
    
    # Filter extreme outliers that would stretch the y-axis (especially for gene windows)
    if use_treatment_covariate and 'rate_treated' in windows_rates_df.columns:
        rate_col_for_outlier = 'rate_treated'
    else:
        rate_col_for_outlier = 'rate_control' if 'rate_control' in windows_rates_df.columns else rate_col
    
    # Calculate outlier threshold using IQR method
    rates = windows_rates_df[rate_col_for_outlier].dropna()
    if len(rates) > 0:
        Q1 = rates.quantile(0.25)
        Q3 = rates.quantile(0.75)
        IQR = Q3 - Q1
        # Use a more conservative threshold (3x IQR instead of 1.5x) to only remove extreme outliers
        outlier_threshold_high = Q3 + 3 * IQR
        
        n_before_outlier = len(windows_rates_df)
        outlier_mask = windows_rates_df[rate_col_for_outlier] <= outlier_threshold_high
        windows_rates_df = windows_rates_df[outlier_mask].copy()
        n_after_outlier = len(windows_rates_df)
        n_outliers_removed = n_before_outlier - n_after_outlier
        
        if n_outliers_removed > 0:
            print(f"Filtered out {n_outliers_removed} extreme outlier windows (rate > {outlier_threshold_high:.6e}) to improve plot readability ({n_after_outlier} remaining)")
    
    # Determine significance using statistical tests on data that passed fold-change filter
    # (but before outlier filter, so clustering can span across outliers)
    # Test if each window's rate is significantly different from the overall mean rate
    # This identifies regions with unusually high or low mutation rates (hotspots/coldspots)
    # We do this on data that passed the fold-change filter (valid treatment effect)
    # but before outlier filter, so clustering can span across outliers without breaking up
    if use_treatment_covariate:
        # Use treated rate for comparison (since that's what we're interested in)
        rate_col = 'rate_treated'
        
        # Calculate overall mean treated rate across windows that passed fold-change filter
        overall_mean = windows_rates_df_before_outlier_filter[rate_col].dropna().mean()
        
        # Do significance testing on data that passed fold-change filter
        windows_rates_df_before_outlier_filter['p_value'] = np.nan
        
        for idx, row in windows_rates_df_before_outlier_filter.iterrows():
            p_val = np.nan
            
            # Test if window rate differs from overall mean using CI-based test
            if (not pd.isna(row[rate_col]) and not pd.isna(overall_mean) and
                row[rate_col] > 0 and
                not pd.isna(row.get('CI_low_treated')) and not pd.isna(row.get('CI_high_treated'))):
                
                # Calculate SE on log scale from CI
                # For log-normal: SE_log ≈ (log(CI_high) - log(CI_low)) / (2 * 1.96)
                se_log_rate = (np.log(row['CI_high_treated']) - np.log(row['CI_low_treated'])) / (2 * 1.96)
                
                if se_log_rate > 0 and np.isfinite(se_log_rate):
                    # Test if log(rate) differs from log(overall_mean)
                    log_rate = np.log(row[rate_col])
                    log_mean = np.log(overall_mean)
                    log_diff = log_rate - log_mean
                    
                    # SE of the difference (assuming independence)
                    # Approximate SE of overall mean (using median absolute deviation as proxy)
                    # For simplicity, use the window's own SE (conservative)
                    se_log_diff = se_log_rate
                    
                    if se_log_diff > 0 and np.isfinite(se_log_diff):
                        z = log_diff / se_log_diff
                        if np.isfinite(z):
                            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Store p-value (significance will be determined after FDR correction)
            if np.isfinite(p_val) and not pd.isna(p_val):
                windows_rates_df_before_outlier_filter.at[idx, 'p_value'] = p_val
        
        # Apply FDR correction to data that passed fold-change filter
        valid_pvalues = windows_rates_df_before_outlier_filter['p_value'].dropna()
        if len(valid_pvalues) > 1:
            pvalues_array = valid_pvalues.values
            rejected, pvalues_corrected, _, _ = multipletests(
                pvalues_array, 
                alpha=significance_threshold, 
                method='fdr_bh'
            )
            windows_rates_df_before_outlier_filter['p_value_fdr_corrected'] = np.nan
            windows_rates_df_before_outlier_filter.loc[valid_pvalues.index, 'p_value_fdr_corrected'] = pvalues_corrected
            windows_rates_df_before_outlier_filter['is_significant'] = False
            windows_rates_df_before_outlier_filter.loc[valid_pvalues.index[rejected], 'is_significant'] = True
            n_significant = rejected.sum()
            n_tested = len(valid_pvalues)
            print(f"Significance testing (FDR-corrected): {n_significant} of {n_tested} windows significant at FDR<{significance_threshold}")
        else:
            windows_rates_df_before_outlier_filter['is_significant'] = False
            windows_rates_df_before_outlier_filter['p_value_fdr_corrected'] = windows_rates_df_before_outlier_filter['p_value']
        
        # Calculate overall mean on filtered data for plotting
        overall_mean_filtered = windows_rates_df[rate_col].dropna().mean()
        
        print(f"Comparing each window's treated rate to overall mean treated rate: {overall_mean:.6e} (after fold-change filter)")
        print(f"Filtered dataset mean (after outlier filter): {overall_mean_filtered:.6e}")
        
        # Copy significance from before-outlier-filter to filtered dataframe (for plotting)
        windows_rates_df['p_value'] = windows_rates_df.index.map(
            lambda idx: windows_rates_df_before_outlier_filter.loc[idx, 'p_value'] if idx in windows_rates_df_before_outlier_filter.index else np.nan
        )
        windows_rates_df['p_value_fdr_corrected'] = windows_rates_df.index.map(
            lambda idx: windows_rates_df_before_outlier_filter.loc[idx, 'p_value_fdr_corrected'] if idx in windows_rates_df_before_outlier_filter.index else np.nan
        )
        windows_rates_df['is_significant'] = windows_rates_df.index.map(
            lambda idx: windows_rates_df_before_outlier_filter.loc[idx, 'is_significant'] if idx in windows_rates_df_before_outlier_filter.index else False
        )
        
        # Calculate fold-change relative to overall mean (for coloring) - use filtered mean for plot
        windows_rates_df['fold_change'] = (
            windows_rates_df[rate_col] / overall_mean_filtered
        ).replace([np.inf, -np.inf], np.nan)
    else:
        # Single rate per window - compare to overall mean
        overall_mean = windows_rates_df['rate_control'].dropna().mean()
        windows_rates_df['p_value'] = np.nan
        
        for idx, row in windows_rates_df.iterrows():
            if (not pd.isna(row['rate_control']) and not pd.isna(overall_mean) and
                not pd.isna(row['CI_low_control']) and not pd.isna(row['CI_high_control'])):
                # Approximate SE from CI
                se = (row['CI_high_control'] - row['CI_low_control']) / (2 * 1.96)
                if se > 0 and np.isfinite(se):
                    z = (row['rate_control'] - overall_mean) / se
                    if np.isfinite(z):
                        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                        windows_rates_df.at[idx, 'p_value'] = p_val
        
        rate_col = 'rate_control'
    
    # Significance testing and FDR correction already done on original data above
    # Just report stats for filtered data
    n_significant = windows_rates_df['is_significant'].sum() if 'is_significant' in windows_rates_df.columns else 0
    n_tested = windows_rates_df['p_value'].notna().sum()
    print(f"Significant windows in filtered dataset: {n_significant} of {n_tested} windows")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Set log scale for y-axis BEFORE plotting (ensures proper transformation)
    ax.set_yscale('log')
    
    # Ensure is_significant column exists
    if 'is_significant' not in windows_rates_df.columns:
        windows_rates_df['is_significant'] = False
    
    # Calculate overall mean for coloring (use filtered mean for plot)
    if use_treatment_covariate:
        overall_mean_plot = overall_mean_filtered if use_treatment_covariate else windows_rates_df[rate_col].dropna().mean()
    else:
        overall_mean_plot = windows_rates_df[rate_col].dropna().mean()
    
    # Identify clusters of significant windows
    # Use data that passed fold-change filter (but before outlier filter) for clustering
    # This ensures we only cluster windows with valid treatment effects, but can span across outliers
    clusters = identify_significant_clusters(
        windows_rates_df_before_outlier_filter,  # Use data after fold-change filter, before outlier filter
        chrom_order, 
        chrom_offsets,
        rate_col, 
        overall_mean,  # Use mean from data that passed fold-change filter
        min_cluster_size=min_cluster_size,
        max_gap=max_cluster_gap,
        cluster_p_threshold=cluster_p_threshold
    )
    
    # Identify which windows are in clusters (for subtle highlighting)
    cluster_window_ids = set()
    for cluster in clusters:
        cluster_window_ids.update(cluster['window_indices'])
    
    # Plot non-significant windows (original style)
    non_sig = windows_rates_df[~windows_rates_df['is_significant']]
    if len(non_sig) > 0:
        ax.scatter(non_sig['x_pos'], non_sig[rate_col], 
                  c='gray', alpha=0.5, s=20)
    
    # Plot significant windows (original style)
    sig = windows_rates_df[windows_rates_df['is_significant']]
    if len(sig) > 0:
        increased = sig[sig[rate_col] > overall_mean_plot]
        decreased = sig[sig[rate_col] < overall_mean_plot]
        
        # Plot all significant windows with original style
        # Cluster windows will be overlaid with different markers
        if len(increased) > 0:
            ax.scatter(increased['x_pos'], increased[rate_col],
                  c='red', alpha=0.7, s=30)
        if len(decreased) > 0:
            ax.scatter(decreased['x_pos'], decreased[rate_col],
                  c='blue', alpha=0.7, s=30)
    
    # Add subtle cluster indicators (very light background shading)
    hotspot_clusters = [c for c in clusters if c['type'] == 'hotspot']
    coldspot_clusters = [c for c in clusters if c['type'] == 'coldspot']
    
    # Very subtle background shading for clusters
    for cluster in hotspot_clusters:
        try:
            valid_indices = [idx for idx in cluster['window_indices'] if idx in windows_rates_df.index]
            if len(valid_indices) == 0:
                continue
            cluster_windows = windows_rates_df.loc[valid_indices]
            if len(cluster_windows) > 0:
                x_min = cluster_windows['x_pos'].min()
                x_max = cluster_windows['x_pos'].max()
                # Very subtle shading
                ax.axvspan(x_min, x_max, alpha=0.08, color='red', zorder=0)
                # Overlay cluster windows with edge markers to indicate cluster membership
                ax.scatter(cluster_windows['x_pos'], cluster_windows[rate_col],
                          c='red', alpha=0.7, s=30, edgecolors='darkred', linewidth=2, zorder=4)
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not plot hotspot cluster: {e}")
            continue
    
    for cluster in coldspot_clusters:
        try:
            valid_indices = [idx for idx in cluster['window_indices'] if idx in windows_rates_df.index]
            if len(valid_indices) == 0:
                continue
            cluster_windows = windows_rates_df.loc[valid_indices]
            if len(cluster_windows) > 0:
                x_min = cluster_windows['x_pos'].min()
                x_max = cluster_windows['x_pos'].max()
                # Very subtle shading
                ax.axvspan(x_min, x_max, alpha=0.08, color='blue', zorder=0)
                # Overlay cluster windows with edge markers to indicate cluster membership
                ax.scatter(cluster_windows['x_pos'], cluster_windows[rate_col],
                          c='blue', alpha=0.7, s=30, edgecolors='darkblue', linewidth=2, zorder=4)
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not plot coldspot cluster: {e}")
            continue
    
    # Add chromosome boundaries
    for i, chrom in enumerate(chrom_order):
        offset = chrom_offsets[chrom]
        if i > 0:
            ax.axvline(x=offset, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Mark origin of replication only if explicitly requested via --oriC-position flag
    if oriC_position is not None:
        oriC = find_origin_of_replication(gff_file, oriC_position=oriC_position, oriC_chrom=oriC_chrom)
        if oriC:
            chrom = oriC['chrom']
            oriC_pos = oriC['position']
            if chrom in chrom_offsets:
                x_pos = chrom_offsets[chrom] + oriC_pos
                y_min, y_max = ax.get_ylim()
                # Draw vertical line for origin
                ax.axvline(x=x_pos, color='purple', linestyle='-', linewidth=2, 
                          alpha=0.8, zorder=5)
                # Add label
                ax.text(x_pos, y_max * 0.98, 'oriC', ha='center', va='top',
                       fontsize=12, fontweight='bold', color='purple',
                       bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                               alpha=0.9, edgecolor='purple', linewidth=1.5),
                       zorder=6)
                if oriC['dnaA_start'] is not None:
                    print(f"Marked origin of replication at {chrom}:{oriC_pos} (dnaA at {oriC['dnaA_start']}-{oriC['dnaA_end']})")
                else:
                    print(f"Marked origin of replication at {chrom}:{oriC_pos} (user-specified)")
    
    # Add horizontal line at overall mean (use filtered mean for plot)
    ax.axhline(y=overall_mean_plot, color='green', linestyle='--', linewidth=1.5, 
               alpha=0.7)
    
    # Labels - matching regression plot font sizes (minimum 10pt, target 12-14pt)
    ax.set_ylabel('Mutation Rate (per base, log scale)', fontsize=14, fontweight='bold')
    # Title removed per user request
    
    # Set log scale for y-axis with proper formatting
    ax.set_yscale('log')
    
    # Format y-axis to show scientific notation with proper log scale ticks
    # Use LogFormatterSciNotation for log scales (not ScalarFormatter which is for linear)
    fmt = LogFormatterSciNotation()
    ax.yaxis.set_major_formatter(fmt)
    
    # Use log locator for major ticks - will be adjusted after limits are set
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    
    ax.grid(alpha=0.3, axis='y', which='both', linestyle='--')
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.tick_params(axis='y', which='minor', labelsize=10)
    
    # X-axis ticks every 100,000 bp
    # Get total genome size
    total_genome_size = cumulative_pos
    
    # Create ticks every 100kb
    tick_interval = 100000
    x_ticks = []
    x_labels = []
    
    for chrom in chrom_order:
        chrom_start = chrom_offsets[chrom]
        chrom_end = chrom_start + chrom_lengths[chrom]
        
        # Find first tick position for this chromosome (round up to next 100kb)
        first_tick_pos = (chrom_start // tick_interval + 1) * tick_interval
        
        # Add ticks at 100kb intervals within this chromosome
        current_pos = first_tick_pos
        while current_pos <= chrom_end:
            # Calculate position within chromosome
            pos_in_chrom = current_pos - chrom_start
            x_ticks.append(current_pos)
            # Label format: position in kb
            x_labels.append(f'{int(pos_in_chrom/1000)}')
            current_pos += tick_interval
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
    ax.set_xlabel('Genomic Position (kb)', fontsize=14, fontweight='bold')
    
    # Add chromosome labels as secondary x-axis or text annotations
    # Add text labels for chromosomes at the top of the plot
    y_min, y_max = ax.get_ylim()
    # For log scale, use multiplicative spacing
    log_range = np.log10(y_max) - np.log10(y_min)
    new_y_max = 10**(np.log10(y_max) + 0.15 * log_range)
    
    for chrom in chrom_order:
        chrom_center = chrom_offsets[chrom] + chrom_lengths[chrom] / 2
        # Position text in log space
        text_y = 10**(np.log10(y_max) + 0.05 * log_range)
        ax.text(chrom_center, text_y, chrom, ha='center', va='bottom',
               fontsize=12, fontweight='bold', transform=ax.transData)
    
    # Expand y-axis slightly to accommodate chromosome labels
    ax.set_ylim(top=new_y_max)
    
    # Now that y-axis limits are finalized, manually generate ticks to ensure we have enough
    y_min_final, y_max_final = ax.get_ylim()
    log_min = np.log10(y_min_final)
    log_max = np.log10(y_max_final)
    log_range_final = log_max - log_min
    
    # Generate tick positions manually
    # For small ranges, use more granular ticks (1, 2, 5 per decade)
    tick_positions = []
    decade_min = int(np.floor(log_min))
    decade_max = int(np.ceil(log_max))
    
    if log_range_final < 1:
        # Small range: generate ticks at 1, 2, 5 for each decade
        for decade in range(decade_min, decade_max + 1):
            for multiplier in [1.0, 2.0, 5.0]:
                tick_val = multiplier * (10 ** decade)
                if y_min_final * 0.95 <= tick_val <= y_max_final * 1.05:  # Slight padding
                    tick_positions.append(tick_val)
    elif log_range_final < 2:
        # Medium range: use 1, 2, 5 per decade
        for decade in range(decade_min, decade_max + 1):
            for multiplier in [1.0, 2.0, 5.0]:
                tick_val = multiplier * (10 ** decade)
                if y_min_final * 0.95 <= tick_val <= y_max_final * 1.05:
                    tick_positions.append(tick_val)
    else:
        # Large range: use standard decade ticks
        for decade in range(decade_min, decade_max + 1):
            tick_val = 10 ** decade
            if y_min_final * 0.95 <= tick_val <= y_max_final * 1.05:
                tick_positions.append(tick_val)
    
    # Remove duplicates and sort
    tick_positions = sorted(list(set(tick_positions)))
    
    # Set the ticks explicitly - use set_yticks which overrides the locator
    if len(tick_positions) >= 2:  # Need at least 2 ticks
        ax.set_yticks(tick_positions)
        ax.yaxis.set_major_formatter(fmt)
    else:
        # Fallback: if we don't have enough ticks, force generate more
        # Expand the range slightly and regenerate
        for decade in range(decade_min - 1, decade_max + 2):
            for multiplier in [1.0, 2.0, 5.0]:
                tick_val = multiplier * (10 ** decade)
                if y_min_final * 0.8 <= tick_val <= y_max_final * 1.2:
                    tick_positions.append(tick_val)
        tick_positions = sorted(list(set(tick_positions)))
        if len(tick_positions) >= 2:
            ax.set_yticks(tick_positions)
            ax.yaxis.set_major_formatter(fmt)
        else:
            # Last resort: use locator with subs
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=20))
            ax.yaxis.set_major_formatter(fmt)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Manhattan plot saved to {output_path}")
    
    # Save cluster information to TSV
    if clusters:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            cluster_path = os.path.join(output_dir, 'significant_clusters.tsv')
        else:
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            cluster_path = f'{base_name}_clusters.tsv'
        
        cluster_df = pd.DataFrame([
            {
                'chrom': c['chrom'],
                'start': c['start'],
                'end': c['end'],
                'type': c['type'],
                'n_windows': c['n_windows'],
                'genomic_span_bp': c['genomic_span'],
                'mean_rate': c['mean_rate'],
                'p_value': c['p_value']
            }
            for c in clusters
        ])
        cluster_df = cluster_df.sort_values(['chrom', 'start'])
        cluster_df.to_csv(cluster_path, sep='\t', index=False)
        print(f"Significant clusters saved to {cluster_path}")
        print(f"  Found {len([c for c in clusters if c['type'] == 'hotspot'])} hotspot clusters")
        print(f"  Found {len([c for c in clusters if c['type'] == 'coldspot'])} coldspot clusters")
        
        # Perform gene analysis for hotspot clusters if GFF file provided
        if gff_file and os.path.exists(gff_file):
            print(f"\nPerforming gene analysis for hotspot clusters with GFF file: {gff_file}")
            # First try with 50% threshold, if no results, try with any overlap
            gene_hits_df = analyze_genes_in_clusters(clusters, windows_rates_df, gff_file, min_gene_coverage=50.0, report_all_overlaps=False)
            if gene_hits_df.empty or gene_hits_df['n_genes'].sum() == 0:
                print(f"\nNo genes found with >=50% coverage. Trying with any overlap...")
                gene_hits_df = analyze_genes_in_clusters(clusters, windows_rates_df, gff_file, min_gene_coverage=0.0, report_all_overlaps=True)
            
            if not gene_hits_df.empty:
                if output_dir:
                    gene_hits_path = os.path.join(output_dir, 'genes_in_hotspot_clusters.tsv')
                else:
                    base_name = os.path.splitext(os.path.basename(output_path))[0]
                    gene_hits_path = f'{base_name}_genes_in_clusters.tsv'
                
                gene_hits_df.to_csv(gene_hits_path, sep='\t', index=False)
                print(f"Gene hits in hotspot clusters saved to {gene_hits_path}")
            else:
                print("No genes found in hotspot clusters (with >=50% coverage)")
        elif gff_file:
            print(f"Warning: GFF file specified but not found: {gff_file}")
    else:
        print("No significant clusters found")
    
    # Generate gene hits files for highest and lowest rate windows
    if gff_file and os.path.exists(gff_file) and gene_info_cache_file:
        # Use explicitly provided output_dir, or derive from output_path
        if output_dir is None:
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        # Ensure output_dir is absolute
        output_dir = os.path.abspath(output_dir)
        
        # Load gene info cache
        gene_info_cache = load_gene_info_cache(gene_info_cache_file)
        print(f"\nLoaded gene info for {len(gene_info_cache)} genes from cache")
        
        # rate_col is already defined above in the significance testing section
        # Analyze highest rate windows
        print(f"\nAnalyzing genes in highest rate windows...")
        highest_gene_hits = analyze_genes_in_rate_windows(
            windows_rates_df, gff_file, gene_info_cache, rate_col, n_windows=10, highest=True
        )
        if not highest_gene_hits.empty:
            highest_path = os.path.join(output_dir, 'gene_hits_high_rate_windows.tsv')
            highest_gene_hits.to_csv(highest_path, sep='\t', index=False)
            print(f"Gene hits in highest rate windows saved to {highest_path}")
        
        # Analyze lowest rate windows
        print(f"\nAnalyzing genes in lowest rate windows...")
        lowest_gene_hits = analyze_genes_in_rate_windows(
            windows_rates_df, gff_file, gene_info_cache, rate_col, n_windows=10, highest=False
        )
        if not lowest_gene_hits.empty:
            lowest_path = os.path.join(output_dir, 'gene_hits_low_rate_windows.tsv')
            lowest_gene_hits.to_csv(lowest_path, sep='\t', index=False)
            print(f"Gene hits in lowest rate windows saved to {lowest_path}")


def create_multiplot(output_dir, output_path=None):
    """Create a single multiplot figure with all three panels."""
    # Load data
    data = load_data(output_dir)
    
    # Create figure with subplots - stacked layout: panel 1 wide on top, panels 2&3 side by side below
    fig = plt.figure(figsize=(24, 20))  # Taller to accommodate sample labels
    # Grid spec: 2 rows, 2 columns
    # Row 0: Panel 1 spans both columns (full width)
    # Row 1: Panel 2 in left column, Panel 3 in right column
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, 
                         left=0.10, right=0.96, top=0.96, bottom=0.10,
                         height_ratios=[1.3, 1.0], width_ratios=[1.0, 1.0])
    
    # Panel 1: Per-sample GLM rates (spans full width on top)
    ax1 = fig.add_subplot(gs[0, :])
    sample_color_dict = plot_glm_per_sample(ax1, data['glm_per_sample'])
    
    # Panel 2: Category significances (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_category_significances(ax2, data['category_rates'], data['category_tests'])
    
    # Panel 3: Treatment day rates (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    plot_treatment_day_rates(ax3, data['treatment_day_rates'], 
                            data['treatment_day_per_sample'], 
                            data['treatment_day_tests'],
                            sample_color_dict=sample_color_dict)
    
    # No overall title to avoid overlap
    
    # Save figure
    if output_path is None:
        output_path = os.path.join(output_dir, 'combined_mutation_rates_plot.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    print(f"Multiplot saved to {output_path}")
    plt.close()


def create_control_baseline_comparison_plot(output_dir, output_path=None):
    """Create a multi-panel plot comparing control-baseline approach to original approach."""
    # Load data
    data = load_data(output_dir)
    
    if data['category_rates'] is None or data['category_rates'].empty:
        print("Warning: No category rate data available for comparison plot")
        return
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(24, 8))
    
    # Create multi-panel comparison
    plot_control_baseline_model_comparison(fig, data['category_rates'], data['category_tests'])
    
    # Save figure
    if output_path is None:
        output_path = os.path.join(output_dir, 'control_baseline_comparison.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    print(f"Control-baseline comparison plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate publication-ready plots from estimate_rates.py output TSV files"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory from estimate_rates.py (should contain TSV files)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for the combined plot (default: output_dir/combined_mutation_rates_plot.png)'
    )
    parser.add_argument(
        '--comparison-plot-only',
        action='store_true',
        help='Only create the control-baseline comparison plot (skip main multiplot)'
    )
    parser.add_argument(
        '--comparison-plot-path',
        type=str,
        default=None,
        help='Output path for the control-baseline comparison plot (default: output_dir/control_baseline_comparison.png)'
    )
    parser.add_argument(
        '--create-manhattan-plot',
        type=str,
        default=None,
        help='Directory containing window_rates.tsv file. Manhattan plot and gene hits files will be created in this directory.'
    )
    parser.add_argument(
        '--manhattan-plot-path',
        type=str,
        default=None,
        help='Output path for Manhattan plot (default: <manhattan-dir>/manhattan_plot.png)'
    )
    parser.add_argument(
        '--gff-file',
        type=str,
        default=None,
        help='GFF file for gene hit analysis in Manhattan plot'
    )
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=3,
        help='Minimum number of consecutive significant windows in a cluster (default: 3)'
    )
    parser.add_argument(
        '--max-cluster-gap',
        type=int,
        default=2,
        help='Maximum gap (in windows) allowed within a cluster (default: 2)'
    )
    parser.add_argument(
        '--cluster-p-threshold',
        type=float,
        default=0.05,
        help='P-value threshold for cluster significance (default: 0.05)'
    )
    parser.add_argument(
        '--oriC-position',
        type=int,
        default=None,
        help='Explicit oriC position (center coordinate, e.g., 988565 for range 988364-988765)'
    )
    parser.add_argument(
        '--oriC-chrom',
        type=str,
        default=None,
        help='Chromosome name for oriC (optional, will use first chromosome if not specified)'
    )
    parser.add_argument(
        '--gene-info-cache',
        type=str,
        default=None,
        help='Path to gene_info_cache.json file for adding gene names and descriptions to gene hits files. Default: looks for data/references/gene_info_cache.json relative to script location'
    )
    
    args = parser.parse_args()
    
    # Try to find default gene_info_cache.json if not provided
    if args.gene_info_cache is None:
        # Try relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_cache = os.path.join(script_dir, '..', '..', '..', 'data', 'references', 'gene_info_cache.json')
        default_cache = os.path.normpath(default_cache)
        if os.path.exists(default_cache):
            args.gene_info_cache = default_cache
            print(f"Using default gene info cache: {args.gene_info_cache}")
    
    if not os.path.isdir(args.output_dir):
        print(f"Error: {args.output_dir} is not a directory")
        return 1
    
    if args.comparison_plot_only:
        create_control_baseline_comparison_plot(args.output_dir, args.comparison_plot_path)
    else:
        create_multiplot(args.output_dir, args.output_path)
        # Also create comparison plot
        create_control_baseline_comparison_plot(args.output_dir, args.comparison_plot_path)
    
    # Create Manhattan plot if requested
    if args.create_manhattan_plot:
        manhattan_dir = args.create_manhattan_plot
        if not os.path.isdir(manhattan_dir):
            print(f"Error: Manhattan plot directory does not exist: {manhattan_dir}")
            return 1
        
        # Load window rates from the specified directory
        window_rates_path = os.path.join(manhattan_dir, 'window_rates.tsv')
        if not os.path.exists(window_rates_path):
            print(f"Error: window_rates.tsv not found in {manhattan_dir}")
            return 1
        
        print(f"Loading window rates from {window_rates_path}")
        windows_df = pd.read_csv(window_rates_path, sep='\t')
        print(f"Loaded {len(windows_df)} windows")
        
        # Set output path
        if args.manhattan_plot_path is None:
            args.manhattan_plot_path = os.path.join(manhattan_dir, 'manhattan_plot.png')
        
        # Determine if we have treated rates
        use_treatment = 'rate_treated' in windows_df.columns
        
        plot_regional_rates_manhattan(
            windows_df,
            args.manhattan_plot_path,
            use_treatment_covariate=use_treatment,
            gff_file=args.gff_file,
            min_cluster_size=args.min_cluster_size,
            max_cluster_gap=args.max_cluster_gap,
            cluster_p_threshold=args.cluster_p_threshold,
            oriC_position=args.oriC_position,
            oriC_chrom=args.oriC_chrom,
            gene_info_cache_file=args.gene_info_cache,
            output_dir=manhattan_dir
        )
        print(f"Manhattan plot created: {args.manhattan_plot_path}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

