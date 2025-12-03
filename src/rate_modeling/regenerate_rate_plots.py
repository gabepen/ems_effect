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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter, LogLocator, MaxNLocator
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
rcParams['font.size'] = 22
rcParams['axes.labelsize'] = 26
rcParams['axes.titlesize'] = 30
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20
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
    # Prefer category_rates_summary.tsv which has 5mer-normalized rates if computed
    category_path = os.path.join(output_dir, 'category_comparison', 'category_rates_summary.tsv')
    if not os.path.exists(category_path):
        # Fall back to NB GLM results (may not have 5mer normalization)
        category_path = os.path.join(output_dir, 'category_comparison', 'category_rates_nb_glm.tsv')
    if os.path.exists(category_path):
        data['category_rates'] = pd.read_csv(category_path, sep='\t')
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
    ax.set_yticklabels(labels, fontsize=20)
    
    # Format x-axis with ticks at 1e-4 and 3e-4
    ax.set_xlabel('Mutation Rate (per base)', fontsize=28, fontweight='bold', labelpad=25)
    ax.set_xscale('log')
    
    # Set explicit ticks at 1e-4 and 3e-4
    ax.set_xticks([1e-4, 3e-4])
    
    # Format with scientific notation
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(fmt)
    
    ax.tick_params(axis='x', labelsize=20, pad=12, which='major')
    ax.tick_params(axis='y', labelsize=20, pad=10)
    
    # Add legend - use plot_ems_spectra colors
    control_patch = mpatches.Patch(color=NT_GREY, label='Control', alpha=0.8)
    treated_patch = mpatches.Patch(color=TREATED_COLORS[1], label='Treated', alpha=0.8)
    ax.legend(handles=[control_patch, treated_patch], loc='lower right', frameon=True, 
              fancybox=True, shadow=True, fontsize=20, framealpha=0.9)
    
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
    
    # Panel 2: Treatment effects - show BOTH absolute and fold-change
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate both absolute effects and fold-changes
    absolute_effects = []
    fold_changes = []
    categories_ordered = []
    for cat in control_df['category']:
        control_rate = control_df[control_df['category'] == cat]['rate'].values[0]
        treated_rate = treated_df[treated_df['category'] == cat]['rate'].values[0]
        if control_rate > 0:
            absolute_effect = treated_rate - control_rate
            fold_change = treated_rate / control_rate
            absolute_effects.append(absolute_effect)
            fold_changes.append(fold_change)
            categories_ordered.append(cat)
    
    x_pos2 = np.arange(len(absolute_effects))
    labels2 = [category_labels.get(cat, cat) for cat in categories_ordered]
    
    # Plot absolute effects (primary)
    bars2 = ax2.bar(x_pos2, absolute_effects, color=TREATED_COLORS[1], alpha=0.7,
                   edgecolor='black', linewidth=1, width=0.6, label='Absolute effect')
    
    # Add horizontal line at 0 (no change)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add absolute effect values on bars
    for bar, abs_eff in zip(bars2, absolute_effects):
        height = bar.get_height()
        y_pos = height + abs_eff * 0.05 if abs_eff >= 0 else height + abs_eff * 0.05
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{abs_eff:.2e}', ha='center', 
                va='bottom' if abs_eff >= 0 else 'top', 
                fontsize=14, fontweight='bold')
    
    # Also show fold-change as text below bars
    for i, (bar, fc) in enumerate(zip(bars2, fold_changes)):
        ax2.text(bar.get_x() + bar.get_width()/2., -max(absolute_effects) * 0.15,
                f'({fc:.2f}x)', ha='center', va='top', 
                fontsize=12, style='italic', color='gray')
    
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(labels2, fontsize=18, fontweight='bold', rotation=15, ha='right')
    ax2.set_ylabel('Absolute Effect (Treated - Control)', fontsize=22, fontweight='bold', labelpad=20)
    ax2.set_title('Treatment Effect\n(Absolute & Fold-change)', fontsize=24, fontweight='bold', pad=15)
    ax2.legend(fontsize=16)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax2.tick_params(axis='y', labelsize=18, pad=10)
    
    # Adjust y-axis to show both positive and negative space for labels
    max_abs = max([abs(x) for x in absolute_effects])
    ax2.set_ylim(bottom=-max_abs * 0.25, top=max_abs * 1.2)
    
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
    """Plot mutation category rates with absolute treatment effects.
    
    Shows:
    - Bar plot of treated rates by category
    - Absolute treatment effect annotations (if available)
    - Significance table
    """
    if category_df is None or category_df.empty:
        ax.text(0.5, 0.5, 'No category data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=24)
        return
    
    # Check if we have absolute effect data
    has_absolute_effects = 'absolute_effect' in category_df.columns
    
    # Filter to treated samples only for main comparison
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
    
    # Sort by rate (lowest to highest) - darker colors will be on higher rates
    treated_df = treated_df.sort_values('rate', ascending=True)
    
    # Create x positions - squeeze boundaries by reducing spacing
    n_bars = len(treated_df)
    # Use much thinner bars - make them half the previous width
    bar_width = 0.1  # Much thinner bars
    # Calculate x positions with minimal spacing - pack bars closer together
    if n_bars > 1:
        # Use tighter spacing: bars are closer together
        spacing = 0.25  # Reduced spacing between bars
        total_width = (n_bars - 1) * spacing
        x_pos = np.linspace(0, total_width, n_bars)
    else:
        x_pos = np.array([0])
    
    # Map categories to colors - darker colors for higher rates
    # Since we sorted ascending, assign colors based on rate order (darker = higher rate)
    # Use blue-to-purple gradient for middle panel (complements orange-to-red used elsewhere)
    colors = []
    n_categories = len(treated_df)
    
    if n_categories == 1:
        colors.append(CATEGORY_COLORS[1])  # Medium blue for single bar
    else:
        # Create gradient from light to dark based on position (darker = higher rate)
        # Use blue-to-purple gradient instead of orange-to-red
        for i in range(n_categories):
            # Map position to color gradient (0 = lightest, n-1 = darkest)
            color_idx = int((i / (n_categories - 1)) * (len(CATEGORY_COLORS) - 1))
            color_idx = min(color_idx, len(CATEGORY_COLORS) - 1)
            colors.append(CATEGORY_COLORS[color_idx])
    
    # Plot bars - make them much thinner and closer together
    bars = ax.bar(x_pos, treated_df['rate'], color=colors, alpha=0.7,
                  edgecolor='black', linewidth=0.5, width=bar_width)
    
    # Add error bars
    if 'CI_low' in treated_df.columns and 'CI_high' in treated_df.columns:
        yerr_low = treated_df['rate'] - treated_df['CI_low']
        yerr_high = treated_df['CI_high'] - treated_df['rate']
        ax.errorbar(x_pos, treated_df['rate'],
                   yerr=[yerr_low, yerr_high], fmt='none',
                   color='black', capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8)
    
    # Add absolute effect annotations on bars if available
    if has_absolute_effects:
        for i, (bar, (_, row)) in enumerate(zip(bars, treated_df.iterrows())):
            abs_eff = row.get('absolute_effect', np.nan)
            if np.isfinite(abs_eff):
                height = bar.get_height()
                # Add absolute effect text above the bar
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                       f'+{abs_eff:.2e}',
                       ha='center', va='bottom', fontsize=14, fontweight='bold',
                       color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', 
                                alpha=0.7, edgecolor='darkgreen', linewidth=1))
    
    # Format category labels - use shorter labels to avoid overlap
    category_labels = {
        'intergenic': 'Intergenic',
        'synonymous': 'Synonymous',
        'non_synonymous': 'Non-syn',
        'coding': 'Coding'
    }
    labels = [category_labels.get(cat, cat) for cat in treated_df['category']]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=20, fontweight='bold', rotation=15, ha='right')
    
    # Squeeze plot boundaries - reduce padding on left and right to bring bars closer
    if len(x_pos) > 0:
        padding = 0.15  # Minimal padding
        ax.set_xlim(left=x_pos[0] - padding, right=x_pos[-1] + padding)
    
    # Format y-axis
    ylabel = 'Mutation Rate (per base)'
    if has_absolute_effects:
        ylabel += '\n(+abs. effect)'
    ax.set_ylabel(ylabel, fontsize=28, fontweight='bold', labelpad=25)
    ax.set_yscale('log')
    # Set y-axis range to include all rates and their CI bounds
    min_rate = treated_df['rate'].min() * 0.9  # Slightly below minimum rate
    # Get maximum rate including CI_high bounds to ensure all error bars are visible
    max_rate = treated_df['rate'].max()
    if 'CI_high' in treated_df.columns:
        max_ci_high = treated_df['CI_high'].max()
        max_rate = max(max_rate, max_ci_high)
    # Add padding above the maximum (more if showing absolute effects)
    padding_mult = 1.6 if has_absolute_effects else 1.3
    ax.set_ylim(min_rate, max_rate * padding_mult)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis='y', labelsize=20, pad=12)
    ax.tick_params(axis='x', labelsize=20, pad=14)
    
    # Create significance table if tests available
    if tests_df is not None and not tests_df.empty:
        # Extract p-values for comparisons
        table_data = []
        
        # syn vs intergenic
        syn_vs_inter = tests_df[tests_df['test'].str.contains('synonymous vs intergenic.*IN TREATED', regex=True, na=False)]
        if not syn_vs_inter.empty:
            p = syn_vs_inter.iloc[0]['p_value']
            symbol, label = format_pvalue(p)
            table_data.append(['Syn vs Inter', symbol, label])
        
        # nonsyn vs intergenic
        nonsyn_vs_inter = tests_df[tests_df['test'].str.contains('non_synonymous vs intergenic.*IN TREATED', regex=True, na=False)]
        if not nonsyn_vs_inter.empty:
            p = nonsyn_vs_inter.iloc[0]['p_value']
            symbol, label = format_pvalue(p)
            table_data.append(['Nonsyn vs Inter', symbol, label])
        
        # syn vs nonsyn
        syn_vs_nonsyn = tests_df[tests_df['test'].str.contains('synonymous vs non_synonymous.*IN TREATED', regex=True, na=False)]
        if not syn_vs_nonsyn.empty:
            p = syn_vs_nonsyn.iloc[0]['p_value']
            symbol, label = format_pvalue(p)
            table_data.append(['Syn vs Nonsyn', symbol, label])
        
        # Display significance values in the center of the plot
        if table_data:
            ax_pos = ax.get_position()
            text_x_center = (ax_pos.x0 + ax_pos.x1) / 2  # Center of the plot area
            
            # Collect significance data with p-values for color coding
            sig_data = []
            for comp, symbol, p_str in table_data:
                # Extract p-value for color coding
                p_val = 1.0
                try:
                    if 'p=' in p_str:
                        p_val = float(p_str.split('p=')[1].strip())
                    elif 'p<' in p_str:
                        p_val = float(p_str.split('p<')[1].strip())
                except (ValueError, IndexError):
                    pass
                sig_data.append((comp, symbol, p_str, p_val))
            
            # Add significance text boxes at the top of the plot
            y_start = ax_pos.y1 - 0.05  # Start near top of plot
            for i, (comp, symbol, p_str, p_val) in enumerate(sig_data):
                sig_text = f'{comp}:\n{symbol} {p_str}'
                
                # Determine color based on significance
                if p_val < 0.001:
                    text_color = 'red'
                    bg_color = '#FFE6E6'
                elif p_val < 0.01:
                    text_color = 'darkorange'
                    bg_color = '#FFE6CC'
                elif p_val < 0.05:
                    text_color = 'orange'
                    bg_color = '#FFF4E6'
                else:
                    text_color = 'gray'
                    bg_color = '#F5F5F5'
                
                fig = ax.get_figure()
                fig.text(text_x_center, y_start - i * 0.08, sig_text, 
                        fontsize=18, fontweight='bold', color=text_color,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, alpha=0.8, edgecolor=text_color),
                        ha='center', va='top', transform=fig.transFigure)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Add title indicating model type if using absolute effects
    if has_absolute_effects:
        ax.set_title('Treated Rates\n(Identity Link Model)', fontsize=22, fontweight='bold', pad=10)


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
    
    # Format y-axis
    ax.set_ylabel('Mutation Rate (per base)', fontsize=28, fontweight='bold', labelpad=25)
    ax.set_yscale('log')
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis='y', labelsize=20, pad=12)
    ax.tick_params(axis='x', labelsize=20, pad=14)
    
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
            
            # Display as compact text in bottom right
            table_text = "Comparisons:\n"
            for day1_label, day2_label, symbol, p_str, _ in comparisons[:6]:  # Limit to 6 comparisons
                table_text += f"{day1_label} vs {day2_label}: {symbol} ({p_str})\n"
            
            ax.text(0.98, 0.02, table_text, transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=18,
                   bbox=dict(boxstyle='round,pad=1.0', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=2),
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
    n_cluster = len(indices)
    cluster_start = cluster_windows['start'].min()
    cluster_end = cluster_windows['end'].max()
    mean_rate = cluster_windows[rate_col].mean()
    
    # Test cluster significance using binomial test
    # Null hypothesis: cluster is random (proportion of significant windows matches genome-wide)
    # Alternative: cluster has more significant windows than expected by chance
    p_sig_genome = total_sig / total_windows if total_windows > 0 else 0
    
    # Binomial test: probability of observing n_cluster significant windows
    # out of n_cluster total windows, given genome-wide proportion
    if p_sig_genome > 0 and p_sig_genome < 1 and n_cluster > 0:
        # Use binomtest for newer scipy, fallback to binom_test for older versions
        try:
            from scipy.stats import binomtest
            # binomtest(k, n, p, alternative='greater')
            result = binomtest(n_cluster, n_cluster, p_sig_genome, alternative='greater')
            p_value = result.pvalue
        except (ImportError, AttributeError):
            # Fallback for older scipy versions
            try:
                p_value = stats.binom_test(n_cluster, n_cluster, p_sig_genome, alternative='greater')
            except (AttributeError, TypeError):
                # Manual calculation if binom_test not available
                from scipy.stats import binom
                p_value = 1 - binom.cdf(n_cluster - 1, n_cluster, p_sig_genome)
    else:
        # If all or no windows are significant, can't test
        p_value = 1.0
    
    return {
        'type': cluster['type'],
        'chrom': chrom,
        'start': int(cluster_start),
        'end': int(cluster_end),
        'n_windows': n_cluster,
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
    oriC_chrom: str = None
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
    
    # Determine significance using statistical tests
    # Test if each window's rate is significantly different from the overall mean rate
    # This identifies regions with unusually high or low mutation rates (hotspots/coldspots)
    if use_treatment_covariate:
        # Use treated rate for comparison (since that's what we're interested in)
        rate_col = 'rate_treated'
        
        # Calculate overall mean treated rate across all windows
        overall_mean = windows_rates_df[rate_col].dropna().mean()
        
        print(f"Comparing each window's treated rate to overall mean treated rate: {overall_mean:.6e}")
        
        # For each window, test if treated rate differs from overall mean
        windows_rates_df['p_value'] = np.nan
        
        for idx, row in windows_rates_df.iterrows():
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
                windows_rates_df.at[idx, 'p_value'] = p_val
        
        # Calculate fold-change relative to overall mean (for coloring)
        windows_rates_df['fold_change'] = (
            windows_rates_df[rate_col] / overall_mean
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
    
    # Apply FDR correction (Benjamini-Hochberg) to p-values
    # Get all valid p-values
    valid_pvalues = windows_rates_df['p_value'].dropna()
    if len(valid_pvalues) > 1:
        # Apply FDR correction
        pvalues_array = valid_pvalues.values
        rejected, pvalues_corrected, alpha_sidak, alpha_bonf = multipletests(
            pvalues_array, 
            alpha=significance_threshold, 
            method='fdr_bh'  # Benjamini-Hochberg FDR correction
        )
        
        # Store corrected p-values
        windows_rates_df['p_value_fdr_corrected'] = np.nan
        windows_rates_df.loc[valid_pvalues.index, 'p_value_fdr_corrected'] = pvalues_corrected
        
        # Update significance based on FDR-corrected p-values
        windows_rates_df['is_significant'] = False
        windows_rates_df.loc[valid_pvalues.index[rejected], 'is_significant'] = True
        
        n_significant = rejected.sum()
        n_tested = len(valid_pvalues)
        print(f"Significance testing (FDR-corrected): {n_significant} of {n_tested} windows significant at FDR<{significance_threshold}")
        print(f"  Applied Benjamini-Hochberg FDR correction for multiple comparisons")
    else:
        # If only one or no p-values, no correction needed
        n_significant = windows_rates_df['is_significant'].sum() if 'is_significant' in windows_rates_df.columns else 0
        n_tested = windows_rates_df['p_value'].notna().sum()
        windows_rates_df['p_value_fdr_corrected'] = windows_rates_df['p_value']
        print(f"Significance testing: {n_significant} of {n_tested} windows significant at p<{significance_threshold}")
        if n_tested <= 1:
            print(f"  No multiple comparisons correction needed (only {n_tested} test(s))")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Ensure is_significant column exists
    if 'is_significant' not in windows_rates_df.columns:
        windows_rates_df['is_significant'] = False
    
    # Calculate overall mean for coloring
    if use_treatment_covariate:
        overall_mean = windows_rates_df[rate_col].dropna().mean()
    else:
        overall_mean = windows_rates_df[rate_col].dropna().mean()
    
    # Identify clusters of significant windows
    clusters = identify_significant_clusters(
        windows_rates_df, 
        chrom_order, 
        chrom_offsets,
        rate_col, 
        overall_mean,
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
        increased = sig[sig[rate_col] > overall_mean]
        decreased = sig[sig[rate_col] < overall_mean]
        
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
    
    # Mark origin of replication if GFF file provided or explicit position given
    if gff_file or oriC_position is not None:
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
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               alpha=0.9, edgecolor='purple', linewidth=1.5),
                       zorder=6)
                if oriC['dnaA_start'] is not None:
                    print(f"Marked origin of replication at {chrom}:{oriC_pos} (dnaA at {oriC['dnaA_start']}-{oriC['dnaA_end']})")
                else:
                    print(f"Marked origin of replication at {chrom}:{oriC_pos} (user-specified)")
    
    # Add horizontal line at overall mean
    ax.axhline(y=overall_mean, color='green', linestyle='--', linewidth=1.5, 
               alpha=0.7)
    
    # Labels
    ax.set_ylabel('Mutation Rate (per base, log scale)', fontsize=12, fontweight='bold')
    if use_treatment_covariate:
        ax.set_title('Treated Mutation Rates Across Genome\n(compared to overall mean; FDR-corrected)', 
                    fontsize=14, fontweight='bold')
    else:
        ax.set_title('Mutation Rates Across Genome\n(compared to overall mean; FDR-corrected)', 
                    fontsize=14, fontweight='bold')
    
    # Set log scale for y-axis with proper formatting
    ax.set_yscale('log')
    
    # Format y-axis to show scientific notation with proper log scale ticks
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    
    # Use log locator for major ticks
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    
    ax.grid(alpha=0.3, axis='y', which='both', linestyle='--')
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.tick_params(axis='y', which='minor', labelsize=8)
    
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
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Genomic Position (kb)', fontsize=12)
    
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
               fontsize=10, fontweight='bold', transform=ax.transData)
    
    # Expand y-axis slightly to accommodate chromosome labels
    ax.set_ylim(top=new_y_max)
    
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


def create_multiplot(output_dir, output_path=None):
    """Create a single multiplot figure with all three panels."""
    # Load data
    data = load_data(output_dir)
    
    # Create figure with subplots - larger size to accommodate larger fonts and significance boxes
    fig = plt.figure(figsize=(32, 12))  # Slightly wider to accommodate significance text boxes
    # Adjust grid spec to give more space for larger fonts and prevent overlap
    # Make center panel narrower since we removed coding bar - more space for significance boxes on left
    # Use width_ratios to make center panel thinner
    gs = fig.add_gridspec(1, 3, hspace=0.75, wspace=0.5, 
                         left=0.10, right=0.92, top=0.90, bottom=0.15,
                         width_ratios=[1.2, 0.6, 1.2])  # Center panel is 0.6x width (thinner), sides are 1.2x
    
    # Panel 1: Per-sample GLM rates
    ax1 = fig.add_subplot(gs[0, 0])
    sample_color_dict = plot_glm_per_sample(ax1, data['glm_per_sample'])
    
    # Panel 2: Category significances
    ax2 = fig.add_subplot(gs[0, 1])
    plot_category_significances(ax2, data['category_rates'], data['category_tests'])
    
    # Panel 3: Treatment day rates (pass sample colors from first plot)
    ax3 = fig.add_subplot(gs[0, 2])
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
        action='store_true',
        help='Create Manhattan plot from 5mer normalized window data (if available)'
    )
    parser.add_argument(
        '--manhattan-plot-path',
        type=str,
        default=None,
        help='Output path for Manhattan plot (default: output_dir/5mer_normalized_windows/manhattan_plot.png)'
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
    
    args = parser.parse_args()
    
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
        data = load_data(args.output_dir)
        if data['kmer5_windows'] is not None and not data['kmer5_windows'].empty:
            if args.manhattan_plot_path is None:
                manhattan_dir = os.path.join(args.output_dir, '5mer_normalized_windows')
                os.makedirs(manhattan_dir, exist_ok=True)
                args.manhattan_plot_path = os.path.join(manhattan_dir, 'manhattan_plot.png')
            
            # Determine if we have treated rates
            use_treatment = 'rate_treated' in data['kmer5_windows'].columns
            plot_regional_rates_manhattan(
                data['kmer5_windows'],
                args.manhattan_plot_path,
                use_treatment_covariate=use_treatment,
                gff_file=args.gff_file,
                min_cluster_size=args.min_cluster_size,
                max_cluster_gap=args.max_cluster_gap,
                cluster_p_threshold=args.cluster_p_threshold,
                oriC_position=args.oriC_position,
                oriC_chrom=args.oriC_chrom
            )
            print(f"Manhattan plot created: {args.manhattan_plot_path}")
        else:
            print("Warning: 5mer normalized window data not found. Skipping Manhattan plot.")
            print(f"  Expected file: {os.path.join(args.output_dir, '5mer_normalized_windows', 'window_rates.tsv')}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

