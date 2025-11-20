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
    category_path = os.path.join(output_dir, 'category_comparison', 'category_rates_nb_glm.tsv')
    if not os.path.exists(category_path):
        category_path = os.path.join(output_dir, 'category_comparison', 'category_rates_summary.tsv')
    if os.path.exists(category_path):
        data['category_rates'] = pd.read_csv(category_path, sep='\t')
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


def plot_category_significances(ax, category_df, tests_df):
    """Plot mutation category rates with significance table."""
    if category_df is None or category_df.empty:
        ax.text(0.5, 0.5, 'No category data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=24)
        return
    
    # Filter to treated samples only for main comparison
    treated_df = category_df[category_df['treatment'] == 'treated'].copy()
    if treated_df.empty:
        ax.text(0.5, 0.5, 'No treated category data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=24)
        return
    
    # Remove coding category from the plot
    treated_df = treated_df[treated_df['category'] != 'coding'].copy()
    
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
    # Use gradient: lightest for lowest rate, darkest for highest rate
    colors = []
    n_categories = len(treated_df)
    
    if n_categories == 1:
        colors.append(TREATED_COLORS[1])  # Medium color for single bar
    else:
        # Create gradient from light to dark based on position (darker = higher rate)
        for i in range(n_categories):
            # Map position to color gradient (0 = lightest, n-1 = darkest)
            color_idx = int((i / (n_categories - 1)) * (len(TREATED_COLORS) - 1))
            color_idx = min(color_idx, len(TREATED_COLORS) - 1)
            colors.append(TREATED_COLORS[color_idx])
    
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
    ax.set_ylabel('Mutation Rate (per base)', fontsize=28, fontweight='bold', labelpad=25)
    ax.set_yscale('log')
    # Set y-axis range to include all rates and their CI bounds
    min_rate = treated_df['rate'].min() * 0.9  # Slightly below minimum rate
    # Get maximum rate including CI_high bounds to ensure all error bars are visible
    max_rate = treated_df['rate'].max()
    if 'CI_high' in treated_df.columns:
        max_ci_high = treated_df['CI_high'].max()
        max_rate = max(max_rate, max_ci_high)
    # Add padding above the maximum
    ax.set_ylim(min_rate, max_rate * 1.3)  # 30% padding above max
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
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        print(f"Error: {args.output_dir} is not a directory")
        return 1
    
    create_multiplot(args.output_dir, args.output_path)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

