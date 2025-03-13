#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import glob
from matplotlib.colors import LinearSegmentedColormap
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Plot dN/dS analysis results across different mutation count filters')
    parser.add_argument('-i', '--input_dir', required=True, help='Directory containing dnds_summary_X.csv files')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for plots')
    parser.add_argument('-s', '--samples', help='Comma-separated list of sample names to highlight (optional)')
    parser.add_argument('--ems-only', action='store_true', help='Only include EMS samples in analysis')
    return parser.parse_args()

def load_filter_data(input_dir, ems_only=False):
    """Load data from all filter CSV files and combine into a single DataFrame."""
    # Find all summary CSV files
    csv_files = glob.glob(f"{input_dir}/dnds_summary_*.csv")
    
    if not csv_files:
        raise ValueError(f"No dnds_summary_*.csv files found in {input_dir}")
    
    # Extract filter values and load data
    all_data = []
    for csv_file in csv_files:
        # Extract filter value from filename
        match = re.search(r'dnds_summary_(\d+)\.csv', csv_file)
        if match:
            filter_value = int(match.group(1))
            
            # Load data
            df = pd.read_csv(csv_file)
            
            # Replace zeros with NaN to properly handle them in plots
            df['Original_dNdS'] = df['Original_dNdS'].replace(0, np.nan)
            
            df['Filter'] = filter_value
            
            # Add shortened sample IDs
            df['ShortID'] = df['Sample'].apply(extract_sample_id)
            
            # Filter for EMS samples if requested
            if ems_only:
                df = df[df['Sample'].str.contains('EMS', case=False)]
                
            all_data.append(df)
    
    # Combine all data
    if not all_data:
        raise ValueError("No valid data files found")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove rows with NaN dN/dS values
    combined_df = combined_df.dropna(subset=['Original_dNdS'])
    
    # Add dN and dS rates using hardcoded site counts
    syn_sites = 90149
    non_syn_sites = 262133
    
    combined_df['dN_Rate'] = combined_df['NonSyn_Mutations'] / non_syn_sites
    combined_df['dS_Rate'] = combined_df['Syn_Mutations'] / syn_sites
    
    # Try to load intergenic data if available
    try:
        intergenic_file = f"{input_dir}/intergeniccounts.json"
        if Path(intergenic_file).exists():
            print(f"Loading intergenic data from {intergenic_file}")
            with open(intergenic_file) as f:
                intergenic_data = json.load(f)
            
            # Create a dictionary to store intergenic rates by sample and filter
            intergenic_rates = {}
            
            # Process each sample's data
            for sample, sample_data in intergenic_data.items():
                intergenic_rates[sample] = {}
                
                # The JSON doesn't have filter levels as keys
                # Instead, we need to calculate rates for each filter level
                if 'count_bins' in sample_data and 'gc_sites' in sample_data:
                    gc_sites = sample_data['gc_sites']
                    count_bins = sample_data['count_bins']
                    
                    # Calculate rates for each filter level
                    for filter_value in range(1, 16):
                        # Sum all bins above the filter level
                        total_mutations = sum(
                            int(count) * int(bin_count) 
                            for count, bin_count in count_bins.items() 
                            if int(count) >= filter_value
                        )
                        
                        if gc_sites > 0:
                            intergenic_rates[sample][filter_value] = total_mutations / gc_sites
                        else:
                            intergenic_rates[sample][filter_value] = np.nan
            
            # Add intergenic rate to the dataframe based on sample and filter
            def get_intergenic_rate(row):
                sample = row['Sample']
                filter_value = row['Filter']
                rate = intergenic_rates.get(sample, {}).get(filter_value, np.nan)
                return rate
            
            combined_df['Intergenic_Rate'] = combined_df.apply(get_intergenic_rate, axis=1)
            
    except Exception as e:
        print(f"Warning: Could not load intergenic data: {e}")
        import traceback
        print(traceback.format_exc())
        combined_df['Intergenic_Rate'] = np.nan
    
    return combined_df

def extract_sample_id(sample):
    '''Extract abbreviated sample ID from full sample name.
    
    Args:
        sample (str): Full sample name
        
    Returns:
        str: Abbreviated sample identifier
    '''
    # First get the basic ID
    if 'EMS-' in sample:
        sample_id = sample.split('EMS-')[1]
        prefix = 'EMS'
    elif 'EMS' in sample:
        sample_id = sample.split('EMS')[1]
        prefix = 'EMS'
    elif 'NT-' in sample:
        sample_id = sample.split('NT-')[1]
        prefix = 'NT'
    elif 'NT' in sample:
        sample_id = sample.split('NT')[1]
        prefix = 'NT'
    else:
        return sample
        
    # Clean up the ID to just keep number and treatment time if present
    if '_' in sample_id:
        # Handle cases like "1_3d" or "6_7d"
        parts = sample_id.split('_')
        if len(parts) >= 2 and 'd' in parts[1]:
            return f"{prefix}{parts[0]}_{parts[1].split('_')[0]}"  # Keep just the number and days
        return f"{prefix}{parts[0]}"  # Just keep the number if no valid treatment time
    
    # Remove any trailing text after numbers
    number = re.match(r'\d+', sample_id)
    return f"{prefix}{number.group()}" if number else sample_id

def plot_enhanced_heatmap(df, output_dir):
    """Plot enhanced heatmap of dN/dS values by sample and filter."""
    # Create a copy with short IDs as index
    df_short = df.copy()
    
    # Pivot data for heatmap using short IDs
    pivot_df = df_short.pivot(index='ShortID', columns='Filter', values='Original_dNdS')
    
    # Sort by average dN/dS value (ignoring NaN values)
    pivot_df['avg_dnds'] = pivot_df.mean(axis=1, skipna=True)
    pivot_df = pivot_df.sort_values('avg_dnds', ascending=False)
    pivot_df = pivot_df.drop('avg_dnds', axis=1)
    
    # Create a custom colormap that's white at 1.0 with a wider range
    # Blue for values < 1, white at 1, and gradient from white to red to dark red for values > 1
    cmap = LinearSegmentedColormap.from_list(
        'custom_dnds_cmap',
        [(0, 'blue'), 
         (0.2, 'lightblue'),
         (0.5, 'white'),  # White at 1.0 (normalized)
         (0.6, 'pink'),
         (0.7, 'red'),
         (0.85, 'darkred'),
         (1.0, 'black')],  # Very high values approach black
        N=256
    )
    
    # Calculate vmin and vmax for better color scaling
    vmin = max(0.1, pivot_df.min().min())
    vmax = 5.0  # Set maximum to 5.0 to better visualize the range
    
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(pivot_df, cmap=cmap, annot=True, fmt='.2f', 
                     vmin=vmin, vmax=vmax, center=1.0)
    
    # Improve readability of annotations
    for text in ax.texts:
        text.set_size(8)
        # Make text white for very dark cells
        cell_value = float(text.get_text())
        if cell_value > 3.0:
            text.set_color('white')
    
    plt.title('dN/dS Values by Sample and Filter', fontsize=16)
    plt.xlabel('Mutation Count Filter', fontsize=14)
    plt.ylabel('Sample', fontsize=14)
    
    # Add a colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label('dN/dS Ratio', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dnds_heatmap.png", dpi=300)
    plt.close()

def plot_boxplot_by_filter(df, output_dir):
    """Plot boxplot of dN/dS values by filter."""
    # Make a copy to avoid modifying the original dataframe
    plot_df = df.copy()
    
    plt.figure(figsize=(14, 8))
    
    # Create boxplot
    sns.boxplot(x='Filter', y='Original_dNdS', data=plot_df)
    
    # Add individual points
    sns.stripplot(x='Filter', y='Original_dNdS', data=plot_df, 
                  size=4, color='black', alpha=0.5)
    
    # Add horizontal line at dN/dS = 1
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Mutation Count Filter', fontsize=14)
    plt.ylabel('dN/dS Ratio', fontsize=14)
    plt.title('Distribution of dN/dS Ratios by Mutation Count Filter', fontsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits based on data range
    data_min = plot_df['Original_dNdS'].min()
    data_max = plot_df['Original_dNdS'].max()
    data_range = data_max - data_min
    
    # Set limits with padding
    y_min = max(0, data_min - 0.05 * data_range)
    y_max = data_max + 0.1 * data_range
    
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dnds_boxplot.png", dpi=300)
    plt.close()

def plot_violin_by_filter(df, output_dir):
    """Plot violin plot of dN/dS values by filter."""
    # Make a copy to avoid modifying the original dataframe
    plot_df = df.copy()
    
    # Count samples per filter to determine which filters to include
    filter_counts = plot_df.groupby('Filter').size()
    valid_filters = filter_counts[filter_counts >= 3].index.tolist()
    
    # Only include filters with enough data for a meaningful violin plot
    plot_df = plot_df[plot_df['Filter'].isin(valid_filters)]
    
    plt.figure(figsize=(14, 8))
    
    # Create violin plot
    sns.violinplot(x='Filter', y='Original_dNdS', data=plot_df, inner='quartile')
    
    # Add individual points
    sns.stripplot(x='Filter', y='Original_dNdS', data=plot_df, 
                  size=4, color='black', alpha=0.5)
    
    # Add horizontal line at dN/dS = 1
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Mutation Count Filter', fontsize=14)
    plt.ylabel('dN/dS Ratio', fontsize=14)
    plt.title('Distribution of dN/dS Ratios by Mutation Count Filter', fontsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits based on data range
    data_min = plot_df['Original_dNdS'].min()
    data_max = plot_df['Original_dNdS'].max()
    data_range = data_max - data_min
    
    # Set limits with padding
    y_min = max(0, data_min - 0.05 * data_range)
    y_max = data_max + 0.1 * data_range
    
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dnds_violin.png", dpi=300)
    plt.close()

def plot_sample_trajectories(df, output_dir, max_dnds=20.0):
    """Plot sample trajectories with filter value on x-axis and dN/dS on y-axis."""
    # Filter out extreme values that are likely noise
    plot_df = df[df['Original_dNdS'] <= max_dnds].copy()
    
    plt.figure(figsize=(14, 8))
    
    # Get unique samples and create a colormap
    unique_samples = plot_df['ShortID'].unique()
    
    # Group samples by EMS vs NT
    ems_samples = [s for s in unique_samples if 'EMS' in s]
    nt_samples = [s for s in unique_samples if 'NT' in s]
    other_samples = [s for s in unique_samples if 'EMS' not in s and 'NT' not in s]
    
    # Plot EMS samples in red
    for sample in ems_samples:
        sample_data = plot_df[plot_df['ShortID'] == sample].sort_values('Filter')
        if len(sample_data) > 1:  # Only plot if we have at least 2 points
            plt.plot(sample_data['Filter'], sample_data['Original_dNdS'], 
                     color='red', alpha=0.7, linewidth=1)
    
    # Plot NT samples in blue
    for sample in nt_samples:
        sample_data = plot_df[plot_df['ShortID'] == sample].sort_values('Filter')
        if len(sample_data) > 1:  # Only plot if we have at least 2 points
            plt.plot(sample_data['Filter'], sample_data['Original_dNdS'], 
                     color='blue', alpha=0.7, linewidth=1)
    
    # Plot other samples in gray
    for sample in other_samples:
        sample_data = plot_df[plot_df['ShortID'] == sample].sort_values('Filter')
        if len(sample_data) > 1:  # Only plot if we have at least 2 points
            plt.plot(sample_data['Filter'], sample_data['Original_dNdS'], 
                     color='gray', alpha=0.7, linewidth=1)
    
    # Add horizontal line at dN/dS = 1
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7)
    
    # Add average line
    avg_by_filter = plot_df.groupby('Filter')['Original_dNdS'].mean().reset_index()
    plt.plot(avg_by_filter['Filter'], avg_by_filter['Original_dNdS'], 
             color='black', linewidth=3, label='Average')
    
    # Add legend for sample types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='EMS Samples'),
        Line2D([0], [0], color='blue', lw=2, label='NT Samples'),
        Line2D([0], [0], color='gray', lw=2, label='Other Samples'),
        Line2D([0], [0], color='black', lw=2, label='Average')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.xlabel('Mutation Count Filter', fontsize=14)
    plt.ylabel('dN/dS Ratio', fontsize=14)
    plt.title('dN/dS Ratio Trajectories by Sample', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits based on data range
    data_min = plot_df['Original_dNdS'].min()
    data_max = plot_df['Original_dNdS'].max()
    data_range = data_max - data_min
    
    # Set limits with padding
    y_min = max(0, data_min - 0.05 * data_range)
    y_max = data_max + 0.1 * data_range
    
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dnds_trajectories.png", dpi=300)
    plt.close()

def plot_detailed_filter_analysis(df, output_dir, max_filters=10, max_dnds=20.0):
    """Create detailed plots for the first N filter values.
    
    Each plot shows:
    1. Original vs Random dN/dS with confidence intervals
    2. Scatter plot of mutation count vs dN/dS ratio with sample labels
    """
    # Filter out extreme values
    plot_df = df[df['Original_dNdS'] <= max_dnds].copy()
    
    # Get available filters and sort them
    available_filters = sorted(plot_df['Filter'].unique())
    
    # Only use the first max_filters values
    filters_to_plot = available_filters[:max_filters]
    
    for filter_value in filters_to_plot:
        # Get data for this filter
        filter_data = plot_df[plot_df['Filter'] == filter_value].copy()
        
        if len(filter_data) < 3:  # Skip if not enough data
            continue
            
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Sort by ShortID for consistent ordering
        filter_data = filter_data.sort_values('ShortID')
        
        # Add sample type column for coloring
        filter_data['SampleType'] = 'Other'
        filter_data.loc[filter_data['ShortID'].str.contains('EMS'), 'SampleType'] = 'EMS'
        filter_data.loc[filter_data['ShortID'].str.contains('NT'), 'SampleType'] = 'NT'
        
        # Create a column with just the numeric part of the sample ID (without EMS/NT prefix)
        filter_data['NumericID'] = filter_data['ShortID'].str.replace('EMS', '').str.replace('NT', '')
        
        # Define colors for sample types
        color_map = {'EMS': 'red', 'NT': 'blue', 'Other': 'gray'}
        
        # 1. Plot Original vs Random dN/dS with confidence intervals
        x = np.arange(len(filter_data))
        width = 0.35
        
        # Plot bars for original dN/dS
        bars1 = ax1.bar(x - width/2, filter_data['Original_dNdS'], width, 
                        label='Original dN/dS', 
                        color=[color_map[t] for t in filter_data['SampleType']])
        
        # Plot bars for random dN/dS
        bars2 = ax1.bar(x + width/2, filter_data['Random_dNdS'], width, 
                        label='Random dN/dS', color='lightgray')
        
        # Add confidence interval error bars for original dN/dS
        if 'Original_dNdS_Lower' in filter_data.columns and 'Original_dNdS_Upper' in filter_data.columns:
            yerr_orig = np.array([
                filter_data['Original_dNdS'] - filter_data['Original_dNdS_Lower'],
                filter_data['Original_dNdS_Upper'] - filter_data['Original_dNdS']
            ])
            ax1.errorbar(x - width/2, filter_data['Original_dNdS'], yerr=yerr_orig, 
                         fmt='none', ecolor='black', capsize=3)
        
        # Add confidence interval error bars for random dN/dS
        if 'Random_dNdS_Lower' in filter_data.columns and 'Random_dNdS_Upper' in filter_data.columns:
            yerr_rand = np.array([
                filter_data['Random_dNdS'] - filter_data['Random_dNdS_Lower'],
                filter_data['Random_dNdS_Upper'] - filter_data['Random_dNdS']
            ])
            ax1.errorbar(x + width/2, filter_data['Random_dNdS'], yerr=yerr_rand, 
                         fmt='none', ecolor='black', capsize=3)
        
        # Add horizontal line at dN/dS = 1
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.7)
        
        # Customize plot
        ax1.set_ylabel('dN/dS Ratio', fontsize=12)
        ax1.set_title(f'Original vs Random dN/dS (Filter = {filter_value})', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(filter_data['ShortID'], rotation=45, ha='right', fontsize=8)
        ax1.legend()
        
        # Set y-axis limits
        y_max = min(5.0, filter_data['Original_dNdS'].max() * 1.2)
        ax1.set_ylim(0, y_max)
        
        # 2. Scatter plot of mutation count vs dN/dS ratio with sample labels
        for sample_type, color in color_map.items():
            type_data = filter_data[filter_data['SampleType'] == sample_type]
            if not type_data.empty:
                ax2.scatter(type_data['Total_Mutations'], type_data['Original_dNdS'], 
                           color=color, alpha=0.7, label=sample_type)
                
                # Add sample labels (numeric part only)
                for i, row in type_data.iterrows():
                    ax2.annotate(row['NumericID'], 
                                (row['Total_Mutations'], row['Original_dNdS']),
                                xytext=(5, 0), textcoords='offset points',
                                fontsize=8, color=color)
        
        # Add horizontal line at dN/dS = 1
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7)
        
        # Customize plot
        ax2.set_xlabel('Total Mutations', fontsize=12)
        ax2.set_ylabel('dN/dS Ratio', fontsize=12)
        ax2.set_title(f'Mutation Count vs dN/dS (Filter = {filter_value})', fontsize=14)
        ax2.legend()
        
        # Set y-axis limits to match the first plot
        ax2.set_ylim(ax1.get_ylim())
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detailed_filter_{filter_value}.png", dpi=300)
        plt.close()

def plot_mutation_rate_comparison(df, output_dir, filter_values=[1, 3, 5]):
    """Create grouped bar plots comparing dN, dS, and intergenic mutation rates.
    
    Args:
        df (DataFrame): Combined data with dN/dS values and intergenic rates
        output_dir (str): Directory to save plots
        filter_values (list): List of filter values to plot
    """
    # Ensure we have the required columns
    required_columns = ['Sample', 'ShortID', 'Filter', 'dN_Rate', 'dS_Rate', 'Intergenic_Rate']
    if not all(col in df.columns for col in required_columns):
        print("Warning: Missing required columns for mutation rate comparison plot")
        print(f"Required: {required_columns}")
        print(f"Available: {df.columns.tolist()}")
        return
    
    # Filter for specific filter values
    plot_df = df[df['Filter'].isin(filter_values)].copy()
    
    if plot_df.empty:
        print(f"No data found for filter values {filter_values}")
        return
    
    # Create a figure for each filter value
    for filter_value in filter_values:
        filter_data = plot_df[plot_df['Filter'] == filter_value].copy()
        
        if filter_data.empty:
            print(f"No data for filter value {filter_value}")
            continue
        
        # Sort by ShortID for consistent ordering
        filter_data = filter_data.sort_values('ShortID')
        
        # Add sample type column for coloring
        filter_data['SampleType'] = 'Other'
        filter_data.loc[filter_data['ShortID'].str.contains('EMS'), 'SampleType'] = 'EMS'
        filter_data.loc[filter_data['ShortID'].str.contains('NT'), 'SampleType'] = 'NT'
        
        # Define color maps for different sample types
        ems_colors = {'dN': '#8B0000', 'dS': '#FF4500', 'Intergenic': '#32CD32'}  # Dark red, Orange-red, Lime green
        nt_colors = {'dN': '#00008B', 'dS': '#1E90FF', 'Intergenic': '#32CD32'}   # Dark blue, Dodger blue, Lime green
        other_colors = {'dN': '#696969', 'dS': '#A9A9A9', 'Intergenic': '#32CD32'}  # Dark gray, Light gray, Lime green
        
        # Set up the figure
        plt.figure(figsize=(14, 8))
        
        # Set up bar positions
        x = np.arange(len(filter_data))
        width = 0.25  # narrower bars for 3 groups
        
        # Create bars
        ax = plt.gca()
        
        # Create empty lists to store bars for legend
        ems_dn_bar = None
        ems_ds_bar = None
        nt_dn_bar = None
        nt_ds_bar = None
        intergenic_bars = []
        
        # Plot dN rate bars
        for i, (idx, row) in enumerate(filter_data.iterrows()):
            if row['SampleType'] == 'EMS':
                bar = ax.bar(x[i] - width, row['dN_Rate'], width, alpha=0.8, color=ems_colors['dN'])
                if ems_dn_bar is None:
                    ems_dn_bar = bar
            elif row['SampleType'] == 'NT':
                bar = ax.bar(x[i] - width, row['dN_Rate'], width, alpha=0.8, color=nt_colors['dN'])
                if nt_dn_bar is None:
                    nt_dn_bar = bar
            else:
                ax.bar(x[i] - width, row['dN_Rate'], width, alpha=0.8, color=other_colors['dN'])
        
        # Plot dS rate bars
        for i, (idx, row) in enumerate(filter_data.iterrows()):
            if row['SampleType'] == 'EMS':
                bar = ax.bar(x[i], row['dS_Rate'], width, alpha=0.8, color=ems_colors['dS'])
                if ems_ds_bar is None:
                    ems_ds_bar = bar
            elif row['SampleType'] == 'NT':
                bar = ax.bar(x[i], row['dS_Rate'], width, alpha=0.8, color=nt_colors['dS'])
                if nt_ds_bar is None:
                    nt_ds_bar = bar
            else:
                ax.bar(x[i], row['dS_Rate'], width, alpha=0.8, color=other_colors['dS'])
        
        # Plot intergenic rate bars
        for i, (idx, row) in enumerate(filter_data.iterrows()):
            bar = ax.bar(x[i] + width, row['Intergenic_Rate'], width, alpha=0.8, color=ems_colors['Intergenic'])
            intergenic_bars.append(bar)
        
        # Create custom legend
        legend_elements = []
        if ems_dn_bar is not None:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, color=ems_colors['dN'], alpha=0.8, label='EMS dN Rate'))
        if ems_ds_bar is not None:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, color=ems_colors['dS'], alpha=0.8, label='EMS dS Rate'))
        if nt_dn_bar is not None:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, color=nt_colors['dN'], alpha=0.8, label='NT dN Rate'))
        if nt_ds_bar is not None:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, color=nt_colors['dS'], alpha=0.8, label='NT dS Rate'))
        if intergenic_bars:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, color=ems_colors['Intergenic'], alpha=0.8, label='Intergenic Rate'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Customize plot
        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('Mutation Rate (per site)', fontsize=12)
        ax.set_title(f'Mutation Rates Comparison (Filter = {filter_value})', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(filter_data['ShortID'], rotation=45, ha='right', fontsize=8)
        
        # Set y-axis limits based on data range
        data_values = np.concatenate([
            filter_data['dN_Rate'].values,
            filter_data['dS_Rate'].values,
            filter_data['Intergenic_Rate'].values
        ])
        data_values = data_values[~np.isnan(data_values)]  # Remove NaN values
        
        if len(data_values) > 0:
            data_min = np.min(data_values)
            data_max = np.max(data_values)
            data_range = data_max - data_min
            
            # Set limits with padding
            y_min = max(0, data_min - 0.05 * data_range)
            y_max = data_max + 0.1 * data_range
            
            ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mutation_rate_comparison_filter_{filter_value}.png", dpi=300)
        plt.close()

def plot_multipanel_rate_comparison(df, output_dir, filter_values=[1, 3, 5]):
    """Create a multipanel figure comparing mutation rates across multiple filter levels.
    
    Args:
        df (DataFrame): Combined data with dN/dS values and intergenic rates
        output_dir (str): Directory to save plots
        filter_values (list): List of filter values to plot
    """
    # Ensure we have the required columns
    required_columns = ['Sample', 'ShortID', 'Filter', 'dN_Rate', 'dS_Rate', 'Intergenic_Rate']
    if not all(col in df.columns for col in required_columns):
        print("Warning: Missing required columns for multipanel rate comparison plot")
        print(f"Required: {required_columns}")
        print(f"Available: {df.columns.tolist()}")
        return
    
    # Filter for specific filter values
    plot_df = df[df['Filter'].isin(filter_values)].copy()
    
    if plot_df.empty:
        print(f"No data found for filter values {filter_values}")
        return
    
    # Create a multipanel figure
    fig, axes = plt.subplots(1, len(filter_values), figsize=(18, 8), sharey=True)
    
    # Define color maps for different sample types
    ems_colors = {'dN': '#8B0000', 'dS': '#FF4500', 'Intergenic': '#32CD32'}  # Dark red, Orange-red, Lime green
    nt_colors = {'dN': '#00008B', 'dS': '#1E90FF', 'Intergenic': '#32CD32'}   # Dark blue, Dodger blue, Lime green
    other_colors = {'dN': '#696969', 'dS': '#A9A9A9', 'Intergenic': '#32CD32'}  # Dark gray, Light gray, Lime green
    
    # Track legend elements
    legend_elements = []
    ems_dn_added = False
    ems_ds_added = False
    nt_dn_added = False
    nt_ds_added = False
    intergenic_added = False
    
    # Process each filter value
    for i, filter_value in enumerate(filter_values):
        ax = axes[i]
        filter_data = plot_df[plot_df['Filter'] == filter_value].copy()
        
        if filter_data.empty:
            ax.text(0.5, 0.5, f"No data for filter {filter_value}", 
                    ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Sort by ShortID for consistent ordering
        filter_data = filter_data.sort_values('ShortID')
        
        # Add sample type column for coloring
        filter_data['SampleType'] = 'Other'
        filter_data.loc[filter_data['ShortID'].str.contains('EMS'), 'SampleType'] = 'EMS'
        filter_data.loc[filter_data['ShortID'].str.contains('NT'), 'SampleType'] = 'NT'
        
        # Set up bar positions
        x = np.arange(len(filter_data))
        width = 0.25  # narrower bars for 3 groups
        
        # Plot dN rate bars
        for j, (idx, row) in enumerate(filter_data.iterrows()):
            if row['SampleType'] == 'EMS':
                bar = ax.bar(x[j] - width, row['dN_Rate'], width, alpha=0.8, color=ems_colors['dN'])
                if not ems_dn_added:
                    legend_elements.append(plt.Rectangle((0,0), 1, 1, color=ems_colors['dN'], alpha=0.8, label='EMS dN Rate'))
                    ems_dn_added = True
            elif row['SampleType'] == 'NT':
                bar = ax.bar(x[j] - width, row['dN_Rate'], width, alpha=0.8, color=nt_colors['dN'])
                if not nt_dn_added:
                    legend_elements.append(plt.Rectangle((0,0), 1, 1, color=nt_colors['dN'], alpha=0.8, label='NT dN Rate'))
                    nt_dn_added = True
            else:
                ax.bar(x[j] - width, row['dN_Rate'], width, alpha=0.8, color=other_colors['dN'])
        
        # Plot dS rate bars
        for j, (idx, row) in enumerate(filter_data.iterrows()):
            if row['SampleType'] == 'EMS':
                bar = ax.bar(x[j], row['dS_Rate'], width, alpha=0.8, color=ems_colors['dS'])
                if not ems_ds_added:
                    legend_elements.append(plt.Rectangle((0,0), 1, 1, color=ems_colors['dS'], alpha=0.8, label='EMS dS Rate'))
                    ems_ds_added = True
            elif row['SampleType'] == 'NT':
                bar = ax.bar(x[j], row['dS_Rate'], width, alpha=0.8, color=nt_colors['dS'])
                if not nt_ds_added:
                    legend_elements.append(plt.Rectangle((0,0), 1, 1, color=nt_colors['dS'], alpha=0.8, label='NT dS Rate'))
                    nt_ds_added = True
            else:
                ax.bar(x[j], row['dS_Rate'], width, alpha=0.8, color=other_colors['dS'])
        
        # Plot intergenic rate bars
        for j, (idx, row) in enumerate(filter_data.iterrows()):
            bar = ax.bar(x[j] + width, row['Intergenic_Rate'], width, alpha=0.8, color=ems_colors['Intergenic'])
            if not intergenic_added:
                legend_elements.append(plt.Rectangle((0,0), 1, 1, color=ems_colors['Intergenic'], alpha=0.8, label='Intergenic Rate'))
                intergenic_added = True
        
        # Customize subplot
        ax.set_xlabel('Sample', fontsize=12)
        if i == 0:  # Only add y-label to the first subplot
            ax.set_ylabel('Mutation Rate (per site)', fontsize=12)
        ax.set_title(f'Filter = {filter_value}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(filter_data['ShortID'], rotation=45, ha='right', fontsize=8)
        
        # Calculate y-axis limits for this panel
        data_values = np.concatenate([
            filter_data['dN_Rate'].values,
            filter_data['dS_Rate'].values,
            filter_data['Intergenic_Rate'].values
        ])
        data_values = data_values[~np.isnan(data_values)]  # Remove NaN values
        
        if len(data_values) > 0:
            # Store max value for later use in setting global y-axis limits
            if i == 0:
                global_max = np.max(data_values)
            else:
                global_max = max(global_max, np.max(data_values))
    
    # Set common y-axis limit
    for ax in axes:
        ax.set_ylim(0, global_max * 1.1)  # Add 10% padding
    
    # Add a single legend for the entire figure
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
               fancybox=True, shadow=True, ncol=5)
    
    # Add overall title
    fig.suptitle('Mutation Rates Comparison Across Filter Levels', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to make room for legend and title
    plt.savefig(f"{output_dir}/multipanel_rate_comparison.png", dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input_dir}...")
    df = load_filter_data(args.input_dir, args.ems_only)
    
    # Generate plots
    print("Generating plots...")
    plot_enhanced_heatmap(df, output_dir)
    plot_boxplot_by_filter(df, output_dir)
    plot_violin_by_filter(df, output_dir)
    plot_sample_trajectories(df, output_dir, max_dnds=20.0)
    plot_detailed_filter_analysis(df, output_dir, max_filters=10, max_dnds=20.0)
    
    # Generate mutation rate comparison plots
    print("Generating mutation rate comparison plots...")
    plot_mutation_rate_comparison(df, output_dir, filter_values=[1, 3, 5])
    
    # Generate multipanel comparison plot
    print("Generating multipanel rate comparison plot...")
    plot_multipanel_rate_comparison(df, output_dir, filter_values=[1, 3, 5])
    
    print(f"Plots saved to {output_dir}")

if __name__ == '__main__':
    main() 