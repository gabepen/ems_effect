#!/usr/bin/env python3
"""
Prediction accuracy metrics for GLM models.

Computes various metrics on held-out test data:
- MSE, RMSE
- R² (pseudo-R² for GLMs)
- Correlation (Pearson, Spearman)
- Poisson-specific metrics
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_prediction_metrics(
    y_observed: np.ndarray,
    y_predicted: np.ndarray,
    mu_predicted: np.ndarray = None,  # Mean predictions for GLM
    nb_alpha: float = None  # Negative Binomial dispersion parameter (if None, uses Poisson deviance)
) -> Dict:
    """
    Compute comprehensive prediction accuracy metrics.
    
    Args:
        y_observed: Actual observed counts
        y_predicted: Predicted counts (can be mean or integer predictions)
        mu_predicted: Mean predictions (if different from y_predicted)
        nb_alpha: Negative Binomial dispersion parameter. If provided, calculates NB deviance instead of Poisson.
    
    Returns:
        Dictionary with all metrics
    """
    if mu_predicted is None:
        mu_predicted = y_predicted
    
    # Basic metrics
    mse = np.mean((y_observed - mu_predicted) ** 2)
    mae = np.mean(np.abs(y_observed - mu_predicted))
    rmse = np.sqrt(mse)
    
    # Relative errors
    mean_obs = np.mean(y_observed)
    mape = np.mean(np.abs((y_observed - mu_predicted) / (mean_obs + 1e-10))) * 100  # Mean absolute percentage error
    
    # Correlation metrics
    if len(y_observed) > 1:
        pearson_r, pearson_p = stats.pearsonr(y_observed, mu_predicted)
        spearman_r, spearman_p = stats.spearmanr(y_observed, mu_predicted)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
    
    # R² (coefficient of determination)
    ss_res = np.sum((y_observed - mu_predicted) ** 2)
    ss_tot = np.sum((y_observed - mean_obs) ** 2)
    r_squared = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Adjusted R² (accounting for parameters)
    # For now, use same formula (would need n_params)
    adj_r_squared = r_squared  # Could be improved with parameter count
    
    # Deviance calculation: NB if alpha provided, otherwise Poisson
    mu_safe = np.maximum(mu_predicted, 1e-10)
    y_safe = np.maximum(y_observed, 0)
    
    if nb_alpha is not None and nb_alpha > 0:
        # Negative Binomial deviance
        # Formula: 2 * Σ [y * log(y/μ) - (y + 1/α) * log((y + 1/α)/(μ + 1/α))]
        inv_alpha = 1.0 / nb_alpha
        y_plus_inv_alpha = y_safe + inv_alpha
        mu_plus_inv_alpha = mu_safe + inv_alpha
        
        # Handle zeros carefully
        y_log_term = np.where(y_safe > 0, y_safe * np.log((y_safe + 1e-10) / mu_safe), 0)
        combined_log_term = y_plus_inv_alpha * np.log((y_plus_inv_alpha + 1e-10) / (mu_plus_inv_alpha + 1e-10))
        
        nb_deviance = 2 * np.sum(y_log_term - combined_log_term)
        poisson_deviance = nb_deviance  # Use NB deviance as the main metric
    else:
        # Poisson deviance (fallback for Poisson models or when alpha not provided)
        poisson_deviance = 2 * np.sum(
            y_safe * np.log((y_safe + 1e-10) / mu_safe) - (y_safe - mu_safe)
        )
    
    # Mean squared log error (MSLE) - good for count data
    y_obs_log = np.log1p(y_observed)  # log(1+x) to handle zeros
    mu_pred_log = np.log1p(mu_predicted)
    msle = np.mean((y_obs_log - mu_pred_log) ** 2)
    
    # Root mean squared log error
    rmsle = np.sqrt(msle)
    
    # Classification metrics for low vs high counts
    # Define thresholds (e.g., median split)
    median_obs = np.median(y_observed)
    high_obs = y_observed > median_obs
    high_pred = mu_predicted > median_obs
    
    # Accuracy
    accuracy = np.mean(high_obs == high_pred)
    
    # Precision, recall (treating "high" as positive class)
    tp = np.sum(high_obs & high_pred)
    fp = np.sum(~high_obs & high_pred)
    fn = np.sum(high_obs & ~high_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        # Basic metrics
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        
        # Correlation
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        
        # R²
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        
        # Deviance (NB if nb_alpha provided, otherwise Poisson)
        'deviance': poisson_deviance,
        'poisson_deviance': poisson_deviance,  # Keep for backwards compatibility
        'msle': msle,
        'rmsle': rmsle,
        
        # Classification (high/low)
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        
        # Summary stats
        'n_observations': len(y_observed),
        'mean_observed': mean_obs,
        'mean_predicted': np.mean(mu_predicted),
    }


def predict_with_7mer_splits(
    split_models: list,
    df_test: pd.DataFrame
) -> np.ndarray:
    """
    Predict using 7mer split models.
    
    Args:
        split_models: List of split model dicts (from fit_7mer_model_splits)
        df_test: DataFrame with test data (must have kmer7, treatment, depth)
    
    Returns:
        Array of predictions (same length as df_test, indexed by df_test.index)
    """
    import statsmodels.api as sm
    
    df7 = df_test[df_test['kmer7'].notna()].copy()
    if df7.empty:
        # If no 7mer data, return zeros
        return np.zeros(len(df_test))
    
    if 'log_depth' not in df7.columns:
        df7['log_depth'] = np.log(df7['depth'])
    
    # Initialize predictions dictionary (index -> prediction)
    predictions_dict = {}
    
    # Predict for each split
    for split_model in split_models:
        if 'model' not in split_model or 'group' not in split_model or 'want_cols' not in split_model:
            continue
        
        model = split_model['model']
        group = split_model['group']
        want_cols = split_model['want_cols']
        
        # Get test data for this group
        sub_test = df7[df7['kmer7'].isin(group)].copy()
        if sub_test.empty:
            continue
        
        # Create one-hot features for this group
        dummies_test = pd.DataFrame(0, index=sub_test.index, columns=want_cols, dtype='uint8')
        for k in group:
            col = f'7mer_{k}'
            if col in dummies_test.columns:
                dummies_test[col] = (sub_test['kmer7'] == k).astype('uint8')
        
        sub_test[dummies_test.columns] = dummies_test
        
        X_test = sub_test[['treatment'] + want_cols]
        offset_test = sub_test['log_depth'].values
        
        try:
            X_test_df = pd.DataFrame(X_test, columns=['treatment'] + want_cols)
            design_test = sm.add_constant(X_test_df, has_constant='add')
            mu_test = model.predict(design_test, offset=offset_test)
            
            # Store predictions by index
            for idx, pred in zip(sub_test.index, mu_test):
                predictions_dict[idx] = pred
        except Exception as e:
            print(f"Warning: Failed to predict for 7mer split group {group}: {e}")
            continue
    
    # Convert to array aligned with df_test
    predictions = np.array([predictions_dict.get(idx, np.nan) for idx in df_test.index])
    
    # Handle rows without 7mer (use mean of predictions)
    if np.any(np.isnan(predictions)):
        mean_pred = np.nanmean(predictions)
        if not np.isnan(mean_pred):
            predictions[np.isnan(predictions)] = mean_pred
        else:
            predictions[np.isnan(predictions)] = 0
    
    return predictions


def evaluate_model_on_test_set(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    offset_test: np.ndarray,
    model_name: str = "model"
) -> Dict:
    """
    Evaluate a fitted GLM model on test data.
    
    Returns dictionary with all prediction metrics.
    """
    # Prepare design matrix
    design_test = sm.add_constant(X_test, has_constant='add')
    
    # Get predictions (mean expected counts)
    mu_pred = model.predict(design_test, offset=offset_test)
    
    # Extract nb_alpha from model if it's a Negative Binomial model
    nb_alpha = None
    # Check if it's a GLM NB model (has family.alpha)
    if hasattr(model, 'family') and hasattr(model.family, 'alpha'):
        nb_alpha = model.family.alpha
    # Check if it's an NBDiscrete model (statsmodels.discrete.discrete_model.NegativeBinomial)
    elif 'NegativeBinomial' in str(type(model)) or 'NBDiscrete' in str(type(model)):
        try:
            # For NBDiscrete models, alpha is the last parameter
            if hasattr(model, 'params') and len(model.params) > 0:
                nb_alpha = float(model.params.iloc[-1])
        except:
            pass
    
    # Compute all metrics
    metrics = compute_prediction_metrics(y_test, mu_pred, mu_pred, nb_alpha=nb_alpha)
    metrics['model_name'] = model_name
    
    return metrics


def compare_models_predictions(
    models: Dict,
    df_test: pd.DataFrame,
    output_dir: str = None,
    metadata: Dict = None,
    cv_results_path: str = None
) -> pd.DataFrame:
    """
    Compare prediction accuracy across multiple models.
    
    Args:
        models: Dictionary of {model_name: fitted_model}
        df_test: DataFrame with test data (must have: treatment, depth, ems_count, kmer5)
        output_dir: Output directory for results
        metadata: Optional metadata dict with feature column names
    
    Returns DataFrame with metrics for each model.
    """
    df_test = df_test[df_test['depth'] > 0].copy()
    df_test['log_depth'] = np.log(df_test['depth'])
    y_test = df_test['ems_count'].values
    offset_test = df_test['log_depth'].values
    
    all_metrics = []
    
    for model_name, model in models.items():
        if model is None:
            continue
        
        print(f"Evaluating {model_name} on test set...")
        
        # Debug: Check if this looks like a 7mer model
        if '7mer' in model_name.lower():
            print(f"  Model name contains '7mer', model type: {type(model)}")
            if isinstance(model, list):
                print(f"  Model is a list with {len(model)} elements")
            else:
                print(f"  Warning: 7mer model is not a list - may not be handled correctly")
        
        try:
            # Special handling for 7mer_split which uses split models
            # Check for 7mer in model name (could be '7mer_split' or '7mer')
            is_7mer_model = ('7mer' in model_name.lower()) and isinstance(model, list)
            if is_7mer_model:
                print(f"  Detected 7mer model with {len(model)} split models")
                # Use split model prediction
                mu_pred = predict_with_7mer_splits(model, df_test)
                print(f"  Generated {len(mu_pred)} predictions, {np.sum(~np.isnan(mu_pred))} valid")
                # Filter to rows where we have 7mer data
                df7_test = df_test[df_test['kmer7'].notna()].copy()
                print(f"  Found {len(df7_test)} rows with 7mer data out of {len(df_test)} total rows")
                if df7_test.empty:
                    print(f"  Warning: No 7mer data in test set, skipping {model_name}")
                    continue
                
                # Get predictions and actual values for 7mer rows
                # predictions array is aligned with df_test.index
                y_test_7mer = df7_test['ems_count'].values
                
                # Create mask for 7mer rows in original df_test
                mask_7mer = df_test.index.isin(df7_test.index)
                mu_pred_7mer = mu_pred[mask_7mer]
                
                # Ensure same length and order - df7_test should be in same order as mask
                if len(mu_pred_7mer) != len(y_test_7mer):
                    # If lengths don't match, try to align by index
                    # Create a series with predictions indexed by df_test.index
                    pred_series = pd.Series(mu_pred, index=df_test.index)
                    mu_pred_7mer = pred_series.loc[df7_test.index].values
                
                # Remove any NaN values
                valid_mask = ~np.isnan(mu_pred_7mer)
                if np.sum(valid_mask) < len(mu_pred_7mer) * 0.9:  # Less than 90% valid
                    print(f"  Warning: Too many missing predictions for {model_name} ({np.sum(valid_mask)}/{len(mu_pred_7mer)} valid), skipping")
                    continue
                
                mu_pred_7mer = mu_pred_7mer[valid_mask]
                y_test_7mer = y_test_7mer[valid_mask]
                
                if len(mu_pred_7mer) == 0 or len(mu_pred_7mer) != len(y_test_7mer):
                    print(f"  Warning: Length mismatch for {model_name} ({len(mu_pred_7mer)} vs {len(y_test_7mer)}), skipping")
                    continue
                
                # Extract nb_alpha from 7mer split model metadata if available
                nb_alpha = None
                if isinstance(model, list) and len(model) > 0:
                    # Check first split model for nb_alpha in metadata
                    first_split = model[0]
                    if isinstance(first_split, dict) and 'nb_alpha' in first_split:
                        nb_alpha = first_split['nb_alpha']
                    # Or try to extract from the actual model object
                    elif isinstance(first_split, dict) and 'model' in first_split:
                        split_model = first_split['model']
                        # Check if it's a GLM NB model
                        if hasattr(split_model, 'family') and hasattr(split_model.family, 'alpha'):
                            nb_alpha = split_model.family.alpha
                        # Check if it's an NBDiscrete model
                        elif 'NegativeBinomial' in str(type(split_model)) or 'NBDiscrete' in str(type(split_model)):
                            try:
                                if hasattr(split_model, 'params') and len(split_model.params) > 0:
                                    nb_alpha = float(split_model.params.iloc[-1])
                            except:
                                pass
                    # If first_split is directly a model object (not a dict)
                    elif not isinstance(first_split, dict):
                        split_model = first_split
                        # Check if it's a GLM NB model
                        if hasattr(split_model, 'family') and hasattr(split_model.family, 'alpha'):
                            nb_alpha = split_model.family.alpha
                        # Check if it's an NBDiscrete model
                        elif 'NegativeBinomial' in str(type(split_model)) or 'NBDiscrete' in str(type(split_model)):
                            try:
                                if hasattr(split_model, 'params') and len(split_model.params) > 0:
                                    nb_alpha = float(split_model.params.iloc[-1])
                            except:
                                pass
                
                # Compute metrics
                metrics = compute_prediction_metrics(y_test_7mer, mu_pred_7mer, mu_pred_7mer, nb_alpha=nb_alpha)
                metrics['model_name'] = model_name
                all_metrics.append(metrics)
                print(f"  Successfully evaluated {model_name}: {len(y_test_7mer)} observations, R²={metrics['r_squared']:.4f}, Pearson r={metrics['pearson_r']:.4f}")
            else:
                # Prepare test features
                from load_saved_models import prepare_data_for_model
                X_test, _ = prepare_data_for_model(df_test, model_name, metadata)
                
                metrics = evaluate_model_on_test_set(
                    model, X_test, y_test, offset_test, model_name
                )
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_metrics:
        return None
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save if output directory provided
    if output_dir:
        print(f"DEBUG: output_dir is {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        metrics_df.to_csv(
            os.path.join(output_dir, 'prediction_accuracy_metrics.tsv'),
            sep='\t', index=False
        )
        
        # Create comparison plot (3-panel)
        print(f"DEBUG: Creating 3-panel plot...")
        plot_prediction_comparison(metrics_df, output_dir)
        
        # Always create combined 8-panel plot (with or without CV results)
        print(f"DEBUG: About to create 8-panel plot...")
        # Check for CV results in standard location if not provided
        if cv_results_path is None:
            # Try to find CV results in parent directory
            parent_dir = os.path.dirname(os.path.dirname(output_dir)) if os.path.dirname(output_dir) else output_dir
            potential_cv_path = os.path.join(parent_dir, 'cross_validation', 'cv_results_summary.json')
            if os.path.exists(potential_cv_path):
                cv_results_path = potential_cv_path
                print(f"  Found CV results at: {potential_cv_path}")
            else:
                # Try in same directory structure (output_dir's parent)
                potential_cv_path2 = os.path.join(os.path.dirname(output_dir), 'cross_validation', 'cv_results_summary.json')
                if os.path.exists(potential_cv_path2):
                    cv_results_path = potential_cv_path2
                    print(f"  Found CV results at: {potential_cv_path2}")
                else:
                    # Try in output_dir itself
                    potential_cv_path3 = os.path.join(output_dir, '..', 'cross_validation', 'cv_results_summary.json')
                    potential_cv_path3 = os.path.normpath(potential_cv_path3)
                    if os.path.exists(potential_cv_path3):
                        cv_results_path = potential_cv_path3
                        print(f"  Found CV results at: {potential_cv_path3}")
        
        print(f"\nCreating 8-panel combined model evaluation plot...")
        if cv_results_path and os.path.exists(cv_results_path):
            print(f"  Using CV results from: {cv_results_path}")
        else:
            print(f"  Note: CV results not found. Bottom row will show 'CV results not available'")
            print(f"    Searched in: {os.path.join(os.path.dirname(output_dir), 'cross_validation', 'cv_results_summary.json')}")
            cv_results_path = None
        
        # Save to final_figs directory if it exists, otherwise to output_dir
        final_figs_dir = os.path.join(os.path.dirname(output_dir), 'final_figs')
        if os.path.exists(final_figs_dir):
            plot_output_dir = final_figs_dir
        else:
            # Create final_figs directory
            plot_output_dir = final_figs_dir
            os.makedirs(plot_output_dir, exist_ok=True)
        
        print(f"  Calling plot_combined_model_evaluation with:")
        print(f"    metrics_df: {len(metrics_df)} rows")
        print(f"    cv_results_path: {cv_results_path}")
        print(f"    plot_output_dir: {plot_output_dir}")
        
        try:
            plot_combined_model_evaluation(metrics_df, cv_results_path, plot_output_dir)
            print(f"  ✓ Successfully generated 8-panel plot")
        except Exception as e:
            print(f"  ✗ ERROR: Failed to generate 8-panel plot: {e}")
            import traceback
            traceback.print_exc()
    
    return metrics_df


def plot_prediction_comparison(metrics_df: pd.DataFrame, output_dir: str):
    """
    Create publication-ready visualization comparing prediction accuracy across models.
    """
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as ticker
    
    # Set publication-quality style with larger fonts for document readability
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.labelsize': 16,
        'axes.titlesize': 17,
        'axes.linewidth': 1.2,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'grid.linestyle': '--',
    })
    
    fig = plt.figure(figsize=(15, 6))
    
    # Create grid: single row with 3 plots
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.35, 
                  left=0.06, right=0.98, top=0.95, bottom=0.25)
    
    model_names = metrics_df['model_name'].values
    
    # Professional color palette (consistent, colorblind-friendly)
    colors = {
        'mse': '#2E86AB',      # Blue
        'pearson': '#F18F01',  # Orange
        'r2': '#C73E1D',       # Red
        'deviance': '#6A994E', # Green
        'rmsle': '#BC4749'     # Dark red
    }
    
    # Helper function to set y-axis limits to show differences clearly
    def set_yaxis_to_show_differences(ax, values, min_pad=0.05, max_pad=0.1, start_at_zero=True, zoom_when_similar=True):
        """Set y-axis limits to make differences between values clearly visible.
        
        If values are very similar (range < 5% of mean), zoom in to show differences.
        """
        if len(values) == 0:
            return
        min_val = np.min(values)
        max_val = np.max(values)
        val_range = max_val - min_val
        mean_val = np.mean(values)
        
        # If values are very similar (range < 5% of mean), always zoom in
        if zoom_when_similar and val_range > 0 and mean_val > 0:
            range_pct = (val_range / mean_val) * 100
            if range_pct < 5:  # Less than 5% difference
                # Zoom in to show differences - don't start at zero
                start_at_zero = False
                # Add more padding to make differences visible
                min_pad = 0.2  # 20% padding below
                max_pad = 0.2  # 20% padding above
        
        if start_at_zero and min_val >= 0:
            y_min = 0
        else:
            # Add padding below minimum
            y_min = min_val - (val_range * min_pad) if val_range > 0 else min_val * (1 - min_pad)
            if start_at_zero and y_min < 0:
                y_min = 0
        
        # Add padding above maximum to show differences
        if val_range > 0:
            y_max = max_val + (val_range * max_pad)
        else:
            # If all values are the same, add small padding
            y_max = max_val * (1 + max_pad) if max_val > 0 else max_pad
        
        ax.set_ylim([y_min, y_max])
    
    # Helper function to format axes consistently
    def format_axes(ax, ylabel, title, use_scientific=False):
        """Format axes with consistent styling."""
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=17, fontweight='bold', pad=12)
        ax.tick_params(axis='x', rotation=45, labelsize=13, length=4, width=1.2)
        ax.tick_params(axis='y', labelsize=13, length=4, width=1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        ax.set_axisbelow(True)
        
        # Format y-axis with scientific notation if needed
        if use_scientific:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2e}'))
        else:
            # Use standard formatting with appropriate precision
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.4f}' if abs(x) < 1 else f'{x:.3f}'))
    
    # Plot 1: MSE (lower is better)
    ax = fig.add_subplot(gs[0, 0])
    mse_vals = metrics_df['mse'].values
    bars = ax.bar(model_names, mse_vals, color=colors['mse'], alpha=0.85, edgecolor='black', linewidth=0.8)
    format_axes(ax, 'Mean Squared Error', 'MSE', use_scientific=True)
    set_yaxis_to_show_differences(ax, mse_vals, start_at_zero=False, zoom_when_similar=True)
    
    # Plot 2: Pearson correlation (higher is better)
    ax = fig.add_subplot(gs[0, 1])
    pearson_vals = metrics_df['pearson_r'].values
    bars = ax.bar(model_names, pearson_vals, color=colors['pearson'], alpha=0.85, edgecolor='black', linewidth=0.8)
    format_axes(ax, 'Pearson Correlation', 'Correlation', use_scientific=False)
    set_yaxis_to_show_differences(ax, pearson_vals, start_at_zero=False, zoom_when_similar=True)
    # Set reasonable limits for correlation (typically 0-1, but allow negative)
    y_min_corr = min(pearson_vals) - 0.05 if min(pearson_vals) < 0 else max(0, min(pearson_vals) - 0.05)
    y_max_corr = min(1.05, max(pearson_vals) + 0.05) if max(pearson_vals) <= 1 else max(pearson_vals) + 0.05
    ax.set_ylim([y_min_corr, y_max_corr])
    
    # Plot 3: R² (higher is better)
    ax = fig.add_subplot(gs[0, 2])
    r2_vals = metrics_df['r_squared'].values
    bars = ax.bar(model_names, r2_vals, color=colors['r2'], alpha=0.85, edgecolor='black', linewidth=0.8)
    format_axes(ax, 'R²', 'R²', use_scientific=False)
    set_yaxis_to_show_differences(ax, r2_vals, start_at_zero=False, zoom_when_similar=True)
    # Set reasonable limits for R² (can be negative for poor models)
    y_min_r2 = min(r2_vals) - 0.05 if min(r2_vals) < 0 else max(0, min(r2_vals) - 0.05)
    y_max_r2 = min(1.05, max(r2_vals) + 0.05) if max(r2_vals) <= 1 else max(r2_vals) + 0.05
    ax.set_ylim([y_min_r2, y_max_r2])
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with high quality
    output_path = os.path.join(output_dir, 'prediction_accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved prediction comparison plot to {output_path}")


def plot_combined_model_evaluation(
    prediction_metrics_df: pd.DataFrame,
    cv_results_path: str,
    output_dir: str
):
    """
    Create publication-ready multiplot with:
    Top row: Top 4 prediction accuracy metrics (MSE, RMSE, Pearson, R²)
    Bottom row: Top 4 cross-validation metrics (MSE, MAE, Pearson, Poisson Deviance)
    
    Args:
        prediction_metrics_df: DataFrame with prediction accuracy metrics
        cv_results_path: Path to CV results JSON file
        output_dir: Output directory for plots
    """
    import json
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as ticker
    
    # Load CV results if available
    cv_results = None
    if cv_results_path and os.path.exists(cv_results_path):
        with open(cv_results_path, 'r') as f:
            cv_results = json.load(f)
    
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.linewidth': 1.2,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'grid.linestyle': '--',
    })
    
    fig = plt.figure(figsize=(20, 10))
    # Removed overall title
    
    # Create grid: 2 rows x 4 columns
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.4, 
                  left=0.05, right=0.98, top=0.95, bottom=0.12)
    
    model_names = prediction_metrics_df['model_name'].values
    
    # Professional color palette
    colors = {
        'mse': '#2E86AB',      # Blue
        'rmse': '#1B998B',     # Teal
        'mae': '#6A994E',      # Green
        'pearson': '#F18F01',  # Orange
        'r2': '#C73E1D',       # Red
        'deviance': '#7209B7'  # Purple
    }
    
    # Helper function to set y-axis limits
    def set_yaxis_to_show_differences(ax, values, min_pad=0.05, max_pad=0.1, start_at_zero=True, zoom_when_similar=True):
        if len(values) == 0:
            return
        min_val = np.min(values)
        max_val = np.max(values)
        val_range = max_val - min_val
        mean_val = np.mean(values)
        
        if zoom_when_similar and val_range > 0 and mean_val > 0:
            range_pct = (val_range / mean_val) * 100
            if range_pct < 5:
                start_at_zero = False
                min_pad = 0.2
                max_pad = 0.2
        
        if start_at_zero and min_val >= 0:
            y_min = 0
        else:
            y_min = min_val - (val_range * min_pad) if val_range > 0 else min_val * (1 - min_pad)
            if start_at_zero and y_min < 0:
                y_min = 0
        
        if val_range > 0:
            y_max = max_val + (val_range * max_pad)
        else:
            y_max = max_val * (1 + max_pad) if max_val > 0 else max_pad
        
        ax.set_ylim([y_min, y_max])
    
    # Helper function to format axes consistently
    def format_axes(ax, ylabel, title, use_scientific=False):
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.tick_params(axis='x', rotation=45, labelsize=9, length=4, width=1.2)
        ax.tick_params(axis='y', labelsize=9, length=4, width=1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        ax.set_axisbelow(True)
        
        if use_scientific:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2e}'))
        else:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.4f}' if abs(x) < 1 else f'{x:.3f}'))
    
    # TOP ROW: Top 4 Prediction Accuracy Metrics
    # Row 1, Column 1: MSE (lower is better)
    ax = fig.add_subplot(gs[0, 0])
    mse_vals = prediction_metrics_df['mse'].values
    bars = ax.bar(model_names, mse_vals, color=colors['mse'], alpha=0.85, edgecolor='black', linewidth=0.8)
    format_axes(ax, 'Mean Squared Error', 'MSE', use_scientific=True)
    set_yaxis_to_show_differences(ax, mse_vals, start_at_zero=False, zoom_when_similar=True)
    
    # Row 1, Column 2: RMSE (lower is better)
    ax = fig.add_subplot(gs[0, 1])
    if 'rmse' in prediction_metrics_df.columns:
        rmse_vals = prediction_metrics_df['rmse'].values
        bars = ax.bar(model_names, rmse_vals, color=colors['rmse'], alpha=0.85, edgecolor='black', linewidth=0.8)
        format_axes(ax, 'Root Mean Squared Error', 'RMSE', use_scientific=False)
        set_yaxis_to_show_differences(ax, rmse_vals, start_at_zero=False, zoom_when_similar=True)
    
    # Row 1, Column 3: Pearson correlation (higher is better)
    ax = fig.add_subplot(gs[0, 2])
    pearson_vals = prediction_metrics_df['pearson_r'].values
    bars = ax.bar(model_names, pearson_vals, color=colors['pearson'], alpha=0.85, edgecolor='black', linewidth=0.8)
    format_axes(ax, 'Pearson Correlation', 'Pearson r', use_scientific=False)
    set_yaxis_to_show_differences(ax, pearson_vals, start_at_zero=False, zoom_when_similar=True)
    y_min_corr = min(pearson_vals) - 0.05 if min(pearson_vals) < 0 else max(0, min(pearson_vals) - 0.05)
    y_max_corr = min(1.05, max(pearson_vals) + 0.05) if max(pearson_vals) <= 1 else max(pearson_vals) + 0.05
    ax.set_ylim([y_min_corr, y_max_corr])
    
    # Row 1, Column 4: R² (higher is better)
    ax = fig.add_subplot(gs[0, 3])
    r2_vals = prediction_metrics_df['r_squared'].values
    bars = ax.bar(model_names, r2_vals, color=colors['r2'], alpha=0.85, edgecolor='black', linewidth=0.8)
    format_axes(ax, 'R²', 'R²', use_scientific=False)
    set_yaxis_to_show_differences(ax, r2_vals, start_at_zero=False, zoom_when_similar=True)
    y_min_r2 = min(r2_vals) - 0.05 if min(r2_vals) < 0 else max(0, min(r2_vals) - 0.05)
    y_max_r2 = min(1.05, max(r2_vals) + 0.05) if max(r2_vals) <= 1 else max(r2_vals) + 0.05
    ax.set_ylim([y_min_r2, y_max_r2])
    
    # BOTTOM ROW: Top 4 Cross-Validation Metrics
    if cv_results:
        # Row 2, Column 1: CV MSE (lower is better)
        ax = fig.add_subplot(gs[1, 0])
        cv_mse_vals = [cv_results.get(model, {}).get('mse_mean', np.nan) for model in model_names]
        if not all(np.isnan(cv_mse_vals)):
            bars = ax.bar(model_names, cv_mse_vals, color=colors['mse'], alpha=0.85, edgecolor='black', linewidth=0.8)
            format_axes(ax, 'CV MSE (mean)', 'CV MSE', use_scientific=True)
            set_yaxis_to_show_differences(ax, [v for v in cv_mse_vals if not np.isnan(v)], start_at_zero=False, zoom_when_similar=True)
        
        # Row 2, Column 2: CV MAE (lower is better)
        ax = fig.add_subplot(gs[1, 1])
        cv_mae_vals = [cv_results.get(model, {}).get('mae_mean', np.nan) for model in model_names]
        if not all(np.isnan(cv_mae_vals)):
            bars = ax.bar(model_names, cv_mae_vals, color=colors['mae'], alpha=0.85, edgecolor='black', linewidth=0.8)
            format_axes(ax, 'CV MAE (mean)', 'CV MAE', use_scientific=False)
            set_yaxis_to_show_differences(ax, [v for v in cv_mae_vals if not np.isnan(v)], start_at_zero=False, zoom_when_similar=True)
        
        # Row 2, Column 3: CV Pearson correlation (higher is better)
        ax = fig.add_subplot(gs[1, 2])
        cv_pearson_vals = [cv_results.get(model, {}).get('pearson_r_mean', np.nan) for model in model_names]
        if not all(np.isnan(cv_pearson_vals)):
            bars = ax.bar(model_names, cv_pearson_vals, color=colors['pearson'], alpha=0.85, edgecolor='black', linewidth=0.8)
            format_axes(ax, 'CV Pearson r (mean)', 'CV Pearson r', use_scientific=False)
            valid_vals = [v for v in cv_pearson_vals if not np.isnan(v)]
            if valid_vals:
                set_yaxis_to_show_differences(ax, valid_vals, start_at_zero=False, zoom_when_similar=True)
                y_min = min(valid_vals) - 0.05 if min(valid_vals) < 0 else max(0, min(valid_vals) - 0.05)
                y_max = min(1.05, max(valid_vals) + 0.05) if max(valid_vals) <= 1 else max(valid_vals) + 0.05
                ax.set_ylim([y_min, y_max])
        
        # Row 2, Column 4: CV Deviance (NB if models were NB, otherwise Poisson)
        ax = fig.add_subplot(gs[1, 3])
        cv_dev_vals = [cv_results.get(model, {}).get('poisson_dev_mean', np.nan) for model in model_names]
        if not all(np.isnan(cv_dev_vals)):
            bars = ax.bar(model_names, cv_dev_vals, color=colors['deviance'], alpha=0.85, edgecolor='black', linewidth=0.8)
            format_axes(ax, 'CV Deviance (mean)', 'CV Deviance', use_scientific=True)
            set_yaxis_to_show_differences(ax, [v for v in cv_dev_vals if not np.isnan(v)], start_at_zero=False, zoom_when_similar=True)
    else:
        # If no CV results, show message
        for i in range(4):
            ax = fig.add_subplot(gs[1, i])
            ax.text(0.5, 0.5, 'CV results not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with high quality
    output_path = os.path.join(output_dir, 'combined_model_evaluation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved combined model evaluation plot to {output_path}")


def print_metrics_summary(metrics_df: pd.DataFrame):
    """
    Print formatted summary of prediction metrics.
    """
    print("\n" + "=" * 80)
    print("PREDICTION ACCURACY METRICS SUMMARY")
    print("=" * 80)
    
    for _, row in metrics_df.iterrows():
        print(f"\n{row['model_name']}:")
        print(f"  MSE: {row['mse']:.4f}")
        print(f"  MAE: {row['mae']:.4f}")
        print(f"  RMSE: {row['rmse']:.4f}")
        print(f"  Pearson r: {row['pearson_r']:.4f} (p={row['pearson_p']:.2e})")
        print(f"  R²: {row['r_squared']:.4f}")
        deviance_key = 'deviance' if 'deviance' in row else 'poisson_deviance'
        print(f"  Deviance: {row[deviance_key]:.2f}")
        print(f"  RMSLE: {row['rmsle']:.4f}")


def regenerate_plots_from_tsv(tsv_path: str, output_dir: str = None, cv_results_path: str = None):
    """
    Regenerate prediction accuracy plots from existing TSV file.
    
    Args:
        tsv_path: Path to prediction_accuracy_metrics.tsv file
        output_dir: Output directory for plots (default: same directory as TSV)
        cv_results_path: Optional path to CV results JSON to create combined plot
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    if output_dir is None:
        output_dir = os.path.dirname(tsv_path)
    
    print(f"Loading metrics from {tsv_path}...")
    metrics_df = pd.read_csv(tsv_path, sep='\t')
    
    print(f"Found {len(metrics_df)} models:")
    for model_name in metrics_df['model_name'].values:
        print(f"  - {model_name}")
    
    print(f"\nRegenerating plots with improved y-axis scaling...")
    plot_prediction_comparison(metrics_df, output_dir)
    
    print(f"\nPlots saved to {output_dir}/prediction_accuracy_comparison.png")
    
    # Always create combined 8-panel plot (with or without CV results)
    # Check for CV results in standard location if not provided
    if cv_results_path is None:
        # Try to find CV results in parent directory
        parent_dir = os.path.dirname(output_dir)
        potential_cv_path = os.path.join(parent_dir, 'cross_validation', 'cv_results_summary.json')
        if os.path.exists(potential_cv_path):
            cv_results_path = potential_cv_path
            print(f"  Found CV results at: {potential_cv_path}")
        else:
            # Try in grandparent directory
            grandparent_dir = os.path.dirname(parent_dir)
            potential_cv_path2 = os.path.join(grandparent_dir, 'cross_validation', 'cv_results_summary.json')
            if os.path.exists(potential_cv_path2):
                cv_results_path = potential_cv_path2
                print(f"  Found CV results at: {potential_cv_path2}")
    
    print(f"\nCreating 8-panel combined model evaluation plot...")
    if cv_results_path and os.path.exists(cv_results_path):
        print(f"  Using CV results from: {cv_results_path}")
    else:
        print(f"  Note: CV results not found. Bottom row will show 'CV results not available'")
        cv_results_path = None
    
    # Save to final_figs directory if it exists, otherwise create it
    final_figs_dir = os.path.join(os.path.dirname(output_dir), 'final_figs')
    if not os.path.exists(final_figs_dir):
        os.makedirs(final_figs_dir, exist_ok=True)
    plot_output_dir = final_figs_dir
    
    try:
        plot_combined_model_evaluation(metrics_df, cv_results_path, plot_output_dir)
        print(f"  ✓ Successfully generated 8-panel plot at {plot_output_dir}/combined_model_evaluation.png")
    except Exception as e:
        print(f"  ✗ ERROR: Failed to generate 8-panel plot: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nSummary of metrics:")
    print_metrics_summary(metrics_df)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate prediction accuracy or regenerate plots')
    parser.add_argument('--output-dir',
                       help='Output directory from main modeling script (contains fitted_models/)')
    parser.add_argument('--regenerate-plots', type=str, metavar='TSV_PATH',
                       help='Regenerate plots from existing TSV file (provide path to prediction_accuracy_metrics.tsv)')
    parser.add_argument('--cv-results', type=str, metavar='JSON_PATH',
                       help='Path to CV results JSON file (for combined plot generation)')
    parser.add_argument('--counts-dir',
                       help='Directory with .counts files (for loading test data)')
    parser.add_argument('--genome-fasta',
                       help='Reference genome FASTA')
    parser.add_argument('--exclusion-mask', default=None,
                       help='Optional exclusion mask TSV')
    parser.add_argument('--test-fraction', type=float, default=0.2,
                       help='Fraction of data to use as test set')
    parser.add_argument('--use-saved-models', action='store_true', default=True,
                       help='Use saved models from output-dir (default: True)')
    
    args = parser.parse_args()
    
    # If regenerating plots, do that and exit
    if args.regenerate_plots:
        regenerate_plots_from_tsv(args.regenerate_plots, args.output_dir, args.cv_results)
        print("\nDone!")
        exit(0)
    
    # Otherwise, run full evaluation
    if not args.output_dir or not args.counts_dir or not args.genome_fasta:
        parser.error("--output-dir, --counts-dir, and --genome-fasta are required (unless using --regenerate-plots)")
    
    # Import here to avoid dependency when just regenerating plots
    from sequence_bias_modeling_sitelevel import load_site_level_data
    from load_saved_models import load_models_and_prepare_data
    from sklearn.model_selection import train_test_split
    
    print("Loading saved models...")
    result = load_models_and_prepare_data(args.output_dir)
    models = result['models']
    metadata = result['metadata']
    
    if not models:
        print("ERROR: No saved models found. Run main modeling script first.")
        exit(1)
    
    print(f"Loaded {len(models)} models")
    
    print("\nLoading data...")
    df = load_site_level_data(args.counts_dir, args.genome_fasta, args.exclusion_mask)
    print(f"Loaded {len(df):,} sites")
    
    # Split into train/test (or use all as test if models already fit)
    if args.use_saved_models:
        # Use all data as test set if models already fit
        df_test = df.copy()
        print(f"Using all {len(df_test):,} sites as test set (models already fit)")
    else:
        # Split into train/test
        print(f"Splitting into train/test ({1-args.test_fraction:.0%}/{args.test_fraction:.0%})...")
        _, df_test = train_test_split(
            df, test_size=args.test_fraction, random_state=42
        )
        print(f"Test set: {len(df_test):,} sites")
    
    # Check for CV results
    cv_results_path = None
    if args.output_dir:
        potential_cv_path = os.path.join(args.output_dir, 'cross_validation', 'cv_results_summary.json')
        if os.path.exists(potential_cv_path):
            cv_results_path = potential_cv_path
    
    # Evaluate each model on test set
    print("\nEvaluating models on test set...")
    test_metrics = compare_models_predictions(
        models, df_test, args.output_dir, metadata, cv_results_path=cv_results_path
    )
    
    if test_metrics is not None:
        print_metrics_summary(test_metrics)
    
    print("\nDone!")
    print("\nTo regenerate plots with improved scaling, run:")
    print(f"  python {__file__} --regenerate-plots {args.output_dir}/prediction_evaluation/prediction_accuracy_metrics.tsv --output-dir {args.output_dir}/prediction_evaluation")

