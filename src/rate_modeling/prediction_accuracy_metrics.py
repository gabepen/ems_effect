#!/usr/bin/env python3
"""
Prediction accuracy metrics for GLM models.

Computes various metrics on held-out test data:
- MSE, MAE, RMSE
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
    mu_predicted: np.ndarray = None  # Mean predictions for GLM
) -> Dict:
    """
    Compute comprehensive prediction accuracy metrics.
    
    Args:
        y_observed: Actual observed counts
        y_predicted: Predicted counts (can be mean or integer predictions)
        mu_predicted: Mean predictions (if different from y_predicted)
    
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
    
    # Poisson-specific metrics
    # Deviance (negative log-likelihood)
    mu_safe = np.maximum(mu_predicted, 1e-10)
    y_safe = np.maximum(y_observed, 0)
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
        
        # Poisson-specific
        'poisson_deviance': poisson_deviance,
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
    
    # Compute all metrics
    metrics = compute_prediction_metrics(y_test, mu_pred, mu_pred)
    metrics['model_name'] = model_name
    
    return metrics


def compare_models_predictions(
    models: Dict,
    df_test: pd.DataFrame,
    output_dir: str = None,
    metadata: Dict = None
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
        
        try:
            # Special handling for 7mer_split which uses split models
            if model_name == '7mer_split' and isinstance(model, list):
                # Use split model prediction
                mu_pred = predict_with_7mer_splits(model, df_test)
                # Filter to rows where we have 7mer data
                df7_test = df_test[df_test['kmer7'].notna()].copy()
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
                
                # Compute metrics
                metrics = compute_prediction_metrics(y_test_7mer, mu_pred_7mer, mu_pred_7mer)
                metrics['model_name'] = model_name
                all_metrics.append(metrics)
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
        os.makedirs(output_dir, exist_ok=True)
        metrics_df.to_csv(
            os.path.join(output_dir, 'prediction_accuracy_metrics.tsv'),
            sep='\t', index=False
        )
        
        # Create comparison plot
        plot_prediction_comparison(metrics_df, output_dir)
    
    return metrics_df


def plot_prediction_comparison(metrics_df: pd.DataFrame, output_dir: str):
    """
    Create visualization comparing prediction accuracy across models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Prediction Accuracy Comparison Across Models', fontsize=16, fontweight='bold')
    
    model_names = metrics_df['model_name'].values
    
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
    
    # Plot 1: MSE (lower is better)
    ax = axes[0, 0]
    mse_vals = metrics_df['mse'].values
    ax.bar(model_names, mse_vals, color='#4C78A8')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('MSE (lower is better)')
    ax.tick_params(axis='x', rotation=45)
    set_yaxis_to_show_differences(ax, mse_vals, start_at_zero=False, zoom_when_similar=True)
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: MAE (lower is better)
    ax = axes[0, 1]
    mae_vals = metrics_df['mae'].values
    ax.bar(model_names, mae_vals, color='#F58518')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('MAE (lower is better)')
    ax.tick_params(axis='x', rotation=45)
    set_yaxis_to_show_differences(ax, mae_vals, start_at_zero=False, zoom_when_similar=True)
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 3: Pearson correlation (higher is better)
    ax = axes[0, 2]
    pearson_vals = metrics_df['pearson_r'].values
    ax.bar(model_names, pearson_vals, color='#54A24B')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Correlation (higher is better)')
    ax.tick_params(axis='x', rotation=45)
    # For correlation, zoom in to show differences if values are similar
    set_yaxis_to_show_differences(ax, pearson_vals, start_at_zero=False, zoom_when_similar=True)
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: R² (higher is better)
    ax = axes[1, 0]
    r2_vals = metrics_df['r_squared'].values
    ax.bar(model_names, r2_vals, color='#E45756')
    ax.set_ylabel('R²')
    ax.set_title('R² (higher is better)')
    ax.tick_params(axis='x', rotation=45)
    # For R², zoom in to show differences
    set_yaxis_to_show_differences(ax, r2_vals, start_at_zero=False, zoom_when_similar=True)
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 5: Poisson Deviance (lower is better)
    ax = axes[1, 1]
    deviance_vals = metrics_df['poisson_deviance'].values
    ax.bar(model_names, deviance_vals, color='#72B7B2')
    ax.set_ylabel('Poisson Deviance')
    ax.set_title('Deviance (lower is better)')
    ax.tick_params(axis='x', rotation=45)
    # For large values, zoom in to show differences
    set_yaxis_to_show_differences(ax, deviance_vals, start_at_zero=False, zoom_when_similar=True)
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 6: RMSLE (lower is better)
    ax = axes[1, 2]
    rmsle_vals = metrics_df['rmsle'].values
    ax.bar(model_names, rmsle_vals, color='#B279A2')
    ax.set_ylabel('Root Mean Squared Log Error')
    ax.set_title('RMSLE (lower is better)')
    ax.tick_params(axis='x', rotation=45)
    set_yaxis_to_show_differences(ax, rmsle_vals, start_at_zero=False, zoom_when_similar=True)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'prediction_accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved prediction comparison plot to {output_path}")


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
        print(f"  Poisson Deviance: {row['poisson_deviance']:.2f}")
        print(f"  RMSLE: {row['rmsle']:.4f}")


def regenerate_plots_from_tsv(tsv_path: str, output_dir: str = None):
    """
    Regenerate prediction accuracy plots from existing TSV file.
    
    Args:
        tsv_path: Path to prediction_accuracy_metrics.tsv file
        output_dir: Output directory for plots (default: same directory as TSV)
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
    print("\nSummary of metrics:")
    print_metrics_summary(metrics_df)


if __name__ == '__main__':
    import argparse
    from sequence_bias_modeling_sitelevel import load_site_level_data
    from load_saved_models import load_models_and_prepare_data
    from sklearn.model_selection import train_test_split
    
    parser = argparse.ArgumentParser(description='Evaluate prediction accuracy or regenerate plots')
    parser.add_argument('--output-dir',
                       help='Output directory from main modeling script (contains fitted_models/)')
    parser.add_argument('--regenerate-plots', type=str, metavar='TSV_PATH',
                       help='Regenerate plots from existing TSV file (provide path to prediction_accuracy_metrics.tsv)')
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
        regenerate_plots_from_tsv(args.regenerate_plots, args.output_dir)
        print("\nDone!")
        exit(0)
    
    # Otherwise, run full evaluation
    if not args.output_dir or not args.counts_dir or not args.genome_fasta:
        parser.error("--output-dir, --counts-dir, and --genome-fasta are required (unless using --regenerate-plots)")
    
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
    
    # Evaluate each model on test set
    print("\nEvaluating models on test set...")
    test_metrics = compare_models_predictions(
        models, df_test, args.output_dir, metadata
    )
    
    if test_metrics is not None:
        print_metrics_summary(test_metrics)
    
    print("\nDone!")
    print("\nTo regenerate plots with improved scaling, run:")
    print(f"  python {__file__} --regenerate-plots {args.output_dir}/prediction_evaluation/prediction_accuracy_metrics.tsv --output-dir {args.output_dir}/prediction_evaluation")

