#!/usr/bin/env python3
"""
Residual analysis for GLM models.

Analyzes prediction errors (observed - predicted) to diagnose model issues:
- Outliers
- Overdispersion (for Poisson models)
- Non-linearity
- Heteroscedasticity
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict
import statsmodels.api as sm
import os


def compute_residuals(model, X: pd.DataFrame, y: np.ndarray, offset: np.ndarray) -> Dict:
    """
    Compute various types of residuals for GLM models.
    
    Returns dictionary with different residual types.
    """
    # Get predictions
    design = sm.add_constant(X, has_constant='add')
    mu = model.predict(design, offset=offset)
    mu = np.maximum(mu, 1e-10)  # Avoid zeros
    
    # Raw residuals: observed - predicted
    raw_residuals = y - mu
    
    # Pearson residuals: (observed - predicted) / sqrt(variance)
    # For Poisson: variance = mu
    pearson_residuals = raw_residuals / np.sqrt(mu)
    
    # Deviance residuals: signed sqrt of deviance contribution
    # For Poisson: sign(y - mu) * sqrt(2 * (y * log(y/mu) - (y - mu)))
    # Handle y=0 case
    y_safe = np.maximum(y, 1e-10)
    deviance_contrib = 2 * (y_safe * np.log(y_safe / mu) - (y_safe - mu))
    deviance_residuals = np.sign(raw_residuals) * np.sqrt(deviance_contrib)
    # Set to 0 when y=0 and mu is small
    deviance_residuals[(y == 0) & (mu < 0.01)] = 0
    
    # Standardized residuals (z-scores)
    # For large samples, pearson residuals ~ N(0,1) under correct model
    standardized = pearson_residuals
    
    return {
        'raw': raw_residuals,
        'pearson': pearson_residuals,
        'deviance': deviance_residuals,
        'standardized': standardized,
        'predicted': mu,
        'observed': y
    }


def analyze_residuals(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    offset: np.ndarray,
    model_name: str = "model"
) -> Dict:
    """
    Perform comprehensive residual analysis.
    
    Returns dictionary with diagnostic statistics.
    """
    residuals_dict = compute_residuals(model, X, y, offset)
    
    pearson = residuals_dict['pearson']
    deviance = residuals_dict['deviance']
    mu = residuals_dict['predicted']
    y_obs = residuals_dict['observed']
    
    # Summary statistics
    diagnostics = {
        'n_obs': len(y),
        'mean_pearson': np.mean(pearson),
        'std_pearson': np.std(pearson),
        'mean_deviance': np.mean(deviance),
        'std_deviance': np.std(deviance),
    }
    
    # Overdispersion test (for Poisson)
    # If Poisson is correct, Var(y) = E(y) = mu
    # Overdispersion means Var(y) > mu
    # Test: sum of squared Pearson residuals should be ~ chi-squared with df = n - p
    sum_sq_pearson = np.sum(pearson ** 2)
    n_params = len(model.params) if hasattr(model, 'params') else 0
    df_resid = len(y) - n_params
    overdispersion_ratio = sum_sq_pearson / df_resid if df_resid > 0 else np.nan
    
    diagnostics['overdispersion_ratio'] = overdispersion_ratio
    diagnostics['sum_sq_pearson'] = sum_sq_pearson
    diagnostics['df_residual'] = df_resid
    
    # Overdispersion is indicated if ratio >> 1
    diagnostics['overdispersed'] = overdispersion_ratio > 1.5
    
    # Outlier detection (Pearson residuals > 3 or < -3)
    n_outliers = np.sum(np.abs(pearson) > 3)
    diagnostics['n_outliers'] = n_outliers
    diagnostics['pct_outliers'] = 100 * n_outliers / len(y) if len(y) > 0 else 0
    
    # Normality test (residuals should be approximately normal for Poisson)
    if len(pearson) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(pearson[:5000])  # Limit to 5000 for speed
        diagnostics['shapiro_stat'] = shapiro_stat
        diagnostics['shapiro_p'] = shapiro_p
        diagnostics['normal_residuals'] = shapiro_p > 0.05
    else:
        diagnostics['normal_residuals'] = None
    
    # Zero-inflation check
    n_zeros = np.sum(y == 0)
    expected_zeros = np.sum(np.exp(-mu))  # P(Y=0) for Poisson
    zero_inflation_ratio = n_zeros / expected_zeros if expected_zeros > 0 else np.nan
    diagnostics['n_zeros'] = n_zeros
    diagnostics['expected_zeros'] = expected_zeros
    diagnostics['zero_inflation_ratio'] = zero_inflation_ratio
    diagnostics['zero_inflated'] = zero_inflation_ratio > 1.5
    
    return {
        'diagnostics': diagnostics,
        'residuals': residuals_dict
    }


def plot_residual_diagnostics(
    residual_analysis: Dict,
    model_name: str,
    output_dir: str
):
    """
    Create comprehensive residual diagnostic plots.
    """
    residuals_dict = residual_analysis['residuals']
    diagnostics = residual_analysis['diagnostics']
    
    pearson = residuals_dict['pearson']
    deviance = residuals_dict['deviance']
    mu = residuals_dict['predicted']
    y_obs = residuals_dict['observed']
    
    # Subsample for plotting if too large
    max_plot = 10000
    if len(pearson) > max_plot:
        idx = np.random.choice(len(pearson), max_plot, replace=False)
        pearson_plot = pearson[idx]
        deviance_plot = deviance[idx]
        mu_plot = mu[idx]
        y_obs_plot = y_obs[idx]
    else:
        pearson_plot = pearson
        deviance_plot = deviance
        mu_plot = mu
        y_obs_plot = y_obs
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Residual Diagnostics: {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Residuals vs Predicted (check for patterns)
    ax = axes[0, 0]
    ax.scatter(mu_plot, pearson_plot, alpha=0.3, s=5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=2, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=-2, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted count (μ)')
    ax.set_ylabel('Pearson residuals')
    ax.set_title('Residuals vs Predicted (should be random)')
    ax.grid(alpha=0.3)
    
    # Plot 2: Q-Q plot (check normality)
    ax = axes[0, 1]
    stats.probplot(pearson_plot, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (should follow diagonal)')
    ax.grid(alpha=0.3)
    
    # Plot 3: Histogram of residuals
    ax = axes[0, 2]
    ax.hist(pearson_plot, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Pearson residuals')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual Distribution (std={diagnostics["std_pearson"]:.2f})')
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Observed vs Predicted
    ax = axes[1, 0]
    # Log scale for better visualization
    max_val = max(mu_plot.max(), y_obs_plot.max())
    ax.scatter(mu_plot, y_obs_plot, alpha=0.3, s=5)
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel('Predicted count')
    ax.set_ylabel('Observed count')
    ax.set_title('Observed vs Predicted')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 5: Deviance residuals
    ax = axes[1, 1]
    ax.scatter(mu_plot, deviance_plot, alpha=0.3, s=5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted count (μ)')
    ax.set_ylabel('Deviance residuals')
    ax.set_title('Deviance Residuals vs Predicted')
    ax.grid(alpha=0.3)
    
    # Plot 6: Residuals summary stats
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    Diagnostic Summary:
    
    Overdispersion ratio: {diagnostics['overdispersion_ratio']:.2f}
      (>1.5 indicates overdispersion)
    
    Outliers (|residual|>3): {diagnostics['n_outliers']} ({diagnostics['pct_outliers']:.1f}%)
    
    Zero inflation ratio: {diagnostics['zero_inflation_ratio']:.2f}
      (>1.5 indicates excess zeros)
    
    Mean Pearson residual: {diagnostics['mean_pearson']:.3f}
    Std Pearson residual: {diagnostics['std_pearson']:.3f}
      (Should be ~0 and ~1 if model is correct)
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'residual_diagnostics_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved residual diagnostics to {output_path}")


def analyze_all_models(
    models: Dict,
    df: pd.DataFrame,
    output_dir: str,
    glm_family: str = 'poisson',
    metadata: Dict = None
):
    """
    Perform residual analysis for all fitted models.
    
    Args:
        models: Dictionary of {model_name: fitted_model}
        df: DataFrame with site-level data
        output_dir: Output directory for plots
        glm_family: GLM family (for reference)
        metadata: Optional metadata dict with feature column names
    """
    df = df[df['depth'] > 0].copy()
    df['log_depth'] = np.log(df['depth'])
    
    results = {}
    
    for model_name, model in models.items():
        if model is None:
            continue
        
        print(f"Analyzing residuals for {model_name}...")
        
        # Get appropriate features
        from load_saved_models import prepare_data_for_model
        try:
            X, feature_cols = prepare_data_for_model(df, model_name, metadata)
        except Exception as e:
            print(f"  Warning: Could not prepare features for {model_name}: {e}")
            continue
        
        y = df['ems_count'].values
        offset = df['log_depth'].values
        
        # Analyze residuals
        residual_analysis = analyze_residuals(model, X, y, offset, model_name)
        results[model_name] = residual_analysis
        
        # Plot diagnostics
        plot_residual_diagnostics(residual_analysis, model_name, output_dir)
        
        # Print summary
        diag = residual_analysis['diagnostics']
        print(f"  Overdispersion ratio: {diag['overdispersion_ratio']:.2f}")
        print(f"  Outliers: {diag['n_outliers']} ({diag['pct_outliers']:.1f}%)")
        print(f"  Zero inflation ratio: {diag['zero_inflation_ratio']:.2f}")
    
    return results

