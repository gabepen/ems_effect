#!/usr/bin/env python3
"""
Cross-validation evaluation for sequence bias models.

Splits data into k folds, trains on k-1 folds, tests on held-out fold.
Provides out-of-sample predictive performance metrics.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List, Tuple
import statsmodels.api as sm
from sequence_bias_modeling_sitelevel import fit_glm, add_positional_features, add_3mer_features, add_pos3mer_features, add_5mer_features
import gc


def cross_validate_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    offset_col: str,
    glm_family: str = 'poisson',
    nb_alpha: float = None,
    n_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Perform k-fold cross-validation on a GLM model.
    
    Returns:
        Dictionary with CV metrics (mean, std across folds)
    """
    # Prepare data
    X = df[['treatment'] + feature_cols].values
    y = df[y_col].values
    offset = df[offset_col].values
    
    # Create folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        offset_train, offset_test = offset[train_idx], offset[test_idx]
        
        # Fit model on training fold
        try:
            model = fit_glm(y_train, pd.DataFrame(X_train, columns=['treatment'] + feature_cols),
                           offset_train, family=glm_family, nb_alpha=nb_alpha)
            
            # Predict on test fold
            X_test_df = pd.DataFrame(X_test, columns=['treatment'] + feature_cols)
            design_test = sm.add_constant(X_test_df, has_constant='add')
            
            # Get predictions (expected counts)
            mu_test = model.predict(design_test, offset=offset_test)
            
            # Calculate metrics
            mse = np.mean((y_test - mu_test) ** 2)
            mae = np.mean(np.abs(y_test - mu_test))
            
            # Pearson correlation
            if len(y_test) > 1 and np.std(mu_test) > 0:
                pearson_r = np.corrcoef(y_test, mu_test)[0, 1]
            else:
                pearson_r = np.nan
            
            # Extract nb_alpha from fitted model if it's a Negative Binomial model
            fold_nb_alpha = nb_alpha  # Use provided alpha if available
            if fold_nb_alpha is None and glm_family == 'negative_binomial':
                # Try to extract alpha from the fitted model
                # Check if it's a GLM NB model (has family.alpha)
                if hasattr(model, 'family') and hasattr(model.family, 'alpha'):
                    fold_nb_alpha = model.family.alpha
                # Check if it's an NBDiscrete model (statsmodels.discrete.discrete_model.NegativeBinomial)
                elif 'NegativeBinomial' in str(type(model)) or 'NBDiscrete' in str(type(model)):
                    try:
                        # For NBDiscrete models, alpha is the last parameter
                        if hasattr(model, 'params') and len(model.params) > 0:
                            fold_nb_alpha = float(model.params.iloc[-1])
                    except:
                        pass
            
            # Deviance calculation: NB if alpha available, otherwise Poisson
            mu_safe = np.maximum(mu_test, 1e-10)
            y_safe = np.maximum(y_test, 0)
            
            if fold_nb_alpha is not None and fold_nb_alpha > 0:
                # Negative Binomial deviance
                # Formula: 2 * Σ [y * log(y/μ) - (y + 1/α) * log((y + 1/α)/(μ + 1/α))]
                inv_alpha = 1.0 / fold_nb_alpha
                y_plus_inv_alpha = y_safe + inv_alpha
                mu_plus_inv_alpha = mu_safe + inv_alpha
                
                y_log_term = np.where(y_safe > 0, y_safe * np.log((y_safe + 1e-10) / mu_safe), 0)
                combined_log_term = y_plus_inv_alpha * np.log((y_plus_inv_alpha + 1e-10) / (mu_plus_inv_alpha + 1e-10))
                
                deviance = 2 * np.sum(y_log_term - combined_log_term)
            else:
                # Poisson deviance (fallback)
                deviance = 2 * np.sum(
                    y_safe * np.log((y_safe + 1e-10) / mu_safe) - (y_safe - mu_safe)
                )
            
            fold_metrics.append({
                'fold': fold_idx + 1,
                'mse': mse,
                'mae': mae,
                'pearson_r': pearson_r,
                'poisson_deviance': deviance,  # Keep key name for compatibility, but value is NB if alpha provided
                'n_test': len(y_test)
            })
            
            # Clean up memory
            del model, X_train, X_test, y_train, y_test, offset_train, offset_test
            del design_test, mu_test
            gc.collect()
        except Exception as e:
            print(f"Warning: Fold {fold_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not fold_metrics:
        return None
    
    # Aggregate across folds
    metrics_df = pd.DataFrame(fold_metrics)
    
    return {
        'mse_mean': metrics_df['mse'].mean(),
        'mse_std': metrics_df['mse'].std(),
        'mae_mean': metrics_df['mae'].mean(),
        'mae_std': metrics_df['mae'].std(),
        'pearson_r_mean': metrics_df['pearson_r'].mean(),
        'pearson_r_std': metrics_df['pearson_r'].std(),
        'poisson_dev_mean': metrics_df['poisson_deviance'].mean(),
        'poisson_dev_std': metrics_df['poisson_deviance'].std(),
        'n_folds': len(fold_metrics),
        'fold_details': fold_metrics
    }


def cross_validate_7mer_splits(
    df: pd.DataFrame,
    glm_family: str = 'poisson',
    nb_alpha: float = None,
    n_folds: int = 5,
    n_splits: int = 8,
    random_state: int = 42
) -> Dict:
    """
    Cross-validate 7mer model using split-fitting approach to control memory.
    
    Returns dictionary with CV metrics.
    """
    if 'kmer7' not in df.columns:
        return None
    df7 = df[df['kmer7'].notna()].copy()
    if df7.empty:
        return None
    if 'log_depth' not in df7.columns:
        df7['log_depth'] = np.log(df7['depth'])
    
    from sklearn.model_selection import KFold
    from sequence_bias_modeling_sitelevel import _split_list_round_robin
    
    unique7 = sorted(df7['kmer7'].dropna().unique())
    if not unique7:
        return None
    
    groups = _split_list_round_robin(unique7, n_splits)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df7)):
        df_train = df7.iloc[train_idx].copy()
        df_test = df7.iloc[test_idx].copy()
        
        try:
            total_llf = 0.0
            total_params = 0
            total_n = 0
            test_predictions = []
            test_actuals = []
            fitted_models = []  # Track fitted models to extract nb_alpha
            
            # Fit model on training fold using splits
            for group in groups:
                if not group:
                    continue
                sub_train = df_train[df_train['kmer7'].isin(group)].copy()
                if sub_train.empty:
                    continue
                
                dummies_train = pd.get_dummies(sub_train['kmer7'], prefix='7mer', dtype='uint8')
                want_cols = [f'7mer_{k}' for k in group]
                missing = [c for c in want_cols if c not in dummies_train.columns]
                if missing:
                    for c in missing:
                        dummies_train[c] = 0
                dummies_train = dummies_train[want_cols]
                sub_train[dummies_train.columns] = dummies_train
                
                X_train = sub_train[['treatment'] + list(dummies_train.columns)]
                y_train = sub_train['ems_count'].values
                offset_train = sub_train['log_depth'].values
                
                result = None
                try:
                    result = fit_glm(y_train, X_train, offset_train, family=glm_family, nb_alpha=nb_alpha)
                    fitted_models.append(result)  # Track for alpha extraction
                    llf = float(getattr(result, 'llf', np.nan))
                    k_params = int(len(result.params)) if hasattr(result, 'params') else (1 + dummies_train.shape[1])
                    n_obs = int(len(sub_train))
                    if np.isfinite(llf):
                        total_llf += llf
                        total_params += k_params
                        total_n += n_obs
                except Exception:
                    continue
                
                # Predict on test fold for this group (only if model fit succeeded)
                if result is None:
                    # Clean up before continuing
                    del sub_train, dummies_train, X_train, y_train, offset_train
                    gc.collect()
                    continue
                    
                sub_test = df_test[df_test['kmer7'].isin(group)].copy()
                if sub_test.empty:
                    del sub_train, dummies_train, X_train, y_train, offset_train, result
                    gc.collect()
                    continue
                
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
                    mu_test = result.predict(design_test, offset=offset_test)
                    test_predictions.extend(mu_test)
                    test_actuals.extend(sub_test['ems_count'].values)
                except Exception:
                    pass
                finally:
                    # Clean up after each group
                    del sub_train, sub_test, dummies_train, dummies_test
                    del X_train, X_test, X_test_df, design_test
                    del y_train, offset_train, offset_test, mu_test
                    del result
                    gc.collect()
            
            if len(test_predictions) == 0:
                continue
            
            test_predictions = np.array(test_predictions)
            test_actuals = np.array(test_actuals)
            
            mse = np.mean((test_actuals - test_predictions) ** 2)
            mae = np.mean(np.abs(test_actuals - test_predictions))
            
            if len(test_actuals) > 1 and np.std(test_predictions) > 0:
                pearson_r = np.corrcoef(test_actuals, test_predictions)[0, 1]
            else:
                pearson_r = np.nan
            
            mu_safe = np.maximum(test_predictions, 1e-10)
            y_safe = np.maximum(test_actuals, 0)
            
            # Extract nb_alpha from fitted models if not provided
            fold_nb_alpha = nb_alpha
            if fold_nb_alpha is None and glm_family == 'negative_binomial' and fitted_models:
                # Try to extract alpha from first successfully fitted model
                for model in fitted_models:
                    if model is None:
                        continue
                    # Check if it's a GLM NB model
                    if hasattr(model, 'family') and hasattr(model.family, 'alpha'):
                        fold_nb_alpha = model.family.alpha
                        break
                    # Check if it's an NBDiscrete model
                    elif 'NegativeBinomial' in str(type(model)) or 'NBDiscrete' in str(type(model)):
                        try:
                            if hasattr(model, 'params') and len(model.params) > 0:
                                fold_nb_alpha = float(model.params.iloc[-1])
                                break
                        except:
                            continue
            
            if fold_nb_alpha is not None and fold_nb_alpha > 0:
                # Negative Binomial deviance
                inv_alpha = 1.0 / fold_nb_alpha
                y_plus_inv_alpha = y_safe + inv_alpha
                mu_plus_inv_alpha = mu_safe + inv_alpha
                
                y_log_term = np.where(y_safe > 0, y_safe * np.log((y_safe + 1e-10) / mu_safe), 0)
                combined_log_term = y_plus_inv_alpha * np.log((y_plus_inv_alpha + 1e-10) / (mu_plus_inv_alpha + 1e-10))
                
                deviance = 2 * np.sum(y_log_term - combined_log_term)
            else:
                # Poisson deviance (fallback)
                deviance = 2 * np.sum(
                    y_safe * np.log((y_safe + 1e-10) / mu_safe) - (y_safe - mu_safe)
                )
            
            fold_metrics.append({
                'fold': fold_idx + 1,
                'mse': mse,
                'mae': mae,
                'pearson_r': pearson_r,
                'poisson_deviance': deviance,  # Keep key name for compatibility, but value is NB if alpha provided
                'n_test': len(test_actuals)
            })
            
            # Clean up memory
            del df_train, df_test, fitted_models
            del test_predictions, test_actuals
            gc.collect()
        except Exception as e:
            print(f"Warning: 7mer CV fold {fold_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not fold_metrics:
        return None
    
    metrics_df = pd.DataFrame(fold_metrics)
    return {
        'mse_mean': metrics_df['mse'].mean(),
        'mse_std': metrics_df['mse'].std(),
        'mae_mean': metrics_df['mae'].mean(),
        'mae_std': metrics_df['mae'].std(),
        'pearson_r_mean': metrics_df['pearson_r'].mean(),
        'pearson_r_std': metrics_df['pearson_r'].std(),
        'poisson_dev_mean': metrics_df['poisson_deviance'].mean(),
        'poisson_dev_std': metrics_df['poisson_deviance'].std(),
        'n_folds': len(fold_metrics),
        'fold_details': fold_metrics
    }


def cross_validate_all_models(
    df: pd.DataFrame,
    glm_family: str = 'poisson',
    nb_alpha: float = None,
    n_folds: int = 5,
    output_dir: str = None,
    sevenmer_splits: int = 8,
    skip_7mer: bool = False
) -> Dict:
    """
    Cross-validate all sequence bias models.
    
    Args:
        skip_7mer: If True, skip the 7mer_split model (useful if memory is limited)
    
    Returns dictionary with CV metrics for each model.
    """
    df = df[df['depth'] > 0].copy()
    df['log_depth'] = np.log(df['depth'])
    
    cv_results = {}
    
    # Positional model
    print("Cross-validating positional model...")
    df_pos = df.copy()
    pos_cols = add_positional_features(df_pos)
    cv_results['positional'] = cross_validate_model(
        df_pos, pos_cols, 'ems_count', 'log_depth',
        glm_family, nb_alpha, n_folds
    )
    del df_pos
    gc.collect()
    
    # 3mer model
    print("Cross-validating 3mer model...")
    df_tri = df.copy()
    tri_cols = add_3mer_features(df_tri)
    cv_results['3mer'] = cross_validate_model(
        df_tri, tri_cols, 'ems_count', 'log_depth',
        glm_family, nb_alpha, n_folds
    )
    del df_tri
    gc.collect()
    
    # Positional-3mer model
    print("Cross-validating pos3mer model...")
    df_p3 = df.copy()
    p3_cols = add_pos3mer_features(df_p3)
    cv_results['pos3mer'] = cross_validate_model(
        df_p3, p3_cols, 'ems_count', 'log_depth',
        glm_family, nb_alpha, n_folds
    )
    del df_p3
    gc.collect()
    
    # 5mer model
    print("Cross-validating 5mer model...")
    df_k5 = df.copy()
    k5_cols = add_5mer_features(df_k5)
    cv_results['5mer'] = cross_validate_model(
        df_k5, k5_cols, 'ems_count', 'log_depth',
        glm_family, nb_alpha, n_folds
    )
    del df_k5
    gc.collect()
    
    # 7mer model (using split-fitting approach)
    if not skip_7mer:
        print("Cross-validating 7mer model (split-fitting approach)...")
        try:
            cv_results['7mer_split'] = cross_validate_7mer_splits(
                df, glm_family, nb_alpha, n_folds, sevenmer_splits
            )
        except MemoryError as e:
            print(f"Warning: 7mer_split CV failed due to memory error: {e}")
            print("Consider using --skip-7mer to skip this model in future runs.")
            cv_results['7mer_split'] = None
    else:
        print("Skipping 7mer_split model (--skip-7mer flag set)")
        cv_results['7mer_split'] = None
    
    if output_dir:
        import json
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        summary = {k: {
            'mse_mean': v['mse_mean'],
            'mae_mean': v['mae_mean'],
            'pearson_r_mean': v['pearson_r_mean'],
            'poisson_dev_mean': v['poisson_dev_mean'],
            'mse_std': v['mse_std'],
            'pearson_r_std': v['pearson_r_std']
        } for k, v in cv_results.items() if v is not None}
        
        with open(os.path.join(output_dir, 'cv_results_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    return cv_results


if __name__ == '__main__':
    import argparse
    from sequence_bias_modeling_sitelevel import load_site_level_data
    
    parser = argparse.ArgumentParser(description='Cross-validate sequence bias models')
    parser.add_argument('--counts-dir', required=True)
    parser.add_argument('--genome-fasta', required=True)
    parser.add_argument('--exclusion-mask', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--glm-family', choices=['poisson', 'negative_binomial'], default='poisson')
    parser.add_argument('--nb-alpha', type=float, default=None)
    parser.add_argument('--skip-7mer', action='store_true', 
                        help='Skip 7mer_split model to save memory')
    parser.add_argument('--sevenmer-splits', type=int, default=8,
                        help='Number of splits for 7mer model (default: 8)')
    
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_site_level_data(args.counts_dir, args.genome_fasta, args.exclusion_mask)
    
    print(f"Loaded {len(df):,} sites")
    print(f"Performing {args.n_folds}-fold cross-validation...")
    
    cv_results = cross_validate_all_models(
        df, args.glm_family, args.nb_alpha, args.n_folds, args.output_dir, 
        sevenmer_splits=args.sevenmer_splits, skip_7mer=args.skip_7mer
    )
    
    print("\nCross-Validation Results:")
    print("=" * 80)
    for model_name, metrics in cv_results.items():
        if metrics:
            print(f"\n{model_name}:")
            print(f"  MSE: {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}")
            print(f"  MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
            print(f"  Pearson r: {metrics['pearson_r_mean']:.4f} ± {metrics['pearson_r_std']:.4f}")
            dev_name = "NB Deviance" if args.nb_alpha is not None and args.nb_alpha > 0 else "Poisson Deviance"
            print(f"  {dev_name}: {metrics['poisson_dev_mean']:.2f} ± {metrics['poisson_dev_std']:.2f}")

