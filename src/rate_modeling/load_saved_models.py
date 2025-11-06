#!/usr/bin/env python3
"""
Utility functions to load saved models from sequence_bias_modeling_sitelevel.py

Allows downstream evaluation scripts to use saved models without rerunning the main script.
"""
import os
import pickle
import json
import pandas as pd
from typing import Dict, Tuple, Optional


def load_saved_models(models_dir: str) -> Tuple[Dict, Dict]:
    """
    Load fitted models and metadata from saved pickle files.
    
    Args:
        models_dir: Path to directory containing fitted models (e.g., output_dir/fitted_models)
    
    Returns:
        Tuple of (models_dict, metadata_dict)
        - models_dict: {model_name: fitted_model_object}
        - metadata_dict: {model_name: {'feature_cols': [...]}}
    """
    models = {}
    metadata = {}
    
    # Load metadata if available
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Load models
    model_files = {
        'positional': 'positional_model.pkl',
        '3mer': '3mer_model.pkl',
        '5mer': '5mer_model.pkl',
        'pos3mer': 'pos3mer_model.pkl',
        '7mer_split': '7mer_split_model.pkl',
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                if model_name == '7mer_split' and isinstance(models[model_name], list):
                    print(f"Loaded {model_name} model ({len(models[model_name])} split models) from {model_path}")
                else:
                    print(f"Loaded {model_name} model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load {model_name} model: {e}")
        else:
            print(f"Warning: Model file not found: {model_path}")
    
    return models, metadata


def load_summary_json(summary_path: str) -> Dict:
    """
    Load summary JSON from main modeling script.
    
    Args:
        summary_path: Path to sequence_bias_sitelevel_summary.json
    
    Returns:
        Dictionary with summary information
    """
    with open(summary_path, 'r') as f:
        return json.load(f)


def prepare_data_for_model(
    df: pd.DataFrame,
    model_name: str,
    metadata: Dict = None
) -> Tuple[pd.DataFrame, list]:
    """
    Prepare data with correct features for a specific model.
    
    Args:
        df: DataFrame with site-level data (must have columns: chrom, pos, treatment, depth, ems_count, kmer5)
        model_name: Name of model ('positional', '3mer', '5mer', 'pos3mer')
        metadata: Optional metadata dict (if not provided, will extract features from df)
    
    Returns:
        Tuple of (X, feature_cols)
        - X: DataFrame with treatment + feature columns
        - feature_cols: List of feature column names
    """
    from sequence_bias_modeling_sitelevel import (
        add_positional_features, add_3mer_features, 
        add_5mer_features, add_pos3mer_features
    )
    
    df = df[df['depth'] > 0].copy()
    
    if model_name == 'positional':
        if metadata and 'positional' in metadata and 'feature_cols' in metadata['positional']:
            feature_cols = metadata['positional']['feature_cols']
        else:
            feature_cols = add_positional_features(df.copy())
    elif model_name == '3mer':
        if metadata and '3mer' in metadata and 'feature_cols' in metadata['3mer']:
            feature_cols = metadata['3mer']['feature_cols']
        else:
            feature_cols = add_3mer_features(df.copy())
    elif model_name == '5mer':
        if metadata and '5mer' in metadata and 'feature_cols' in metadata['5mer']:
            feature_cols = metadata['5mer']['feature_cols']
        else:
            feature_cols = add_5mer_features(df.copy())
    elif model_name == 'pos3mer':
        if metadata and 'pos3mer' in metadata and 'feature_cols' in metadata['pos3mer']:
            feature_cols = metadata['pos3mer']['feature_cols']
        else:
            feature_cols = add_pos3mer_features(df.copy())
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Ensure features exist in df
    df = df.copy()
    if model_name == 'positional' and not all(col in df.columns for col in feature_cols):
        feature_cols = add_positional_features(df)
    elif model_name == '3mer' and not all(col in df.columns for col in feature_cols):
        feature_cols = add_3mer_features(df)
    elif model_name == 'pos3mer' and not all(col in df.columns for col in feature_cols):
        feature_cols = add_pos3mer_features(df)
    elif model_name == '5mer' and not all(col in df.columns for col in feature_cols):
        feature_cols = add_5mer_features(df)
    
    # Create feature matrix
    X = df[['treatment'] + feature_cols].copy()
    
    return X, feature_cols


def load_models_and_prepare_data(
    output_dir: str,
    df: pd.DataFrame = None
) -> Dict:
    """
    Convenience function to load models and prepare data structures.
    
    Args:
        output_dir: Output directory from main modeling script
        df: Optional DataFrame (if None, will need to be provided later)
    
    Returns:
        Dictionary with:
        - 'models': {model_name: fitted_model}
        - 'metadata': {model_name: metadata_dict}
        - 'summary': summary from JSON (if available)
        - 'df': input DataFrame (if provided)
    """
    models_dir = os.path.join(output_dir, 'fitted_models')
    summary_path = os.path.join(output_dir, 'sequence_bias_sitelevel_summary.json')
    
    models, metadata = load_saved_models(models_dir)
    
    result = {
        'models': models,
        'metadata': metadata,
        'summary': None,
        'df': df
    }
    
    if os.path.exists(summary_path):
        result['summary'] = load_summary_json(summary_path)
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test loading saved models')
    parser.add_argument('--output-dir', required=True, help='Output directory from main modeling script')
    
    args = parser.parse_args()
    
    print(f"Loading models from {args.output_dir}...")
    result = load_models_and_prepare_data(args.output_dir)
    
    print(f"\nLoaded {len(result['models'])} models:")
    for model_name in result['models'].keys():
        print(f"  - {model_name}")
    
    if result['summary']:
        print(f"\nSummary info:")
        print(f"  Total sites: {result['summary'].get('df_len', 'N/A')}")
        print(f"  GLM family: {result['summary'].get('glm_family', 'N/A')}")

