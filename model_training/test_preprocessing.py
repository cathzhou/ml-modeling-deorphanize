import os
import pandas as pd
import logging
from pathlib import Path
from data_preprocessing import DataPreprocessor
import json
from typing import Dict, List, Optional
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_preprocessing_test(
    data_config_path: str,
    model_config_path: str,
    output_dir: str
) -> None:
    """
    Run preprocessing with specified config files and save results.
    
    Args:
        data_config_path: Path to data configuration file
        model_config_path: Path to model configuration file  
        output_dir: Directory to save results
    """
    # Load model config to get test name
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    test_name = model_config['name']
    
    # Create output directory
    test_dir = os.path.join(output_dir, test_name)
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize preprocessor with config files
    preprocessor = DataPreprocessor(data_config_path, model_config_path)

    # Run preprocessing
    logging.info(f"\n{'='*60}")
    logging.info(f"Running test: {test_name}")
    logging.info(f"Data config: {data_config_path}")
    logging.info(f"Model config: {model_config_path}")
    logging.info(f"Split method: {model_config['data_params'].get('split_method', 'random')}")
    
    # Load raw data first to check initial state
    data_path = model_config['data_path']
    df = pd.read_csv(data_path)
    logging.info(f"Initial data shape: {df.shape}")
    logging.info(f"Initial known pairs: {(df['known_pair'] == 'known').sum()}")
    logging.info(f"Initial unknown pairs: {(df['known_pair'] == 'unknown').sum()}")
    
    # Check for afpd_dir_name to verify pair structure
    if 'afpd_dir_name' in df.columns:
        unique_pairs = df['afpd_dir_name'].nunique()
        avg_models_per_pair = len(df) / unique_pairs
        logging.info(f"Number of unique pairs: {unique_pairs}")
        logging.info(f"Average models per pair: {avg_models_per_pair:.1f}")
    
    # Show filters being applied
    filters = model_config.get('filters')
    if filters:
        logging.info("Filters applied:")
        if 'column_equals' in filters:
            for col, val in filters['column_equals'].items():
                count = (df[col] == val).sum()
                logging.info(f"  {col} == {val}: {count} rows")
        if 'non_nan_columns' in filters:
            logging.info(f"  Non-NaN required: {filters['non_nan_columns']}")
    
    # Show active feature groups
    active_features = [k for k, v in model_config['feature_groups'].items() if v]
    logging.info(f"Active feature groups: {active_features}")
    
    # Run preprocessing using the config file settings
    try:
        datasets, scalers = preprocessor.preprocess_data()
        logging.info("✓ Preprocessing completed successfully")
    except Exception as e:
        logging.error(f"✗ Preprocessing failed: {e}")
        return

def main():
    parser = argparse.ArgumentParser(description='Test preprocessing for one or more model configs')
    parser.add_argument('--data-config', type=str, required=True, help='Path to data config file')
    parser.add_argument('--model-config', type=str, default=None, help='Path to a single model config file')
    parser.add_argument('--config-dir', type=str, default=None, help='Directory containing model config files (*.json)')
    parser.add_argument('--output-dir', type=str, default='data/preprocessing_tests', help='Directory to save results')
    args = parser.parse_args()

    if args.model_config:
        # Run a single config
        run_preprocessing_test(args.data_config, args.model_config, args.output_dir)
    elif args.config_dir:
        # Run all configs in the directory
        configs = [os.path.join(args.config_dir, f) for f in os.listdir(args.config_dir) if f.endswith('.json')]
        if not configs:
            print(f'No config files found in {args.config_dir}')
            return
        for config_path in configs:
            print(f'\nRunning preprocessing for config: {config_path}')
            run_preprocessing_test(args.data_config, config_path, args.output_dir)
    else:
        parser.print_help()
        return

if __name__ == '__main__':
    main() 