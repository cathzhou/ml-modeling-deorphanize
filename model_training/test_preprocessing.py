import os
import pandas as pd
import logging
from pathlib import Path
from data_preprocessing import DataPreprocessor
import json
from typing import Dict, List, Optional
from datetime import datetime

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
    # Define paths
    data_config_path = "model_training/data_config.json"
    output_dir = "data/preprocessing_tests"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all config files from model_config directory
    model_config_dir = "model_training/model_config"
    #config_files = [f for f in os.listdir(model_config_dir) if f.endswith('.json')]
    config_files = ['train_config_extracellular_random.json']
    
    logging.info(f"Found {len(config_files)} config files to test:")
    for config_file in config_files:
        logging.info(f"  - {config_file}")
    
    # Run preprocessing for each config file
    for config_file in sorted(config_files):
        model_config_path = os.path.join(model_config_dir, config_file)
        run_preprocessing_test(
            data_config_path=data_config_path,
            model_config_path=model_config_path,
            output_dir=output_dir
        )
    
    logging.info(f"\n{'='*60}")
    logging.info("All preprocessing tests completed!")
    logging.info(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main() 