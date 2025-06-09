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
    data_path: str,
    output_dir: str,
    test_name: str,
    filters: Optional[Dict] = None,
    feature_groups: Optional[Dict[str, bool]] = None,
    split_method: str = 'umap',
    test_size: float = 0.1,
    valid_size: float = 0.1
) -> None:
    """
    Run preprocessing with specified filters and save results.
    
    Args:
        data_path: Path to input data CSV
        output_dir: Directory to save results
        test_name: Name of this test configuration
        filters: Optional dictionary of filters to apply
        feature_groups: Optional dictionary to override feature group settings
        split_method: Method for splitting data ('umap' or 'random')
        test_size: Fraction of data for test set
        valid_size: Fraction of data for validation set
    """
    # Create output directory
    test_dir = os.path.join(output_dir, test_name)
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Save test configuration
    config = {
        'test_name': test_name,
        'filters': filters,
        'feature_groups': feature_groups,
        'split_method': split_method,
        'test_size': test_size,
        'valid_size': valid_size,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(test_dir, 'test_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run preprocessing
    logging.info(f"\nRunning test: {test_name}")
    if filters:
        logging.info("Filters:")
        for k, v in filters.items():
            logging.info(f"  {k}: {v}")
    
    # Load raw data first to check initial state
    df = pd.read_csv(data_path)
    logging.info(f"Initial data shape: {df.shape}")
    logging.info(f"Initial known pairs: {(df['known_pair'] == 'known').sum()}")
    logging.info(f"Initial unknown pairs: {(df['known_pair'] == 'unknown').sum()}")
    
    if filters and 'column_equals' in filters:
        for col, val in filters['column_equals'].items():
            count = (df[col] == val).sum()
            logging.info(f"Rows matching {col} == {val}: {count}")
    
    datasets, _ = preprocessor.preprocess_data(
        data_path=data_path,
        split_method=split_method,
        test_size=test_size,
        valid_size=valid_size,
        filters=filters,
        feature_groups=feature_groups
    )
    
    # Save processed data
    processed_path = Path(data_path).parent / 'processed_features_with_splits.csv'
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        output_path = os.path.join(test_dir, 'processed_features.csv')
        df.to_csv(output_path, index=False)
        logging.info(f"Saved processed features to: {output_path}")
        
        # Save split-specific CSVs
        for split in ['train', 'valid', 'test']:
            split_df = df[df['split'] == split]
            split_path = os.path.join(test_dir, f'{split}_set.csv')
            split_df.to_csv(split_path, index=False)
            logging.info(f"Saved {split} set to: {split_path}")
            
        # Generate summary statistics
        summary = {
            'total_samples': len(df),
            'known_pairs': (df['known_pair'] == 1).sum(),
            'unknown_pairs': (df['known_pair'] == 0).sum(),
            'splits': {
                split: {
                    'total': len(df[df['split'] == split]),
                    'known': len(df[(df['split'] == split) & (df['known_pair'] == 1)]),
                    'unknown': len(df[(df['split'] == split) & (df['known_pair'] == 0)])
                }
                for split in ['train', 'valid', 'test']
            }
        }
        
        # Add location statistics if available
        if 'lig1_location' in df.columns:
            summary['location_distribution'] = df['lig1_location'].value_counts().to_dict()
        
        # Save summary
        with open(os.path.join(test_dir, 'summary_stats.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        logging.info("\nSummary Statistics:")
        logging.info(f"Total samples: {summary['total_samples']}")
        logging.info(f"Known pairs: {summary['known_pairs']}")
        logging.info(f"Unknown pairs: {summary['unknown_pairs']}")
        for split, stats in summary['splits'].items():
            logging.info(f"\n{split.capitalize()} set:")
            logging.info(f"  Total: {stats['total']}")
            logging.info(f"  Known: {stats['known']}")
            logging.info(f"  Unknown: {stats['unknown']}")

def main():
    # Define paths
    data_path = "data/bm_update_3_subset_lig_features_coexpression.csv"
    output_dir = "data/preprocessing_tests"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Example test configurations
    test_configs = [
        {
            'name': 'extracellular',
            'filters': {
                'column_equals': {'lig1_location': 'E'},
                'non_nan_columns': ['nmfUMAP1_af_qc', 'nmfUMAP2_af_qc']
            },
            'feature_groups': {
                'residue_contacts': True,
                'distance_metrics': True,
                'ligand_contact_sum': True,
                'ligand_contact_indiv': True,
                'alphafold_metrics': True,
                'expression_features': True,
                'ligand_metadata': True
            }
        }
    ]
    
    # Run all test configurations
    for config in test_configs:
        run_preprocessing_test(
            data_path=data_path,
            output_dir=output_dir,
            test_name=config['name'],
            filters=config.get('filters'),
            feature_groups=config.get('feature_groups'),
            split_method=config.get('split_method', 'umap')
        )

if __name__ == "__main__":
    main() 