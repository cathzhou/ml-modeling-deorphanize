import pandas as pd
import numpy as np
from expression_data_extractor import ExpressionDataExtractor
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/coexpression_processing.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Initialize the expression data extractor
    extractor = ExpressionDataExtractor(cache_dir='data/expression_cache')
    
    # Read the receptor-ligand pairs
    pairs_df = pd.read_csv('data/bm_update_3_subset_rec_lig_pairs.csv')
    total_pairs = len(pairs_df)
    logging.info(f"Processing {total_pairs} receptor-ligand pairs")
    
    # Rename columns if needed
    if 'p1_id' in pairs_df.columns and 'p2_id' in pairs_df.columns:
        pairs_df = pairs_df.rename(columns={
            'p1_id': 'receptor_id',
            'p2_id': 'ligand_id'
        })
    
    # Keep track of missing proteins
    missing_proteins = set()
    
    # Process the pairs and generate features
    features_df = extractor.process_receptor_ligand_pairs(
        pairs_df,
        output_file='data/coexpression_features.csv'
    )
    
    # Log statistics
    if features_df is not None:
        success_rate = (len(features_df) / total_pairs) * 100
        logging.info(f"Successfully processed {len(features_df)} out of {total_pairs} pairs ({success_rate:.2f}%)")
        logging.info(f"Results saved to data/coexpression_features.csv")
    else:
        logging.warning("No features were generated")

if __name__ == "__main__":
    main() 