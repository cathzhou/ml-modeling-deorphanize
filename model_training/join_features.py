import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Read the datasets
    logging.info("Reading input files...")
    bm_df = pd.read_csv('data/bm_update_3_subset_use.csv')
    coexp_df = pd.read_csv('data/coexpression_features.csv')
    
    # Rename columns in coexpression features to match
    coexp_df = coexp_df.rename(columns={
        'receptor_id': 'p1_id',
        'ligand_id': 'p2_id'
    })
    
    # Perform left joins
    logging.info("Performing left joins...")
    
    # First join coexpression features
    merged_df = bm_df.merge(
        coexp_df,
        on=['p1_id', 'p2_id'],
        how='left'
    )
    
    # Save the merged dataset
    output_file = 'data/bm_update_3_subset_lig_features_coexp.csv'
    logging.info(f"Saving merged dataset to {output_file}")
    merged_df.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    main() 