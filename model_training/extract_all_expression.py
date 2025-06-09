import pandas as pd
from expression_data_extractor import ExpressionDataExtractor
from pathlib import Path
import time
from tqdm import tqdm

def main():
    # Read the receptor-ligand pairs
    print("Reading receptor-ligand pairs...")
    df = pd.read_csv('data/bm_update_3_subset_rec_lig_pairs.csv')
    
    # Get unique protein IDs
    p1_ids = sorted(df['p1_id'].unique())
    p2_ids = sorted(df['p2_id'].unique())
    all_ids = sorted(set(p1_ids) | set(p2_ids))
    
    print(f"Found {len(p1_ids)} unique receptor IDs")
    print(f"Found {len(p2_ids)} unique ligand IDs")
    print(f"Total unique protein IDs: {len(all_ids)}")
    
    # Initialize expression data extractor
    extractor = ExpressionDataExtractor(cache_dir='data/expression_cache')
    
    # Process each protein ID
    print("\nExtracting expression data for all proteins...")
    for protein_id in tqdm(all_ids):
        try:
            tissue_matrix, cell_type_matrix = extractor.get_expression_matrix(protein_id)
            if tissue_matrix is None or cell_type_matrix is None:
                print(f"\nNo expression data found for {protein_id}")
            time.sleep(1)  # Rate limiting for API calls
        except Exception as e:
            print(f"\nError processing {protein_id}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 