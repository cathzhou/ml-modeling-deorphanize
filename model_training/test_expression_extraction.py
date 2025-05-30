import pandas as pd
from expression_data_extractor import ExpressionDataExtractor

def test_extraction():
    # Create a small test set of receptor-ligand pairs using UniProt IDs
    test_pairs = pd.DataFrame({
        'receptor_id': [
            'P50052',  # AGTR2
            'P30411'   # BKRB2
        ],
        'ligand_id': [
            'O15467',  # CCL16	
            'O15263'   # DFB4A	
        ]
    })
    
    print("Testing expression data extraction with the following pairs:")
    for _, row in test_pairs.iterrows():
        print(f"Receptor: {row['receptor_id']} (UniProt)")
        print(f"Ligand: {row['ligand_id']} (UniProt)\n")
    
    # Initialize extractor
    extractor = ExpressionDataExtractor(cache_dir='data/test_expression_cache')
    
    # Process pairs
    features_df = extractor.process_receptor_ligand_pairs(
        test_pairs,
        output_file='data/test_coexpression_features.h5'
    )
    
    # Print results
    print("\nExtracted features:")
    print(features_df)
    
    if features_df is not None and not features_df.empty:
        print("\nFeature descriptions:")
        print("- pearson_corr: Correlation between receptor and ligand expression")
        print("- cosine_sim: Cosine similarity between expression vectors")
        print("- jaccard_index: Overlap in binary expression patterns")
        print("- l2_norm_diff: L2 distance between expression vectors")
        print("- overlap_count: Number of cell types with shared expression")
        print("- shared_top10_count: Number of shared top-10 expressing cell types")
        print("- common_cell_types: Total number of cell types with data for both")
    else:
        print("\nNo features could be extracted. This could be because:")
        print("1. The genes were not found in the HPA database")
        print("2. No single-cell RNA expression data is available")
        print("3. There were no common cell types between the pairs")
    
    return features_df

if __name__ == "__main__":
    test_extraction() 