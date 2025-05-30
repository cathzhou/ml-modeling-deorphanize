import requests
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import time
import urllib.parse

class ExpressionDataExtractor:
    def __init__(self, cache_dir='data/expression_cache'):
        """Initialize the expression data extractor.
        
        Args:
            cache_dir (str): Directory to cache expression matrices
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.proteinatlas.org"
    
    def get_expression_matrix(self, uniprot_id):
        """Get expression matrix for a gene from HPA API.
        
        Args:
            uniprot_id (str): UniProt identifier (e.g., 'P41143')
            
        Returns:
            pd.DataFrame: Expression matrix with single cell RNA expression data
        """
        cache_file = self.cache_dir / f"{uniprot_id}_expression.parquet"
        
        # Check cache first
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        
        # Build the search URL with parameters
        params = {
            'search': uniprot_id,
            'format': 'json',
            'columns': 'g,eg,up,rnascd,rnascsm',
            'compress': 'no'
        }
        search_url = f"{self.base_url}/api/search_download.php?{urllib.parse.urlencode(params)}"
        
        try:
            print(f"Searching for UniProt ID: {uniprot_id}")
            print(f"URL: {search_url}")
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"No data found for UniProt ID: {uniprot_id}")
                return None
            
            # Find the matching entry with our UniProt ID
            matching_entries = [entry for entry in data if entry.get('up') == uniprot_id]
            if not matching_entries:
                print(f"No exact match found for UniProt ID: {uniprot_id}")
                return None
            
            # Process the expression data
            expression_data = []
            for entry in matching_entries:
                if 'rnascd' in entry and 'rnascsm' in entry:
                    cell_types = entry['rnascd'].split(';')
                    ntpm_values = entry['rnascsm'].split(';')
                    
                    for cell_type, ntpm in zip(cell_types, ntpm_values):
                        if cell_type and ntpm:
                            try:
                                expression_data.append({
                                    'cell_type': cell_type.strip(),
                                    'nTPM': float(ntpm.strip())
                                })
                            except ValueError:
                                continue
            
            if not expression_data:
                print(f"No RNA expression data found for {uniprot_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(expression_data)
            matrix = df.pivot_table(
                index='cell_type',
                values='nTPM',
                aggfunc='mean'  # In case of duplicates
            )
            
            # Cache result
            matrix.to_parquet(cache_file)
            time.sleep(1)  # Rate limiting
            return matrix
            
        except Exception as e:
            print(f"Error fetching data for {uniprot_id}: {str(e)}")
            return None
    
    def compute_coexpression_features(self, receptor_expr, ligand_expr):
        """Compute co-expression features for a receptor-ligand pair.
        
        Args:
            receptor_expr (pd.DataFrame): Receptor expression matrix
            ligand_expr (pd.DataFrame): Ligand expression matrix
            
        Returns:
            dict: Co-expression features
        """
        if receptor_expr is None or ligand_expr is None:
            return None
            
        # Align matrices - now working with cell types only
        common_cells = sorted(set(receptor_expr.index) & set(ligand_expr.index))
        if not common_cells:
            print("No common cell types found between receptor and ligand")
            return None
            
        rec_mat = receptor_expr.loc[common_cells].values.flatten()
        lig_mat = ligand_expr.loc[common_cells].values.flatten()
        
        # Binary expression (threshold at median)
        rec_binary = (rec_mat > np.median(rec_mat)).astype(int)
        lig_binary = (lig_mat > np.median(lig_mat)).astype(int)
        
        features = {
            'pearson_corr': np.corrcoef(rec_mat, lig_mat)[0,1],
            'cosine_sim': np.dot(rec_mat, lig_mat) / (np.linalg.norm(rec_mat) * np.linalg.norm(lig_mat)),
            'jaccard_index': np.sum(rec_binary & lig_binary) / np.sum(rec_binary | lig_binary),
            'l2_norm_diff': np.linalg.norm(rec_mat - lig_mat),
            'overlap_count': np.sum(rec_binary & lig_binary),
            'shared_top10_count': len(
                set(np.argsort(rec_mat)[-10:]) & 
                set(np.argsort(lig_mat)[-10:])
            ),
            'common_cell_types': len(common_cells)
        }
        
        return features
    
    def process_receptor_ligand_pairs(self, pairs_df, output_file='data/coexpression_features.h5'):
        """Process all receptor-ligand pairs and save features.
        
        Args:
            pairs_df (pd.DataFrame): DataFrame with receptor_id and ligand_id columns
            output_file (str): Path to save results
        """
        features_list = []
        
        for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
            receptor_expr = self.get_expression_matrix(row['receptor_id'])
            ligand_expr = self.get_expression_matrix(row['ligand_id'])
            
            features = self.compute_coexpression_features(receptor_expr, ligand_expr)
            if features:
                features['receptor_id'] = row['receptor_id']
                features['ligand_id'] = row['ligand_id']
                features_list.append(features)
        
        # Save results
        features_df = pd.DataFrame(features_list)
        if not features_df.empty:
            features_df.to_hdf(output_file, key='coexpression_features', mode='w')
        
        return features_df

if __name__ == "__main__":
    # Example usage
    extractor = ExpressionDataExtractor()
    
    # Example with UniProt IDs
    test_pairs = pd.DataFrame({
        'receptor_id': ['P50052', 'P30411'],  # AGTR2, BKRB2
        'ligand_id': ['O15467', 'O15263']     # CCL16, DFB4A
    })
    features_df = extractor.process_receptor_ligand_pairs(test_pairs)
    print(f"Processed {len(features_df) if features_df is not None else 0} pairs successfully") 