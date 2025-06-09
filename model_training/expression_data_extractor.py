import requests
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import time
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class ExpressionDataExtractor:
    def __init__(self, cache_dir='data/expression_cache'):
        """Initialize the expression data extractor.
        
        Args:
            cache_dir (str): Directory to cache expression matrices
        """
        self.cache_dir = Path(cache_dir)
        self.plots_dir = self.cache_dir / 'plots'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.proteinatlas.org"
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')  # Use the updated seaborn style name
    
    def plot_expression_data(self, uniprot_id, tissue_matrix, cell_type_matrix):
        """Create bar plots for tissue and cell type expression data.
        
        Args:
            uniprot_id (str): UniProt ID of the gene
            tissue_matrix (pd.DataFrame): Tissue expression matrix
            cell_type_matrix (pd.DataFrame): Cell type expression matrix
        """
        if tissue_matrix is None or cell_type_matrix is None:
            return
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle(f'Expression Profile for {uniprot_id}', fontsize=16)
        
        # Plot tissue expression
        tissue_data = tissue_matrix.iloc[0]
        tissue_data = tissue_data.sort_values(ascending=False)
        
        sns.barplot(x=tissue_data.values, y=tissue_data.index.str.replace('t_RNA_', '').str.replace('_', ' '), 
                   ax=ax1, color='skyblue')
        ax1.set_title('Tissue Expression (nTPM)')
        ax1.set_xlabel('Expression Level (nTPM)')
        ax1.set_ylabel('Tissue')
        
        # Plot cell type expression
        cell_data = cell_type_matrix.iloc[0]
        cell_data = cell_data.sort_values(ascending=False)
        
        sns.barplot(x=cell_data.values, y=cell_data.index.str.replace('sc_RNA_', '').str.replace('_', ' '), 
                   ax=ax2, color='lightgreen')
        ax2.set_title('Single Cell Type Expression (nTPM)')
        ax2.set_xlabel('Expression Level (nTPM)')
        ax2.set_ylabel('Cell Type')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{uniprot_id}_expression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Expression plots saved to {self.plots_dir}/{uniprot_id}_expression.png")

    def get_expression_matrix(self, uniprot_id):
        """Get expression matrix for a gene from HPA API.
        
        Args:
            uniprot_id (str): UniProt identifier (e.g., 'P41143')
            
        Returns:
            tuple: (tissue_matrix, cell_type_matrix) as pd.DataFrames
        """
        cache_file = self.cache_dir / f"{uniprot_id}_expression.csv"
        
        # Check cache first
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0)
            # Split into tissue and cell type matrices
            tissue_cols = [col for col in df.columns if col.startswith('t_RNA_')]
            cell_type_cols = [col for col in df.columns if col.startswith('sc_RNA_')]
            tissue_matrix = df[tissue_cols]
            cell_type_matrix = df[cell_type_cols]
            
            # Generate plots
            # self.plot_expression_data(uniprot_id, tissue_matrix, cell_type_matrix)
            
            return tissue_matrix, cell_type_matrix
        
        # Build the search URL with parameters
        params = {
            'search': uniprot_id,
            'format': 'json',
            'columns': 'g,gs,up,sc_RNA_Adipocytes,sc_RNA_Alveolar_cells_type_1,sc_RNA_Alveolar_cells_type_2,sc_RNA_Astrocytes,sc_RNA_B-cells,sc_RNA_Basal_keratinocytes,sc_RNA_Basal_prostatic_cells,sc_RNA_Basal_respiratory_cells,sc_RNA_Basal_squamous_epithelial_cells,sc_RNA_Bipolar_cells,sc_RNA_Breast_glandular_cells,sc_RNA_Breast_myoepithelial_cells,sc_RNA_Cardiomyocytes,sc_RNA_Cholangiocytes,sc_RNA_Ciliated_cells,sc_RNA_Club_cells,sc_RNA_Collecting_duct_cells,sc_RNA_Cone_photoreceptor_cells,sc_RNA_Cytotrophoblasts,sc_RNA_dendritic_cells,sc_RNA_Distal_enterocytes,sc_RNA_Distal_tubular_cells,sc_RNA_Ductal_cells,sc_RNA_Early_spermatids,sc_RNA_Endometrial_stromal_cells,sc_RNA_Endothelial_cells,sc_RNA_Enteroendocrine_cells,sc_RNA_Erythroid_cells,sc_RNA_Excitatory_neurons,sc_RNA_Exocrine_glandular_cells,sc_RNA_Extravillous_trophoblasts,sc_RNA_Fibroblasts,sc_RNA_Gastric_mucus-secreting_cells,sc_RNA_Glandular_and_luminal_cells,sc_RNA_granulocytes,sc_RNA_Granulosa_cells,sc_RNA_Hepatocytes,sc_RNA_Hofbauer_cells,sc_RNA_Horizontal_cells,sc_RNA_Inhibitory_neurons,sc_RNA_Intestinal_goblet_cells,sc_RNA_Ionocytes,sc_RNA_Kupffer_cells,sc_RNA_Langerhans_cells,sc_RNA_Late_spermatids,sc_RNA_Leydig_cells,sc_RNA_Lymphatic_endothelial_cells,sc_RNA_Macrophages,sc_RNA_Melanocytes,sc_RNA_Mesothelial_cells,sc_RNA_Microglial_cells,sc_RNA_monocytes,sc_RNA_Mucus_glandular_cells,sc_RNA_Muller_glia_cells,sc_RNA_NK-cells,sc_RNA_Oligodendrocyte_precursor_cells,sc_RNA_Oligodendrocytes,sc_RNA_Oocytes,sc_RNA_Ovarian_stromal_cells,sc_RNA_Pancreatic_endocrine_cells,sc_RNA_Paneth_cells,sc_RNA_Peritubular_cells,sc_RNA_Plasma_cells,sc_RNA_Prostatic_glandular_cells,sc_RNA_Proximal_enterocytes,sc_RNA_Proximal_tubular_cells,sc_RNA_Rod_photoreceptor_cells,sc_RNA_Salivary_duct_cells,sc_RNA_Schwann_cells,sc_RNA_Secretory_cells,sc_RNA_Serous_glandular_cells,sc_RNA_Sertoli_cells,sc_RNA_Skeletal_myocytes,sc_RNA_Smooth_muscle_cells,sc_RNA_Spermatocytes,sc_RNA_Spermatogonia,sc_RNA_Squamous_epithelial_cells,sc_RNA_Suprabasal_keratinocytes,sc_RNA_Syncytiotrophoblasts,sc_RNA_T-cells,sc_RNA_Undifferentiated_cells,t_RNA_adipose_tissue,t_RNA_adrenal_gland,t_RNA_amygdala,t_RNA_appendix,t_RNA_basal_ganglia,t_RNA_bone_marrow,t_RNA_breast,t_RNA_cerebellum,t_RNA_cerebral_cortex,t_RNA_cervix,t_RNA_choroid_plexus,t_RNA_colon,t_RNA_duodenum,t_RNA_endometrium_1,t_RNA_epididymis,t_RNA_esophagus,t_RNA_fallopian_tube,t_RNA_gallbladder,t_RNA_heart_muscle,t_RNA_hippocampal_formation,t_RNA_hypothalamus,t_RNA_kidney,t_RNA_liver,t_RNA_lung,t_RNA_lymph_node,t_RNA_midbrain,t_RNA_ovary,t_RNA_pancreas,t_RNA_parathyroid_gland,t_RNA_pituitary_gland,t_RNA_placenta,t_RNA_prostate,t_RNA_rectum,t_RNA_retina,t_RNA_salivary_gland,t_RNA_seminal_vesicle,t_RNA_skeletal_muscle,t_RNA_skin_1,t_RNA_small_intestine,t_RNA_smooth_muscle,t_RNA_spinal_cord,t_RNA_spleen,t_RNA_stomach_1,t_RNA_testis,t_RNA_thymus,t_RNA_thyroid_gland,t_RNA_tongue,t_RNA_tonsil,t_RNA_urinary_bladder,t_RNA_vagina',
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
                return None, None
            
            # Process the expression data
            expression_data = {}
            
            # Get first entry (should only be one per UniProt ID)
            entry = data[0]
            
            # Extract tissue and single cell RNA data
            for key, value in entry.items():
                if key.startswith(('Tissue RNA - ', 'Single Cell Type RNA - ')) and key.endswith('[nTPM]'):
                    try:
                        cell_type = key.replace(' [nTPM]', '')
                        if key.startswith('Tissue RNA - '):
                            cell_type = 't_RNA_' + cell_type.replace('Tissue RNA - ', '').replace(' ', '_').lower()
                        else:
                            cell_type = 'sc_RNA_' + cell_type.replace('Single Cell Type RNA - ', '').replace(' ', '_')
                        expression_data[cell_type] = float(value)
                    except (ValueError, TypeError):
                        expression_data[cell_type] = 0.0
            
            if not expression_data:
                print(f"No RNA expression data found for {uniprot_id}")
                return None, None
            
            # Create DataFrame with gene as index
            df = pd.DataFrame([expression_data], index=[uniprot_id])
            
            # Split into tissue and cell type data
            tissue_cols = [col for col in df.columns if col.startswith('t_RNA_')]
            cell_type_cols = [col for col in df.columns if col.startswith('sc_RNA_')]
            
            tissue_matrix = df[tissue_cols]
            cell_type_matrix = df[cell_type_cols]
            
            # Save combined matrix to CSV
            df.to_csv(cache_file)
            
            time.sleep(1)  # Rate limiting
            
            # Generate plots
            # self.plot_expression_data(uniprot_id, tissue_matrix, cell_type_matrix)
            
            return tissue_matrix, cell_type_matrix
            
        except Exception as e:
            print(f"Error fetching data for {uniprot_id}: {str(e)}")
            return None, None
    
    def compute_coexpression_features(self, receptor_expr, ligand_expr):
        """Compute co-expression features for a receptor-ligand pair.
        
        Args:
            receptor_expr (tuple): (tissue_matrix, cell_type_matrix) for receptor
            ligand_expr (tuple): (tissue_matrix, cell_type_matrix) for ligand
            
        Returns:
            dict: Co-expression features
        """
        if receptor_expr is None or ligand_expr is None:
            return None
            
        # Unpack matrices
        receptor_tissue, receptor_cell = receptor_expr
        ligand_tissue, ligand_cell = ligand_expr
        
        if receptor_tissue is None or ligand_tissue is None:
            return None
            
        # Compute features for both tissue and cell type data
        tissue_features = self._compute_features(receptor_tissue, ligand_tissue)
        cell_features = self._compute_features(receptor_cell, ligand_cell)
        
        if tissue_features and cell_features:
            return {
                'tissue_' + k: v for k, v in tissue_features.items()
            } | {
                'cell_' + k: v for k, v in cell_features.items()
            }
        return None
        
    def _compute_features(self, matrix1, matrix2):
        """Helper to compute features between two expression matrices."""
        if matrix1 is None or matrix2 is None or matrix1.empty or matrix2.empty:
            return None
            
        # Get common columns
        common_cols = sorted(set(matrix1.columns) & set(matrix2.columns))
        if not common_cols:
            return None
            
        # Extract values
        vec1 = matrix1[common_cols].values.flatten()
        vec2 = matrix2[common_cols].values.flatten()
        
        # Check for zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return {
                'pearson_corr': 0.0,
                'cosine_sim': 0.0,
                'jaccard_index': 0.0,
                'l2_norm_diff': np.linalg.norm(vec1 - vec2),
                'overlap_count': 0
            }
        
        # Binary expression (threshold at median)
        vec1_binary = (vec1 > np.median(vec1)).astype(int)
        vec2_binary = (vec2 > np.median(vec2)).astype(int)
        
        # Safe computation of cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        # Safe computation of Jaccard index
        union = np.sum(vec1_binary | vec2_binary)
        jaccard = np.sum(vec1_binary & vec2_binary) / union if union > 0 else 0.0
        
        return {
            'pearson_corr': np.corrcoef(vec1, vec2)[0,1] if not np.isnan(np.corrcoef(vec1, vec2)[0,1]) else 0.0,
            'cosine_sim': cosine_sim,
            'jaccard_index': jaccard,
            'l2_norm_diff': np.linalg.norm(vec1 - vec2),
            'overlap_count': np.sum(vec1_binary & vec2_binary)
        }
    
    def process_receptor_ligand_pairs(self, pairs_df, output_file='data/coexpression_features.csv'):
        """Process all receptor-ligand pairs and save features.
        
        Args:
            pairs_df (pd.DataFrame): DataFrame with receptor_id and ligand_id columns
            output_file (str): Path to save results
        """
        # Initialize a list to store all pairs and their features
        all_pairs = []
        missing_proteins = set()
        processed_count = 0
        
        # Define feature columns
        feature_cols = [
            'tissue_pearson_corr', 'tissue_cosine_sim', 'tissue_jaccard_index', 
            'tissue_l2_norm_diff', 'tissue_overlap_count',
            'cell_pearson_corr', 'cell_cosine_sim', 'cell_jaccard_index', 
            'cell_l2_norm_diff', 'cell_overlap_count'
        ]
        
        for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Processing pairs"):
            pair_data = {
                'receptor_id': row['receptor_id'],
                'ligand_id': row['ligand_id']
            }
            
            # Initialize features as NaN
            for col in feature_cols:
                pair_data[col] = np.nan
            
            # Get expression data
            receptor_expr = self.get_expression_matrix(row['receptor_id'])
            ligand_expr = self.get_expression_matrix(row['ligand_id'])
            
            # Track missing proteins
            if receptor_expr[0] is None:
                missing_proteins.add(row['receptor_id'])
            if ligand_expr[0] is None:
                missing_proteins.add(row['ligand_id'])
            
            # Compute features if both proteins have data
            if receptor_expr[0] is not None and ligand_expr[0] is not None:
                features = self.compute_coexpression_features(receptor_expr, ligand_expr)
                if features:
                    pair_data.update(features)
            
            all_pairs.append(pair_data)
            
            processed_count += 1
            if processed_count % 100 == 0:
                logging.info(f"Processed {processed_count} pairs")
        
        # Log missing proteins
        if missing_proteins:
            logging.warning(f"Could not find expression data for {len(missing_proteins)} proteins")
            with open(output_file.replace('.csv', '_missing_proteins.txt'), 'w') as f:
                f.write('\n'.join(sorted(missing_proteins)))
        
        # Create DataFrame with all pairs
        features_df = pd.DataFrame(all_pairs)
        
        # Reorder columns to put IDs first
        id_cols = ['receptor_id', 'ligand_id']
        other_cols = [col for col in features_df.columns if col not in id_cols]
        features_df = features_df[id_cols + other_cols]
        
        # Save results
        features_df.to_csv(output_file, index=False)
        logging.info(f"Saved features for all {len(features_df)} pairs to {output_file}")
        logging.info(f"Number of pairs with complete data: {features_df.dropna().shape[0]}")
        
        return features_df

if __name__ == "__main__":
    # Example usage
    extractor = ExpressionDataExtractor()
    
    # Example with UniProt IDs
    test_pairs = pd.DataFrame({
        'receptor_id': ['P50052', 'P30411'],  # AGTR2, BKRB2
        'ligand_id': ['O15467', 'O15263']     # CCL16, DFB4A
    })
    
    # Test single gene expression and plotting
    tissue_mat, cell_mat = extractor.get_expression_matrix('P50052')
    print("\nExpression matrix shapes:")
    print(f"Tissue matrix: {tissue_mat.shape if tissue_mat is not None else 'None'}")
    print(f"Cell type matrix: {cell_mat.shape if cell_mat is not None else 'None'}")
    
    # Process all pairs
    features_df = extractor.process_receptor_ligand_pairs(test_pairs)
    print(f"\nProcessed {len(features_df) if features_df is not None else 0} pairs successfully") 