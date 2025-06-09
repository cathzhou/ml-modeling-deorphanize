import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import numba
from numba import jit, prange

@jit(nopython=True, parallel=True)
def fast_argsort(matrix):
    """Optimized argsort matrix computation using Numba."""
    n = matrix.shape[0]
    result = np.zeros_like(matrix)
    
    for i in prange(n):
        result[i, np.argsort(-matrix[i, :])] = np.arange(n)
    return result

class FastCladeOScope:
    def __init__(self, data_dir='../CladeOScope'):
        """Initialize with pre-computed data structures for faster scoring.
        
        Args:
            data_dir (str): Path to CladeOScope data directory
        """
        self.data_dir = Path(data_dir)
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        self._load_data()
        self._precompute_structures()
        
    def _load_data(self):
        """Load necessary data files."""
        # Load NPP matrix
        self.npp_df = pd.read_table(self.data_dir / '9606_NPP.tsv', delimiter="\t", index_col=0)
        
        # Load bitscore matrix
        self.bitscore_df = pd.read_table(self.data_dir / '9606_bitscore.tsv', delimiter="\t")
        self.bitscore_df = self.bitscore_df.iloc[:, 2:]
        self.bitscore_df[self.bitscore_df.isna()] = 0
        
        # Load clade definitions
        with open(self.data_dir / 'clades_taxid.json', 'r') as f:
            self.clades = {k: v for k, v in json.load(f).items() 
                         if v['size'] >= 20}
            
        # Load clade combinations
        with open(self.data_dir / 'clades_combs.json', 'r') as f:
            self.clade_combs = json.load(f)
            
    def _precompute_structures(self):
        """Pre-compute correlation matrices and conservation masks for each clade."""
        print("Pre-computing correlation matrices...")
        self.clade_data = {}
        
        # Get COMB5 clades
        clades_keep = self.clade_combs['COMB5']
        
        for clade in tqdm(clades_keep):
            species = self.clades[clade]['taxid']
            
            # Get NPP values and create conservation mask
            npp_values = self.npp_df.loc[:, species].values
            conservation_mask = np.quantile(self.bitscore_df.loc[:, species], 0.9, axis=1) <= 40
            
            # Apply conservation mask
            npp_values[conservation_mask] = np.nan
            
            # Compute correlation matrix
            corr_matrix = np.corrcoef(npp_values)
            
            # Compute ranks
            with np.errstate(invalid='ignore'):
                ranks = fast_argsort(corr_matrix).astype(np.float32) + 1
                ranks = np.sqrt(np.multiply(ranks, ranks.T))
            
            self.clade_data[clade] = ranks
            
        # Create gene index mapping
        self.gene_indices = {gene: idx for idx, gene in enumerate(self.npp_df.index)}
        
    def calculate_score(self, gene1_id, gene2_id):
        """Calculate CladeOScope score for two genes.
        
        Args:
            gene1_id (str): First gene UniProt ID
            gene2_id (str): Second gene UniProt ID
            
        Returns:
            float: CladeOScope score (negative rank, lower is better)
        """
        try:
            idx1 = self.gene_indices[gene1_id]
            idx2 = self.gene_indices[gene2_id]
        except KeyError:
            print(f"One or both genes not found: {gene1_id}, {gene2_id}")
            return None
        
        # Find minimum rank across all clades
        min_rank = 30000
        for ranks in self.clade_data.values():
            current_rank = ranks[idx1, idx2]
            if not np.isnan(current_rank):
                min_rank = min(min_rank, current_rank)
        
        return -min_rank if min_rank != 30000 else None

def main():
    parser = argparse.ArgumentParser(description='Fast calculation of CladeOScope score')
    parser.add_argument('gene1', help='UniProt ID of first gene')
    parser.add_argument('gene2', help='UniProt ID of second gene')
    args = parser.parse_args()
    
    # Initialize analyzer (this will take some time but only once)
    analyzer = FastCladeOScope()
    
    # Calculate score (this will be very fast)
    score = analyzer.calculate_score(args.gene1, args.gene2)
    
    if score is not None:
        print(f"\nCladeOScope score for {args.gene1} and {args.gene2}: {score}")
        print("Note: Lower (more negative) scores indicate stronger coevolution")
    else:
        print(f"\nCould not calculate score for {args.gene1} and {args.gene2}")

if __name__ == "__main__":
    main() 