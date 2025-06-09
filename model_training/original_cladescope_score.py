import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import joblib
import tempfile

def argsort_mat(NPP_cor_mat):
    """Original argsort matrix implementation from CladeOScope."""
    adj_argsort = np.zeros((NPP_cor_mat.shape[0], NPP_cor_mat.shape[0]))
    for i in range(NPP_cor_mat.shape[0]):
        inds = np.argsort(-NPP_cor_mat[i, :])
        adj_argsort[i, inds] = np.arange(NPP_cor_mat.shape[0])
    return adj_argsort

def calculate_CladeOScope(NPP, BS_mat, clades_dict, clades_combs, calc="min_rank", combination="COMB5"):
    """Original CladeOScope calculation from Generate_scores.py."""
    if combination == "COMB0":
        clades_keep = list(clades_dict.keys())
    else:
        clades_keep = clades_combs[combination]

    N = NPP.shape[0]

    calc = calc.lower()
    if calc == "min_rank":
        vals = np.full((N, N), 30000, dtype="float32")
    elif calc == "max_cor":
        vals = np.full((N, N), np.nan, dtype="float32")
    elif calc == "cummul_rank_log":
        vals = np.zeros((N, N), dtype="float32")
    elif calc == "npp_rank":
        clades_keep = ["Eukaryota"]

    for x in tqdm(clades_keep, desc="Processing clades"):
        # calc cor
        tmp = NPP.loc[:, clades_dict[x]].values.copy()
        tmp[np.quantile(BS_mat.loc[:, clades_dict[x]], 0.9, axis=1) <= 40, :] = np.nan
        tmp = np.corrcoef(tmp)
        
        if calc == "max_cor":
            vals = np.fmax(vals, tmp)
        else:
            # calc ranks
            tmp = argsort_mat(tmp).astype("float32") + 1
            tmp = np.sqrt(np.multiply(tmp, tmp.T))

            if calc == "npp_rank" and x == "Eukaryota":
                vals = tmp.copy()
            if calc == "min_rank":
                mask = tmp < vals
                vals[mask] = tmp[mask]
            elif calc == "cummul_rank_log":
                vals += np.log2(tmp) / len(clades_keep)
        del tmp

    # Extract upper triangle values
    vals = vals[np.triu_indices_from(vals, 1)]
    if calc == "max_cor":
        vals[np.isnan(vals)] = -1
    else:
        vals = -vals
        
    return vals

class OriginalCladeOScope:
    def __init__(self, data_dir='../CladeOScope'):
        """Initialize with original CladeOScope implementation.
        
        Args:
            data_dir (str): Path to CladeOScope data directory
        """
        self.data_dir = Path(data_dir)
        print("Loading data...")
        self._load_data()
        
    def _load_data(self):
        """Load necessary data files."""
        # Load NPP matrix
        self.npp = pd.read_table(self.data_dir / '9606_NPP.tsv', delimiter="\t", index_col=0)
        
        # Load bitscore matrix
        self.bitscore = pd.read_table(self.data_dir / '9606_bitscore.tsv', delimiter="\t")
        self.bitscore = self.bitscore.iloc[:, 2:]
        self.bitscore[self.bitscore.isna()] = 0
        
        # Load clade definitions
        with open(self.data_dir / 'clades_taxid.json', 'r') as f:
            self.clades = json.load(f)
            self.clades_dict = {k: v['taxid'] for k, v in self.clades.items() 
                              if v['size'] >= 20}
            
        # Load clade combinations
        with open(self.data_dir / 'clades_combs.json', 'r') as f:
            self.clades_combs = json.load(f)
            
    def calculate_score(self, gene1_id, gene2_id):
        """Calculate CladeOScope score for two genes using original implementation.
        
        Args:
            gene1_id (str): First gene UniProt ID
            gene2_id (str): Second gene UniProt ID
            
        Returns:
            float: CladeOScope score (negative rank, lower is better)
        """
        try:
            idx1 = self.npp.index.get_loc(gene1_id)
            idx2 = self.npp.index.get_loc(gene2_id)
        except KeyError:
            print(f"One or both genes not found: {gene1_id}, {gene2_id}")
            return None
            
        # Calculate full matrix using original implementation
        vals = calculate_CladeOScope(
            self.npp, 
            self.bitscore,
            self.clades_dict,
            self.clades_combs,
            calc="min_rank",
            combination="COMB5"
        )
        
        # Get score from upper triangle
        n = len(self.npp)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        score_idx = (idx2 * (idx2 - 1)) // 2 + idx1
        
        return vals[score_idx]

def main():
    parser = argparse.ArgumentParser(description='Original CladeOScope score calculation')
    parser.add_argument('gene1', help='UniProt ID of first gene')
    parser.add_argument('gene2', help='UniProt ID of second gene')
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = OriginalCladeOScope()
    
    # Calculate score
    score = analyzer.calculate_score(args.gene1, args.gene2)
    
    if score is not None:
        print(f"\nOriginal CladeOScope score for {args.gene1} and {args.gene2}: {score}")
        print("Note: Lower (more negative) scores indicate stronger coevolution")
    else:
        print(f"\nCould not calculate score for {args.gene1} and {args.gene2}")

if __name__ == "__main__":
    main() 