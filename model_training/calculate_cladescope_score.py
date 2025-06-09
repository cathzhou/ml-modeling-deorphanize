import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm

def argsort_mat(NPP_cor_mat):
    """Compute argsort matrix as in original CladeOScope implementation."""
    adj_argsort = np.zeros((NPP_cor_mat.shape[0], NPP_cor_mat.shape[0]))
    for i in range(NPP_cor_mat.shape[0]):
        inds = np.argsort(-NPP_cor_mat[i, :])
        adj_argsort[i, inds] = np.arange(NPP_cor_mat.shape[0])
    return adj_argsort

def calculate_score(gene1_id, gene2_id, npp_df, bitscore_df, clades_dict, combination="COMB5"):
    """Calculate CladeOScope score for two genes using min_rank method.
    
    Args:
        gene1_id (str): First gene UniProt ID
        gene2_id (str): Second gene UniProt ID
        npp_df (pd.DataFrame): NPP matrix
        bitscore_df (pd.DataFrame): Bitscore matrix
        clades_dict (dict): Clade definitions
        combination (str): Clade combination to use
    
    Returns:
        float: CladeOScope score
    """
    # Load clade combinations
    with open(Path("../CladeOScope/clades_combs.json"), 'r') as f:
        clades_combs = json.load(f)
    
    if combination == "COMB0":
        clades_keep = list(clades_dict.keys())
    else:
        clades_keep = clades_combs[combination]
    
    # Get indices of genes
    try:
        idx1 = npp_df.index.get_loc(gene1_id)
        idx2 = npp_df.index.get_loc(gene2_id)
    except KeyError:
        print(f"One or both genes not found: {gene1_id}, {gene2_id}")
        return None
    
    # Initialize with high value for min operation
    min_rank = 30000
    
    for clade in tqdm(clades_keep, desc="Processing clades"):
        # Get species in clade
        species = clades_dict[clade]['taxid']
        
        # Get NPP values for current clade
        tmp = npp_df.loc[:, species].values.copy()
        
        # Filter non-conserved genes based on bitscores
        conservation_mask = np.quantile(bitscore_df.loc[:, species], 0.9, axis=1) <= 40
        tmp[conservation_mask, :] = np.nan
        
        # Calculate correlation matrix
        tmp = np.corrcoef(tmp)
        
        # Convert to ranks
        tmp = argsort_mat(tmp).astype("float32") + 1
        tmp = np.sqrt(np.multiply(tmp, tmp.T))
        
        # Update minimum rank
        current_rank = tmp[idx1, idx2]
        if not np.isnan(current_rank):
            min_rank = min(min_rank, current_rank)
    
    return -min_rank if min_rank != 30000 else None

def main():
    parser = argparse.ArgumentParser(description='Calculate CladeOScope score for two genes')
    parser.add_argument('gene1', help='UniProt ID of first gene')
    parser.add_argument('gene2', help='UniProt ID of second gene')
    parser.add_argument('--clade_combo', default='COMB5', help='Clade combination to use')
    args = parser.parse_args()
    
    # Load data
    print("Loading NPP matrix...")
    npp_df = pd.read_table("../CladeOScope/9606_NPP.tsv", delimiter="\t", index_col=0)
    
    print("Loading bitscore matrix...")
    bitscore_df = pd.read_table("../CladeOScope/9606_bitscore.tsv", delimiter="\t")
    bitscore_df = bitscore_df.iloc[:, 2:]  # Remove gene info columns
    bitscore_df[bitscore_df.isna()] = 0
    
    print("Loading clade definitions...")
    with open("../CladeOScope/clades_taxid.json", 'r') as f:
        clades = json.load(f)
        # Filter clades with at least 20 species
        clades = {k: v for k, v in clades.items() if v['size'] >= 20}
    
    # Calculate score
    score = calculate_score(args.gene1, args.gene2, npp_df, bitscore_df, clades, args.clade_combo)
    
    if score is not None:
        print(f"\nCladeOScope score for {args.gene1} and {args.gene2}: {score}")
        print("Note: Lower (more negative) scores indicate stronger coevolution")
    else:
        print(f"\nCould not calculate score for {args.gene1} and {args.gene2}")

if __name__ == "__main__":
    main() 