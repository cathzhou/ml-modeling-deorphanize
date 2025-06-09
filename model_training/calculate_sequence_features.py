import pandas as pd
import numpy as np
from peptides import Peptide
from pathlib import Path
from tqdm import tqdm
import requests
import time
from urllib.parse import quote

def get_uniprot_sequence(uniprot_id):
    """Get protein sequence from UniProt API.
    
    Args:
        uniprot_id (str): UniProt ID
        
    Returns:
        str: Full protein sequence
    """
    url = f"https://rest.uniprot.org/uniprotkb/{quote(uniprot_id)}.fasta"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse FASTA format
        lines = response.text.split('\n')
        sequence = ''.join(lines[1:])  # Skip header line
        return sequence.strip()
        
    except Exception as e:
        print(f"Error fetching sequence for {uniprot_id}: {str(e)}")
        return None

def extract_range(sequence, range_str):
    """Extract subsequence based on range string (e.g., '1-100' or '50-75').
    
    Args:
        sequence (str): Full protein sequence
        range_str (str): Range string in format 'start-end'
        
    Returns:
        str: Subsequence
    """
    try:
        start, end = map(int, range_str.split('-'))
        # Convert to 0-based indexing
        start = start - 1
        # Handle end index
        if end > len(sequence):
            print(f"Warning: Range end {end} exceeds sequence length {len(sequence)}")
            end = len(sequence)
        return sequence[start:end]
    except Exception as e:
        print(f"Error parsing range {range_str}: {str(e)}")
        return None

def calculate_sequence_features(sequence):
    """Calculate sequence-based features for a given protein sequence.
    
    Args:
        sequence (str): Amino acid sequence
        
    Returns:
        dict: Dictionary containing calculated features
    """
    # Create Peptide object
    pep = Peptide(sequence)
    
    # Get first and last 6 AA
    n_term = sequence[:6]
    c_term = sequence[-6:]
    
    # Calculate features
    features = {
        'molecular_weight': pep.molecular_weight(),
        'n_term_pi': Peptide(n_term).isoelectric_point() if len(n_term) > 0 else np.nan,
        'c_term_pi': Peptide(c_term).isoelectric_point() if len(c_term) > 0 else np.nan,
        'length': len(sequence),
        'cysteine_count': sequence.count('C'),
    }
    
    # Find distance of closest cysteine to N and C terminus
    cys_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
    if cys_positions:
        features['dist_cys_to_nterm'] = min(cys_positions)
        features['dist_cys_to_cterm'] = min(len(sequence) - 1 - pos for pos in cys_positions)
    else:
        features['dist_cys_to_nterm'] = np.nan
        features['dist_cys_to_cterm'] = np.nan
        
    return features

def main():
    # Read the data
    print("Reading input data...")
    df = pd.read_csv('data/bm_update_3_subset_use.csv')
    
    # Create cache directory for sequences
    cache_dir = Path('data/sequence_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract sequence features
    print("\nCalculating sequence features...")
    sequence_features = []
    
    # Get unique p2_ids to minimize API calls
    unique_p2_ids = df['p2_id'].unique()
    print(f"Found {len(unique_p2_ids)} unique ligand IDs")
    
    # Cache full sequences
    sequence_cache = {}
    print("\nFetching sequences from UniProt...")
    for p2_id in tqdm(unique_p2_ids):
        cache_file = cache_dir / f"{p2_id}.txt"
        
        # Try to load from cache first
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                sequence_cache[p2_id] = f.read().strip()
        else:
            # Fetch from UniProt API
            sequence = get_uniprot_sequence(p2_id)
            if sequence:
                sequence_cache[p2_id] = sequence
                # Cache for future use
                with open(cache_file, 'w') as f:
                    f.write(sequence)
            time.sleep(0.5)  # Rate limiting
    
    print("\nCalculating features for each sequence range...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            p2_id = row['p2_id']
            p2_range = row['p2_range']
            
            if p2_id not in sequence_cache:
                raise ValueError(f"No sequence found for {p2_id}")
                
            # Get full sequence and extract range
            full_sequence = sequence_cache[p2_id]
            sequence = extract_range(full_sequence, p2_range)
            
            if sequence is None or len(sequence) == 0:
                raise ValueError(f"Invalid sequence range {p2_range} for {p2_id}")
                
            # Calculate features
            features = calculate_sequence_features(sequence)
            features['index'] = idx  # Keep track of original index
            features['p2_id'] = p2_id  # Store protein ID
            features['p2_range'] = p2_range  # Store range
            
            sequence_features.append(features)
            
        except Exception as e:
            print(f"\nError processing row {idx}: {str(e)}")
            # Add NaN values for this row
            sequence_features.append({
                'index': idx,
                'p2_id': row['p2_id'] if 'p2_id' in row else None,
                'p2_range': row['p2_range'] if 'p2_range' in row else None,
                'molecular_weight': np.nan,
                'n_term_pi': np.nan,
                'c_term_pi': np.nan,
                'length': np.nan,
                'cysteine_count': np.nan,
                'dist_cys_to_nterm': np.nan,
                'dist_cys_to_cterm': np.nan
            })
    
    # Convert to DataFrame
    features_df = pd.DataFrame(sequence_features)
    features_df.set_index('index', inplace=True)
    
    # Save features
    output_file = 'data/sequence_features.csv'
    print(f"\nSaving features to {output_file}")
    features_df.to_csv(output_file)
    
    # Print summary statistics
    print("\nFeature summary statistics:")
    print(features_df.describe())

if __name__ == "__main__":
    main() 