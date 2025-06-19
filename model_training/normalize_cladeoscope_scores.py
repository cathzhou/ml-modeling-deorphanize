import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def normalize_scores(scores):
    """Normalize scores to range 1-100."""
    # Convert negative scores to positive by taking absolute value
    scores = np.abs(scores)
    
    # Min-max normalization to 1-100 range
    normalized = (scores - scores.min()) / (scores.max() - scores.min()) * 99 + 1
    return normalized

def main():
    # Read the scores
    logging.info("Reading CladeOScope scores...")
    scores_df = pd.read_csv('data/CladeOScope_scores.csv')
    
    # Get the score column name
    score_col = [col for col in scores_df.columns if 'score' in col.lower()][0]
    logging.info(f"Found score column: {score_col}")
    
    # Normalize the scores
    logging.info("Normalizing scores to 1-100 range...")
    original_scores = scores_df[score_col].values
    normalized_scores = normalize_scores(original_scores)
    
    # Create new dataframe with normalized scores
    normalized_df = scores_df.copy()
    normalized_df[score_col] = normalized_scores
    
    # Save normalized scores
    output_file = 'data/CladeOScope_scores_normalized.csv'
    logging.info(f"Saving normalized scores to {output_file}")
    normalized_df.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    main() 