#!/usr/bin/env python3
"""
Simple script to visualize attention for a single code.
Usage: python visualize_single_code.py <code>
"""

import sys
import os
from visualize_attention import AttentionVisualizer

def visualize_single_code(code):
    """Visualize attention for a single code."""
    # Initialize visualizer
    model_path = '../model_training/models/best_model.pth'
    data_path = '../data/residue_test_data/df_with_splits_mhsa_test_1.csv'
    
    try:
        visualizer = AttentionVisualizer(model_path, data_path)
        
        # Visualize attention for the specified code
        print(f"Visualizing attention for code: {code}")
        fig = visualizer.visualize_attention_for_code(code)
        
        if fig is not None:
            print(f"Attention visualization saved to: attention_analysis/attention_{code}.png")
            # Show the plot
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print(f"Failed to visualize attention for code: {code}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure the model file and data file exist in the correct paths.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_single_code.py <code>")
        print("Example: python visualize_single_code.py P12345")
        sys.exit(1)
    
    code = sys.argv[1]
    visualize_single_code(code)

if __name__ == '__main__':
    main() 