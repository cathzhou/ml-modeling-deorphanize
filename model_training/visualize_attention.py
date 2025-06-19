import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import logging
import argparse
from mhsa_model import ResiduePredictor, ResidueDataset
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttentionVisualizer:
    def __init__(self, model_path, data_path, output_dir='attention_analysis'):
        """
        Initialize the attention visualizer.
        
        Args:
            model_path: Path to the trained model
            data_path: Path to the data CSV file
            output_dir: Directory to save visualizations
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.df = pd.read_csv(data_path)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        # Create datasets
        self.test_dataset = ResidueDataset(self.df, 'test')
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        
    def load_model(self):
        """Load the trained model."""
        # Get input dimension from data
        feature_cols = [col for col in self.df.columns if col not in ['code', 'known_pair', 'split']]
        input_dim = len(feature_cols)
        
        # Create model
        model = ResiduePredictor(input_dim=input_dim)
        
        # Load trained weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        logging.info(f"Model loaded from {self.model_path}")
        logging.info(f"Input dimension: {input_dim}")
        
        return model
    
    def get_attention_weights(self, code):
        """
        Get attention weights for a specific code.
        
        Args:
            code: The code to analyze
            
        Returns:
            attention_weights: Attention weights for the code
            features: Input features
            label: True label
        """
        # Find the data for this code
        code_data = self.df[self.df['code'] == code]
        if code_data.empty:
            raise ValueError(f"Code {code} not found in dataset")
        
        # Get features and label
        feature_cols = [col for col in self.df.columns if col not in ['code', 'known_pair', 'split']]
        features = torch.FloatTensor(code_data[feature_cols].values).to(self.device)
        label = code_data['known_pair'].values[0]
        
        # Get attention weights
        with torch.no_grad():
            # Forward pass to get attention weights
            _ = self.model(features)
            attention_weights = self.model.attention_weights
        
        return attention_weights, features, label
    
    def visualize_attention_for_code(self, code, save_plot=True):
        """
        Visualize attention weights for a specific code.
        
        Args:
            code: The code to visualize
            save_plot: Whether to save the plot
            
        Returns:
            fig: The matplotlib figure
        """
        try:
            attention_weights, features, label = self.get_attention_weights(code)
            
            # Check if attention weights are valid
            if attention_weights is None or attention_weights.shape[0] == 0:
                logging.error(f"No valid attention weights for code {code}")
                return None
            
            # Get feature names
            feature_cols = [col for col in self.df.columns if col not in ['code', 'known_pair', 'split']]
            
            # Create figure with subplots for each attention head
            n_heads = attention_weights.shape[0]
            
            # Ensure we have a valid number of heads
            if n_heads == 0:
                logging.error(f"No attention heads found for code {code}")
                return None
            
            # Calculate subplot layout
            if n_heads == 1:
                fig, axes = plt.subplots(1, 1, figsize=(12, 10))
                axes = [axes]
            else:
                cols = min(n_heads, 4)  # Max 4 columns
                rows = (n_heads + cols - 1) // cols  # Ceiling division
                fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
                if rows == 1:
                    axes = axes.reshape(1, -1)
                axes = axes.flatten()
            
            # Plot attention weights for each head
            for i, ax in enumerate(axes):
                if i < n_heads:
                    # Get attention weights for this head
                    head_weights = attention_weights[i].cpu().numpy()
                    
                    # Check if head_weights has valid dimensions
                    if head_weights.shape[0] == 0 or head_weights.shape[1] == 0:
                        ax.text(0.5, 0.5, f'Head {i+1}\nNo valid weights', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Head {i+1}')
                        continue
                    
                    # Ensure head_weights is 2D for heatmap
                    if head_weights.ndim == 3:
                        # If it's 3D, take the first slice
                        head_weights = head_weights[0, :, :]
                    elif head_weights.ndim > 2:
                        # For any higher dimensions, squeeze to 2D
                        head_weights = head_weights.squeeze()
                    
                    # Final check to ensure 2D
                    if head_weights.ndim != 2:
                        ax.text(0.5, 0.5, f'Head {i+1}\nInvalid shape: {head_weights.shape}', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Head {i+1}')
                        continue
                    
                    # Create heatmap with every other feature name to reduce clutter
                    # Use every other feature name for tick labels
                    every_other_features = feature_cols[::2]  # Take every other feature
                    every_other_indices = list(range(0, len(feature_cols), 2))  # Corresponding indices
                    
                    # Create heatmap with sparse tick labels
                    sns.heatmap(head_weights, ax=ax, cmap='viridis', 
                              xticklabels=False, yticklabels=False)  # Don't show all labels initially
                    ax.set_title(f'Head {i+1}')
                    ax.set_xlabel('Key Position')
                    ax.set_ylabel('Query Position')
                    
                    # Set custom tick positions and labels for every other feature
                    ax.set_xticks(every_other_indices)
                    ax.set_xticklabels(every_other_features, rotation=45, ha='right', fontsize=4)
                    ax.set_yticks(every_other_indices)
                    ax.set_yticklabels(every_other_features, rotation=0, fontsize=4)
                else:
                    # Hide unused subplots
                    ax.set_visible(False)
            
            # Add overall title
            fig.suptitle(f'Attention Weights for Code: {code}\nTrue Label: {label}', 
                        fontsize=16, y=0.95)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = os.path.join(self.output_dir, f'attention_{code}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logging.info(f"Attention plot saved to {plot_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error visualizing attention for code {code}: {str(e)}")
            return None
    
    def create_confidence_matrix(self, threshold=0.5):
        """
        Create confidence matrix showing prediction confidence vs true labels.
        
        Args:
            threshold: Classification threshold
            
        Returns:
            fig: The matplotlib figure
        """
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(features)
                confidences = outputs.cpu().numpy().flatten()
                predictions = (confidences > threshold).astype(int)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences)
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. Confidence Distribution by True Label
        confidences = np.array(all_confidences)
        labels = np.array(all_labels)
        
        axes[0,1].hist(confidences[labels == 0], alpha=0.7, label='Negative', bins=20)
        axes[0,1].hist(confidences[labels == 1], alpha=0.7, label='Positive', bins=20)
        axes[0,1].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        axes[0,1].set_title('Confidence Distribution by True Label')
        axes[0,1].set_xlabel('Confidence')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend()
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_confidences)
        roc_auc = auc(fpr, tpr)
        
        axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1,0].set_xlim([0.0, 1.0])
        axes[1,0].set_ylim([0.0, 1.05])
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].set_title('ROC Curve')
        axes[1,0].legend(loc="lower right")
        
        # 4. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(all_labels, all_confidences)
        avg_precision = average_precision_score(all_labels, all_confidences)
        
        axes[1,1].plot(recall, precision, color='blue', lw=2,
                      label=f'PR curve (AP = {avg_precision:.3f})')
        axes[1,1].set_xlim([0.0, 1.0])
        axes[1,1].set_ylim([0.0, 1.05])
        axes[1,1].set_xlabel('Recall')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].set_title('Precision-Recall Curve')
        axes[1,1].legend(loc="lower left")
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'confidence_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Confidence matrix saved to {plot_path}")
        
        return fig
    
    def test_model_performance(self):
        """
        Test model performance and print detailed metrics.
        
        Returns:
            dict: Dictionary containing performance metrics
        """
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(features)
                confidences = outputs.cpu().numpy().flatten()
                predictions = (confidences > 0.5).astype(int)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        auc_score = roc_auc_score(all_labels, all_confidences)
        
        # Add debugging information
        all_confidences = np.array(all_confidences)
        all_labels = np.array(all_labels).astype(int)
        all_predictions = np.array(all_predictions).astype(int)
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc_score:.4f}")
        print("="*50)
        
        # Debug information
        print(f"\nPrediction Distribution:")
        print(f"  Predictions (0/1): {np.bincount(all_predictions)}")
        print(f"  True Labels (0/1): {np.bincount(all_labels)}")
        print(f"  Confidence range: [{all_confidences.min():.4f}, {all_confidences.max():.4f}]")
        print(f"  Confidence mean: {all_confidences.mean():.4f}")
        print(f"  Confidence std: {all_confidences.std():.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, zero_division=0))
        
        # Save metrics to file
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score
        }
        
        metrics_path = os.path.join(self.output_dir, 'performance_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("MODEL PERFORMANCE METRICS\n")
            f.write("="*50 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
            f.write("="*50 + "\n")
            f.write(f"\nPrediction Distribution:\n")
            f.write(f"  Predictions (0/1): {np.bincount(all_predictions)}\n")
            f.write(f"  True Labels (0/1): {np.bincount(all_labels)}\n")
            f.write(f"  Confidence range: [{all_confidences.min():.4f}, {all_confidences.max():.4f}]\n")
            f.write(f"  Confidence mean: {all_confidences.mean():.4f}\n")
            f.write(f"  Confidence std: {all_confidences.std():.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(all_labels, all_predictions, zero_division=0))
        
        logging.info(f"Performance metrics saved to {metrics_path}")
        
        return metrics
    
    def analyze_feature_importance(self, n_samples=10):
        """
        Analyze feature importance using attention weights.
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            fig: The matplotlib figure
        """
        try:
            # Get attention weights for multiple samples
            all_attention_weights = []
            feature_cols = [col for col in self.df.columns if col not in ['code', 'known_pair', 'split']]
            
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(self.test_loader):
                    if i >= n_samples:
                        break
                        
                    features = batch['features'].to(self.device)
                    _ = self.model(features)
                    attention_weights = self.model.attention_weights
                    
                    # Debug: Print attention weights shape
                    logging.info(f"Sample {i+1} attention weights shape: {attention_weights.shape}")
                    
                    # Handle 4D attention weights: (batch_size, n_heads, seq_len, seq_len)
                    if attention_weights.dim() == 4:
                        # Remove batch dimension and average across heads
                        # attention_weights shape: (1, n_heads, seq_len, seq_len)
                        attention_weights = attention_weights.squeeze(0)  # Remove batch dim: (n_heads, seq_len, seq_len)
                        avg_attention = attention_weights.mean(dim=0).cpu().numpy()  # Average across heads: (seq_len, seq_len)
                    elif attention_weights.dim() == 3:
                        # Already (n_heads, seq_len, seq_len)
                        avg_attention = attention_weights.mean(dim=0).cpu().numpy()
                    else:
                        # If it's already 2D, use as is
                        avg_attention = attention_weights.cpu().numpy()
                    
                    all_attention_weights.append(avg_attention)
            
            if not all_attention_weights:
                logging.warning("No attention weights collected. Skipping feature importance analysis.")
                return None
            
            # Average across samples
            avg_attention_across_samples = np.mean(all_attention_weights, axis=0)
            
            # Debug logging for attention weights
            logging.info(f"Average attention across samples shape: {avg_attention_across_samples.shape}")
            logging.info(f"Number of feature columns: {len(feature_cols)}")
            
            # The attention weights should be (seq_len, seq_len) where seq_len = number of features
            if avg_attention_across_samples.shape[0] != len(feature_cols):
                logging.warning(f"Attention weights shape {avg_attention_across_samples.shape} doesn't match feature count {len(feature_cols)}")
                
                # Try to reshape if it's a flattened square matrix
                total_elements = avg_attention_across_samples.shape[0]
                seq_len = int(np.sqrt(total_elements))
                
                if seq_len * seq_len == total_elements:
                    logging.info(f"Reshaping attention weights from {avg_attention_across_samples.shape} to ({seq_len}, {seq_len})")
                    avg_attention_across_samples = avg_attention_across_samples.reshape(seq_len, seq_len)
                else:
                    # If we can't reshape properly, take the first n_features elements
                    logging.warning(f"Cannot reshape to square matrix. Taking first {len(feature_cols)} elements.")
                    if total_elements > len(feature_cols):
                        avg_attention_across_samples = avg_attention_across_samples[:len(feature_cols)]
                    else:
                        # Pad with zeros if we don't have enough elements
                        padded = np.zeros(len(feature_cols))
                        padded[:total_elements] = avg_attention_across_samples
                        avg_attention_across_samples = padded
            
            # Now avg_attention_across_samples should be (seq_len, seq_len) where seq_len = number of features
            # Get feature importance by averaging attention weights for each feature (row-wise)
            feature_importance = np.mean(avg_attention_across_samples, axis=0)
            
            # Debug logging
            logging.info(f"Final feature importance shape: {feature_importance.shape}")
            logging.info(f"Feature importance type: {type(feature_importance)}")
            logging.info(f"Feature importance sample values: {feature_importance[:5]}")
            
            # Ensure we don't try to get more features than available
            n_features = len(feature_importance)
            n_top_features = min(20, n_features)
            
            # Convert to numpy array and ensure it's 1D
            feature_importance = np.asarray(feature_importance).flatten()
            top_features_idx = np.argsort(feature_importance)[-n_top_features:]  # Top features
            
            logging.info(f"Top features indices: {top_features_idx}")
            logging.info(f"Top features indices type: {type(top_features_idx)}")
            
            # Create visualization
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))
            
            # 1. Feature importance plot
            try:
                # Ensure indices are within bounds and convert to list
                valid_indices = []
                for i in top_features_idx:
                    idx = int(i)
                    if 0 <= idx < len(feature_cols):
                        valid_indices.append(idx)
                
                logging.info(f"Valid indices: {valid_indices}")
                
                if len(valid_indices) == 0:
                    raise ValueError("No valid feature indices found")
                
                top_features_names = [feature_cols[i] for i in valid_indices]
                top_features_importance = feature_importance[valid_indices]
                
                logging.info(f"Top features names: {top_features_names[:3]}...")  # Show first 3
                logging.info(f"Top features importance: {top_features_importance[:3]}...")  # Show first 3
                
                axes[0].barh(range(len(top_features_names)), top_features_importance)
                axes[0].set_yticks(range(len(top_features_names)))
                axes[0].set_yticklabels(top_features_names)
                axes[0].set_xlabel('Average Attention Weight')
                axes[0].set_title(f'Top {len(valid_indices)} Most Important Features')
            except Exception as e:
                logging.error(f"Error creating feature importance plot: {str(e)}")
                logging.error(f"Feature cols length: {len(feature_cols)}")
                logging.error(f"Feature importance shape: {feature_importance.shape}")
                axes[0].text(0.5, 0.5, f'Error creating feature importance plot\n{str(e)}', 
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('Feature Importance Plot')
            
            # 2. Attention heatmap for top features
            if n_top_features > 1:  # Only create heatmap if we have multiple features
                try:
                    # Ensure indices are within bounds for attention weights
                    valid_attn_indices = []
                    for i in top_features_idx:
                        idx = int(i)
                        if 0 <= idx < avg_attention_across_samples.shape[0]:
                            valid_attn_indices.append(idx)
                    
                    logging.info(f"Valid attention indices: {valid_attn_indices}")
                    
                    if len(valid_attn_indices) > 1:
                        # Get the attention weights for the top features
                        top_attention = avg_attention_across_samples[valid_attn_indices][:, valid_attn_indices]
                        valid_feature_names = [feature_cols[i] for i in valid_attn_indices]
                        sns.heatmap(top_attention, ax=axes[1], cmap='viridis', 
                                   xticklabels=valid_feature_names, yticklabels=valid_feature_names)
                        axes[1].set_title(f'Attention Weights for Top {len(valid_attn_indices)} Features')
                        
                        # Set very small font size for tick labels
                        axes[1].tick_params(axis='both', which='major', labelsize=6)
                        
                        # Rotate x-axis labels for better readability
                        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
                        plt.setp(axes[1].get_yticklabels(), rotation=0)
                    else:
                        axes[1].text(0.5, 0.5, 'Not enough valid features for heatmap', 
                                   ha='center', va='center', transform=axes[1].transAxes)
                        axes[1].set_title('Attention Weights Heatmap')
                except Exception as e:
                    logging.error(f"Error creating attention heatmap: {str(e)}")
                    logging.error(f"Attention weights shape: {avg_attention_across_samples.shape}")
                    axes[1].text(0.5, 0.5, f'Error creating attention heatmap\n{str(e)}', 
                               ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title('Attention Weights Heatmap')
            else:
                axes[1].text(0.5, 0.5, 'Not enough features for heatmap', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Attention Weights Heatmap')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info(f"Feature importance plot saved to {plot_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error in feature importance analysis: {str(e)}")
            logging.error(f"Feature cols length: {len(feature_cols) if 'feature_cols' in locals() else 'N/A'}")
            logging.error(f"Attention weights shape: {avg_attention_across_samples.shape if 'avg_attention_across_samples' in locals() else 'N/A'}")
            return None
    
    def visualize_all_test_codes(self, max_codes=None):
        """
        Generate attention plots for every code in the test set.
        
        Args:
            max_codes: Maximum number of codes to visualize (None for all)
            
        Returns:
            int: Number of successful visualizations
        """
        # Get all test codes
        test_codes = self.df[self.df['split'] == 'test']['code'].unique()
        
        if max_codes is not None:
            test_codes = test_codes[:max_codes]
        
        logging.info(f"Generating attention plots for {len(test_codes)} test codes...")
        
        successful_visualizations = 0
        failed_codes = []
        
        for i, code in enumerate(test_codes):
            logging.info(f"Processing code {i+1}/{len(test_codes)}: {code}")
            
            try:
                fig = self.visualize_attention_for_code(code, save_plot=True)
                if fig is not None:
                    successful_visualizations += 1
                    plt.close(fig)  # Close the figure to free memory
                else:
                    failed_codes.append(code)
                    logging.warning(f"Failed to visualize attention for code: {code}")
            except Exception as e:
                failed_codes.append(code)
                logging.error(f"Error visualizing attention for code {code}: {str(e)}")
        
        # Log summary
        logging.info(f"\nVisualization Summary:")
        logging.info(f"  Total codes processed: {len(test_codes)}")
        logging.info(f"  Successful visualizations: {successful_visualizations}")
        logging.info(f"  Failed visualizations: {len(failed_codes)}")
        
        if failed_codes:
            logging.info(f"  Failed codes: {failed_codes[:10]}...")  # Show first 10 failed codes
        
        return successful_visualizations

def main():
    """Main function to run all visualizations and tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize attention weights for MHSA model')
    parser.add_argument('--all-test-codes', action='store_true', 
                       help='Generate attention plots for all test codes')
    parser.add_argument('--max-codes', type=int, default=None,
                       help='Maximum number of test codes to visualize')
    parser.add_argument('--skip-performance', action='store_true',
                       help='Skip performance testing and confidence matrix')
    parser.add_argument('--skip-feature-importance', action='store_true',
                       help='Skip feature importance analysis')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    model_path = 'models/best_model.pth'  # Model saved in models directory
    data_path = '../data/residue_test_data/df_with_splits_mhsa_test_1.csv'
    
    visualizer = AttentionVisualizer(model_path, data_path)
    
    if not args.skip_performance:
        # Test model performance
        print("Testing model performance...")
        metrics = visualizer.test_model_performance()
        
        # Create confidence matrix
        print("\nCreating confidence matrix...")
        visualizer.create_confidence_matrix()
    
    if not args.skip_feature_importance:
        # Analyze feature importance
        print("\nAnalyzing feature importance...")
        visualizer.analyze_feature_importance()
    
    if args.all_test_codes:
        # Visualize attention for all test codes
        print("\nGenerating attention plots for all test codes...")
        successful_visualizations = visualizer.visualize_all_test_codes(max_codes=args.max_codes)
        print(f"\nSuccessfully created {successful_visualizations} attention visualizations")
    else:
        # Visualize attention for a few specific codes (original behavior)
        print("\nVisualizing attention for specific codes...")
        test_codes = visualizer.df[visualizer.df['split'] == 'test']['code'].head(5).tolist()
        
        successful_visualizations = 0
        for code in test_codes:
            print(f"Visualizing attention for code: {code}")
            fig = visualizer.visualize_attention_for_code(code)
            if fig is not None:
                successful_visualizations += 1
            else:
                print(f"  Failed to visualize attention for code: {code}")
        
        print(f"\nSuccessfully created {successful_visualizations}/{len(test_codes)} attention visualizations")
    
    print(f"All visualizations saved to: {visualizer.output_dir}")

if __name__ == '__main__':
    main() 