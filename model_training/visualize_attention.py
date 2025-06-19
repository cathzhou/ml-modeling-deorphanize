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
            
            # Get feature names
            feature_cols = [col for col in self.df.columns if col not in ['code', 'known_pair', 'split']]
            
            # Create figure with subplots for each attention head
            n_heads = attention_weights.shape[0]
            fig, axes = plt.subplots(2, n_heads//2, figsize=(5*n_heads//2, 10))
            if n_heads == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Plot attention weights for each head
            for i, ax in enumerate(axes):
                if i < n_heads:
                    # Get attention weights for this head
                    head_weights = attention_weights[i].cpu().numpy()
                    
                    # Create heatmap
                    sns.heatmap(head_weights, ax=ax, cmap='viridis', 
                              xticklabels=False, yticklabels=False)
                    ax.set_title(f'Head {i+1}')
                    ax.set_xlabel('Key Position')
                    ax.set_ylabel('Query Position')
            
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
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc_score = roc_auc_score(all_labels, all_confidences)
        
        # Print results
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc_score:.4f}")
        print("="*50)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))
        
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
            f.write("\nClassification Report:\n")
            f.write(classification_report(all_labels, all_predictions))
        
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
                
                # Average attention weights across heads
                avg_attention = attention_weights.mean(dim=0).cpu().numpy()
                all_attention_weights.append(avg_attention)
        
        # Average across samples
        avg_attention_across_samples = np.mean(all_attention_weights, axis=0)
        
        # Get top features by attention weight
        feature_importance = np.mean(avg_attention_across_samples, axis=0)
        top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # 1. Feature importance plot
        top_features_names = [feature_cols[i] for i in top_features_idx]
        top_features_importance = feature_importance[top_features_idx]
        
        axes[0].barh(range(len(top_features_names)), top_features_importance)
        axes[0].set_yticks(range(len(top_features_names)))
        axes[0].set_yticklabels(top_features_names)
        axes[0].set_xlabel('Average Attention Weight')
        axes[0].set_title('Top 20 Most Important Features')
        
        # 2. Attention heatmap for top features
        top_attention = avg_attention_across_samples[:, top_features_idx][top_features_idx, :]
        sns.heatmap(top_attention, ax=axes[1], cmap='viridis', 
                   xticklabels=top_features_names, yticklabels=top_features_names)
        axes[1].set_title('Attention Weights for Top 20 Features')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Feature importance plot saved to {plot_path}")
        
        return fig

def main():
    """Main function to run all visualizations and tests."""
    # Initialize visualizer
    model_path = 'models/best_model.pth'  # Model saved in models directory
    data_path = '../data/residue_test_data/df_with_splits_mhsa_test.csv'
    
    visualizer = AttentionVisualizer(model_path, data_path)
    
    # Test model performance
    print("Testing model performance...")
    metrics = visualizer.test_model_performance()
    
    # Create confidence matrix
    print("\nCreating confidence matrix...")
    visualizer.create_confidence_matrix()
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    visualizer.analyze_feature_importance()
    
    # Visualize attention for a few specific codes
    print("\nVisualizing attention for specific codes...")
    test_codes = visualizer.df[visualizer.df['split'] == 'test']['code'].head(5).tolist()
    
    for code in test_codes:
        print(f"Visualizing attention for code: {code}")
        visualizer.visualize_attention_for_code(code)
    
    print(f"\nAll visualizations saved to: {visualizer.output_dir}")

if __name__ == '__main__':
    main() 