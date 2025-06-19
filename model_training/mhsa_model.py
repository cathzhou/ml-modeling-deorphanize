import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging
import time
import os
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def my_scaled_dot_product_attention(query, key=None, value=None):
    key = key if key is not None else query
    value = value if value is not None else query
    assert query.size(-1) == key.size(-1)

    dk = key.size(-1)
    qk = query @ key.transpose(-1, -2) / dk**0.5
    attn_weights = torch.softmax(qk, dim=-1)
    attn = attn_weights @ value
    return attn, attn_weights

class ResidueDataset(Dataset):
    """Dataset for residue contact data."""
    
    def __init__(self, df, split):
        """
        Args:
            df: DataFrame containing the data
            split: 'train', 'valid', or 'test'
        """
        # Filter data for the specified split
        self.data = df[df['split'] == split].copy()
        
        # Store codes for later retrieval
        self.codes = self.data['code'].values
        
        # Get features (all columns except code, known_pair, and split)
        feature_cols = [col for col in self.data.columns if col not in ['code', 'known_pair', 'split']]
        self.features = torch.FloatTensor(self.data[feature_cols].values)
        
        # Get labels
        self.labels = torch.FloatTensor(self.data['known_pair'].values)
        
        logging.info(f"Created {split} dataset with {len(self.data)} samples")
        logging.info(f"Feature shape: {self.features.shape}")
        logging.info(f"Number of positive samples: {self.labels.sum().item()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'code': self.codes[idx]
        }

class MyEfficientMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_embed_dim = self.embed_dim // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # Reshape for multi-head attention
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Scaled dot-product attention
        attn, attn_weights = my_scaled_dot_product_attention(q, k, v)
        
        # Reshape and project back
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch_size, -1, self.embed_dim)
        output = self.projection(attn)
        
        return output, attn_weights
    
    def split_heads(self, x):
        batch_size = x.size(0)
        temp = x.view(batch_size, -1, self.n_heads, self.head_embed_dim)
        return temp.transpose(1, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # Ensure we only use the required sequence length
        return x + self.pe[:, :x.size(1), :]

class ResiduePredictor(nn.Module):
    def __init__(self, input_dim, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Input projection - project each feature to d_model dimensions
        # This maintains the sequence length (input_dim) but changes feature dimension to d_model
        self.input_proj = nn.Linear(1, d_model)  # Project each feature individually
        
        # Positional encoding with max_len matching the original sequence length
        self.pos_encoding = PositionalEncoding(d_model, max_len=input_dim)
        
        # Multi-head self-attention
        self.attention = MyEfficientMultiHeadAttention(d_model, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Store attention weights for visualization
        self.attention_weights = None
        
    def forward(self, x):
        # Input shape: [batch_size, num_features]
        # We need to reshape to [batch_size, num_features, 1] for the linear projection
        x = x.unsqueeze(-1)  # Add feature dimension: [batch_size, num_features, 1]
        
        # Project each feature to d_model dimensions
        x = self.input_proj(x)  # [batch_size, num_features, d_model]
        
        x = self.pos_encoding(x)
        
        # Self-attention block
        attn_output, attn_weights = self.attention(x)
        self.attention_weights = attn_weights
        x = self.norm1(x + attn_output)
        
        # Feed-forward block
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

def visualize_attention_weights(model, batch, epoch, output_dir='attention_plots'):
    """Visualize attention weights for a batch of data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get attention weights for the first sample in the batch
    attn_weights = model.attention_weights[0].cpu().detach().numpy()  # Shape: (num_heads, seq_len, seq_len)
    
    # Create a figure with subplots for each attention head
    n_heads = attn_weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(5*n_heads, 5))
    if n_heads == 1:
        axes = [axes]
    
    # Plot attention weights for each head
    for i, ax in enumerate(axes):
        sns.heatmap(attn_weights[i], ax=ax, cmap='viridis')
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'attention_weights_epoch_{epoch}.png'))
    plt.close()

def train_model(model, train_loader, valid_loader, num_epochs=10, learning_rate=1e-4):
    """Train the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    
    train_start = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch['label'].size(0)
            
            # Visualize attention weights for the first batch of each epoch
            if batch_idx == 0:
                visualize_attention_weights(model, batch, epoch)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        valid_accuracy = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels.unsqueeze(1))
                valid_loss += loss.item() * batch['label'].size(0)
                valid_accuracy += (outputs > 0.5).float().eq(labels.unsqueeze(1)).sum()
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        valid_loss /= len(valid_loader.dataset)
        valid_accuracy /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds > 0.5)
        
        # Log progress
        epoch_duration = time.time() - epoch_start
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        logging.info(f'Training Loss: {train_loss:.4f}')
        logging.info(f'Validation Loss: {valid_loss:.4f}')
        logging.info(f'Validation Accuracy: {valid_accuracy:.4f}')
        logging.info(f'Validation AUC: {auc:.4f}')
        logging.info(f'Validation F1: {f1:.4f}')
        logging.info(f'Epoch Duration: {epoch_duration:.1f} seconds')
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
    
    train_duration = time.time() - train_start
    logging.info(f"Training finished. Took {train_duration:.1f} seconds")
    
    return train_losses, valid_losses

def plot_losses(loss_df):
    """Plot training and validation losses."""
    # Get unique models
    models = loss_df['model'].unique()
    n_models = len(models)
    
    # Create subplots
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 5*n_models))
    if n_models == 1:
        axes = [axes]
    
    # Plot for each model
    for ax, model in zip(axes, models):
        model_data = loss_df[loss_df['model'] == model]
        
        # Plot train and test losses
        ax.plot(model_data['epoch'], model_data['train_loss'], 'b-', label='Train Loss', marker='o', markersize=3)
        ax.plot(model_data['epoch'], model_data['test_loss'], 'r-', label='Validation Loss', marker='o', markersize=3)
        
        # Customize subplot
        ax.set_title(f'Model: {model}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.suptitle('Training and Validation Losses', y=1.02, fontsize=14)
    
    return fig

def main():
    # Load data
    df = pd.read_csv('../data/residue_test_data/df_with_splits_mhsa_test_1.csv')
    
    # Create datasets
    train_dataset = ResidueDataset(df, 'train')
    valid_dataset = ResidueDataset(df, 'valid')
    test_dataset = ResidueDataset(df, 'test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    input_dim = train_dataset.features.shape[1]
    model = ResiduePredictor(input_dim)
    
    # Train model
    train_losses, valid_losses = train_model(model, train_loader, valid_loader)
    
    # Plot losses
    loss_df = pd.DataFrame({
        'epoch': range(len(train_losses)),
        'train_loss': train_losses,
        'test_loss': valid_losses,
        'model': 'ResiduePredictor'
    })
    fig = plot_losses(loss_df)
    plt.savefig('models/loss_plot.png')
    plt.close()

if __name__ == '__main__':
    main() 
# %%
