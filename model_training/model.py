import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Multi-head self-attention module.
        
        Args:
            d_model (int): The dimension of the model.
            n_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the multi-head self-attention module.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): The mask tensor of shape (batch_size, seq_len).
        """
        batch_size = x.size(0)
        
        # Linear projections into Q, K, V matrices and reshape
        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention - attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(out)

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Positional encoding for the transformer.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the sequence.
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model) # positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the positional encoding.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
        """
        return x + self.pe[:, :x.size(1)]

class AttentionPooling(nn.Module):
    """Attention pooling to create fixed-size representation."""
    
    def __init__(self, d_model: int):
        """
        Attention pooling to create fixed-size representation.
        
        Args:
            d_model (int): The dimension of the model.
        """
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the attention pooling.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): The mask tensor of shape (batch_size, seq_len).
        """
        # x shape: [batch_size, seq_len, d_model]
        scores = self.attention(x)  # [batch_size, seq_len, 1]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=1)
        weighted_sum = torch.bmm(x.transpose(1, 2), weights).squeeze(-1)
        return weighted_sum

class ContextEncoder(nn.Module):
    """Encodes various context features."""
    
    def __init__(self, config: Dict):
        """
        Encodes various context features.
        
        Args:
            config (Dict): The data configuration dictionary.
        """
        super().__init__()
        
        # Get dimensions from config
        self.d_model = config['model_params']['d_model']
        
        # Get feature dimensions from the input features if available
        # Otherwise use reasonable defaults based on the dataset
        self.n_distance_features = len(config.get('distance_metric_columns', [])) or 10  # Default: 10 distance features
        self.n_ligand_features = len(config.get('ligand_metadata_columns', [])) or 15    # Default: 15 ligand features
        self.n_alphafold_features = len(config.get('alphafold_metric_columns', [])) or 8 # Default: 8 AlphaFold features
        self.n_expression_features = len(config.get('expression_feature_columns', [])) or 12 # Default: 12 expression features
        
        # Individual embeddings for each categorical feature
        self.categorical_columns = config.get('categorical_columns', [])
        self.categorical_embeddings = nn.ModuleDict({
            col: nn.Linear(1, self.d_model // (4 * max(len(self.categorical_columns), 1)))
            for col in self.categorical_columns
        })
        
        # Project each feature group 
        self.distance_proj = nn.Linear(self.n_distance_features, self.d_model // 4)
        self.ligand_proj = nn.Linear(self.n_ligand_features, self.d_model // 4)
        self.alphafold_proj = nn.Linear(self.n_alphafold_features, self.d_model // 4)
        self.expression_proj = nn.Linear(self.n_expression_features, self.d_model // 4)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Project each feature group
        distance_enc = self.distance_proj(features['distance_metrics'])
        ligand_enc = self.ligand_proj(features['ligand_metadata'])
        alphafold_enc = self.alphafold_proj(features['alphafold_metrics'])
        expression_enc = self.expression_proj(features['expression_features'])
        
        # Process each categorical feature individually
        categorical_encs = []
        for col in self.categorical_columns:
            cat_feature = features[f'{col}_encoded'].unsqueeze(-1).float()  # Add feature dimension
            cat_enc = self.categorical_embeddings[col](cat_feature)
            categorical_encs.append(cat_enc)
        
        # Concatenate all features
        context = torch.cat([
            distance_enc,
            ligand_enc,
            alphafold_enc,
            expression_enc,
            *categorical_encs
        ], dim=-1)
        
        return self.final_proj(context)

class GPCRBindingPredictor(nn.Module):
    """Main model for GPCR-Ligand binding prediction."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Model dimensions
        self.d_model = config['model_params']['d_model']
        self.n_heads = config['model_params']['n_heads']
        self.n_layers = config['model_params']['n_layers']
        self.d_ff = config['model_params']['d_ff']
        self.dropout = config['model_params']['dropout']
        
        # Embeddings and positional encoding
        self.residue_embedding = nn.Linear(1, self.d_model)  # For contact vector
        self.pos_embedding = PositionalEncoding(self.d_model)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(self.d_model, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Layer normalization and feedforward
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.n_layers)
        ])
        
        self.feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.d_model)
            )
            for _ in range(self.n_layers)
        ])
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(self.d_model)
        
        # Context encoder
        self.context_encoder = ContextEncoder(config)
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process residue contacts through attention
        contacts = batch['residue_contacts'].unsqueeze(-1)  # Add feature dimension
        x = self.residue_embedding(contacts)
        x = self.pos_embedding(x)
        
        # Multi-head attention layers
        for attention, norm, ff in zip(self.attention_layers, self.layer_norms, self.feedforward):
            # Self attention
            attended = attention(x)
            x = norm(x + attended)  # Add & Norm
            
            # Feedforward
            ff_out = ff(x)
            x = norm(x + ff_out)  # Add & Norm
        
        # Pool attention outputs
        attention_features = self.attention_pooling(x)
        
        # Process context features
        context_features = self.context_encoder(batch)
        
        # Combine features
        combined = torch.cat([attention_features, context_features], dim=-1)
        
        # Final prediction
        return self.final_layers(combined) 