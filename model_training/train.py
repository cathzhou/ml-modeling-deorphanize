import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, f1_score
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
import wandb
from datetime import datetime

from model import GPCRBindingPredictor
from data_preprocessing import DataPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize wandb
        wandb.init(
            project="gpcr-binding-prediction",
            name=f"{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
        
        # Create model
        self.model = GPCRBindingPredictor(config).to(self.device)
        wandb.watch(self.model)
        
        # Setup loss and optimizer
        self.criterion = nn.BCELoss()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['model_params']['learning_rate'],
            weight_decay=config['model_params']['weight_decay']
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path('model_training/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        """Train the model."""
        n_epochs = self.config['model_params']['max_epochs']
        patience = self.config['model_params']['early_stopping_patience']
        warmup_steps = self.config['model_params'].get('warmup_steps', 1000)
        
        # Setup learning rate scheduler
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['model_params']['learning_rate'],
            epochs=n_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=warmup_steps / (n_epochs * len(train_loader)),
            anneal_strategy='cos'
        )
        
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            for batch in train_loader:
                # Move batch to device
                features, targets = batch
                batch_inputs = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                
                # Record predictions
                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
            
            train_loss /= len(train_loader)
            
            # Validation
            valid_loss, valid_metrics = self.evaluate(valid_loader)
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_auroc': valid_metrics['auroc'],
                'valid_auprc': valid_metrics['auprc'],
                'valid_precision': valid_metrics['precision'],
                'valid_f1': valid_metrics['f1'],
                'learning_rate': scheduler.get_last_lr()[0]
            }
            wandb.log(metrics)
            
            # Early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            logging.info(f"Epoch {epoch + 1}/{n_epochs}")
            logging.info(f"Train Loss: {train_loss:.4f}")
            logging.info(f"Valid Loss: {valid_loss:.4f}")
            logging.info(f"Valid AUROC: {valid_metrics['auroc']:.4f}")
            logging.info(f"Valid AUPRC: {valid_metrics['auprc']:.4f}")
            
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                features, targets = batch
                batch_inputs = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, targets)
                
                # Record predictions
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = {
            'auroc': roc_auc_score(all_targets, all_preds),
            'auprc': average_precision_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds > 0.5),
            'f1': f1_score(all_targets, all_preds > 0.5)
        }
        
        return total_loss / len(loader), metrics
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def run_training(model_config_path: str, data_config_path: str) -> None:
    """Run training with the given configuration files.
    
    Args:
        model_config_path: Path to model configuration file
        data_config_path: Path to data configuration file
    """
    # Load configs
    with open(model_config_path) as f:
        model_config = json.load(f)
    
    # Initialize preprocessor and get datasets
    preprocessor = DataPreprocessor(data_config_path, model_config_path)
    datasets, _ = preprocessor.preprocess_data(
        split_method=model_config.get('split_method', 'random')
    )
    
    # Create data loaders
    batch_size = model_config['model_params']['batch_size']
    num_workers = model_config['data_params'].get('num_workers', 4)
    
    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        datasets['valid'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize trainer and train model
    trainer = Trainer(model_config)
    trainer.train(train_loader, valid_loader)
    
    # Evaluate on test set
    test_loss, test_metrics = trainer.evaluate(test_loader)
    logging.info("\nTest Set Results:")
    logging.info(f"Loss: {test_loss:.4f}")
    logging.info(f"AUROC: {test_metrics['auroc']:.4f}")
    logging.info(f"AUPRC: {test_metrics['auprc']:.4f}")
    logging.info(f"Precision: {test_metrics['precision']:.4f}")
    logging.info(f"F1: {test_metrics['f1']:.4f}")
    
    # Log test metrics to wandb
    wandb.log({
        'test_loss': test_loss,
        'test_auroc': test_metrics['auroc'],
        'test_auprc': test_metrics['auprc'],
        'test_precision': test_metrics['precision'],
        'test_f1': test_metrics['f1']
    })

def main():
    model_config_path = 'model_training/model_config/train_config_umap.json'
    data_config_path = 'model_training/data_config.json'
    run_training(model_config_path, data_config_path)

if __name__ == '__main__':
    main() 