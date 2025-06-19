import os
import json
import logging
import torch
import wandb
from torch.utils.data import DataLoader
from model import GPCRBindingPredictor
from data_preprocessing import DataPreprocessor, create_data_loaders
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, f1_score

class MultiRoundTrainer:
    """Trainer class for multiple rounds of training with different unknown pairs."""
    
    def __init__(self, model_config: Dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_config = model_config
        self.device = device
        self.model_params = model_config['model_params']
        
        # Initialize wandb
        wandb.init(
            project="gpcr-binding-prediction",
            name=f"{model_config['name']}_{wandb.run.id}",
            config=model_config
        )
        
        # Create output directory
        self.output_dir = os.path.join('model_checkpoints', model_config['name'])
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train_round(self, 
                   train_loader: DataLoader,
                   valid_loader: DataLoader,
                   round_idx: int) -> Dict[str, float]:
        """Train model for one round."""
        # Initialize model
        model = GPCRBindingPredictor(self.model_config).to(self.device)
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.model_params['learning_rate'],
            weight_decay=self.model_params['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.model_params['learning_rate'],
            epochs=self.model_params['max_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        # Training loop
        best_valid_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.model_params['max_epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in train_loader:
                features, targets = batch
                batch_inputs = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = torch.nn.BCELoss()(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(targets.cpu().numpy())
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            valid_loss = 0
            valid_preds = []
            valid_labels = []
            
            with torch.no_grad():
                for batch in valid_loader:
                    features, targets = batch
                    batch_inputs = {k: v.to(self.device) for k, v in features.items()}
                    targets = targets.to(self.device)
                    
                    outputs = model(batch_inputs)
                    loss = torch.nn.BCELoss()(outputs, targets)
                    
                    valid_loss += loss.item()
                    valid_preds.extend(outputs.cpu().numpy())
                    valid_labels.extend(targets.cpu().numpy())
            
            valid_loss /= len(valid_loader)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(train_preds, train_labels)
            valid_metrics = self._calculate_metrics(valid_preds, valid_labels)
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'valid_{k}': v for k, v in valid_metrics.items()},
                'epoch': epoch,
                'round': round_idx
            }
            wandb.log(metrics)
            
            # Early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                
                # Save best model
                checkpoint_path = os.path.join(self.output_dir, f'round_{round_idx}_best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'valid_loss': valid_loss,
                    'metrics': valid_metrics
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.model_params['early_stopping_patience']:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model for final evaluation
        model.load_state_dict(best_model_state)
        return valid_metrics
    
    def _calculate_metrics(self, preds: List[float], labels: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        preds = np.array(preds)
        labels = np.array(labels)
        
        # Convert predictions to binary using 0.5 threshold
        binary_preds = (preds >= 0.5).astype(int)
        
        return {
            'auroc': roc_auc_score(labels, preds),
            'auprc': average_precision_score(labels, preds),
            'precision': precision_score(labels, binary_preds),
            'f1': f1_score(labels, binary_preds)
        }

def run_training(model_config_path: str, data_config_path: str, n_rounds: int = 5):
    """Run multiple rounds of training."""
    # Load configurations
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    # Initialize preprocessor and get datasets for all rounds
    preprocessor = DataPreprocessor(data_config_path, model_config_path)
    round_results = preprocessor.preprocess_data_multiple_rounds(n_rounds=n_rounds)
    
    # Initialize trainer
    trainer = MultiRoundTrainer(model_config)
    
    # Train each round
    round_metrics = []
    for round_idx, (datasets, scalers) in enumerate(round_results):
        logging.info(f"\nStarting training round {round_idx + 1}/{len(round_results)}")
        
        # Create data loaders
        train_loader = create_data_loaders(
            datasets,
            batch_size=model_config['model_params']['batch_size'],
            num_workers=model_config['data_params']['num_workers']
        )['train']
        
        valid_loader = create_data_loaders(
            datasets,
            batch_size=model_config['model_params']['batch_size'],
            num_workers=model_config['data_params']['num_workers']
        )['valid']
        
        # Train round
        metrics = trainer.train_round(train_loader, valid_loader, round_idx)
        round_metrics.append(metrics)
        
        # Log round summary
        logging.info(f"\nRound {round_idx + 1} metrics:")
        for metric_name, value in metrics.items():
            logging.info(f"  {metric_name}: {value:.4f}")
    
    # Calculate and log average metrics across rounds
    avg_metrics = {}
    for metric in round_metrics[0].keys():
        values = [m[metric] for m in round_metrics]
        avg_metrics[f'avg_{metric}'] = np.mean(values)
        avg_metrics[f'std_{metric}'] = np.std(values)
    
    logging.info("\nAverage metrics across all rounds:")
    for metric_name, value in avg_metrics.items():
        logging.info(f"  {metric_name}: {value:.4f}")
    
    wandb.log(avg_metrics)
    wandb.finish()

def main():
    model_config_path = 'model_training/model_config/train_config_umap.json'
    data_config_path = 'model_training/model_config/data_config.json'
    run_training(model_config_path, data_config_path)

if __name__ == '__main__':
    main() 