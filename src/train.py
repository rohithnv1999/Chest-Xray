"""
Training Pipeline for Chest X-Ray Multi-Label Classification
Implements training loop, validation, checkpointing, and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config
from model import create_model, count_parameters
from data_preprocessing import create_data_loaders
from utils import EarlyStopping, save_checkpoint, load_checkpoint, compute_metrics


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for class imbalance"""
    
    def __init__(self, class_weights):
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights
    
    def forward(self, outputs, targets):
        # Apply sigmoid to outputs
        outputs = torch.sigmoid(outputs)
        
        # Compute BCE with weights
        bce = -(self.class_weights * targets * torch.log(outputs + 1e-7) +
                (1 - targets) * torch.log(1 - outputs + 1e-7))
        
        return bce.mean()


class Trainer:
    """Main trainer class"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Initialize model
        print("Initializing model...")
        self.model = create_model(config, pretrained=True)
        count_parameters(self.model)
        
        # Load class weights
        with open(config.SPLITS_DIR / 'class_weights.json', 'r') as f:
            class_weights_dict = json.load(f)
        
        # Convert to tensor
        class_weights = torch.tensor([
            class_weights_dict[label]['weight'] 
            for label in config.CLASS_NAMES
        ], dtype=torch.float32).to(self.device)
        
        # Initialize loss function
        self.criterion = WeightedBCELoss(class_weights)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2),
            eps=config.EPSILON,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.T_MAX,
            eta_min=config.MIN_LEARNING_RATE
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.USE_MIXED_PRECISION else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            delta=config.EARLY_STOPPING_DELTA
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.TENSORBOARD_LOG_DIR)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auroc': [],
            'learning_rates': []
        }
        
        self.best_auroc = 0.0
        self.start_epoch = 0
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if batch_idx % self.config.LOG_INTERVAL == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        epoch_loss = running_loss / len(train_loader)
        return epoch_loss
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Store predictions and labels
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                running_loss += loss.item()
        
        # Compute metrics
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        metrics = compute_metrics(all_labels, all_predictions, self.config.CLASS_NAMES)
        
        epoch_loss = running_loss / len(val_loader)
        mean_auroc = metrics['mean_auroc']
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Val/AUROC', mean_auroc, epoch)
        
        for i, class_name in enumerate(self.config.CLASS_NAMES):
            self.writer.add_scalar(f'Val/AUROC_{class_name}', 
                                 metrics['class_auroc'][i], epoch)
        
        return epoch_loss, mean_auroc, metrics
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_auroc, metrics = self.validate(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auroc'].append(val_auroc)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val AUROC: {val_auroc:.4f} | Time: {epoch_time:.2f}s")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_auroc > self.best_auroc
            if is_best:
                self.best_auroc = val_auroc
                print(f"✓ New best AUROC: {self.best_auroc:.4f}")
            
            if (epoch + 1) % self.config.SAVE_CHECKPOINT_EVERY == 0 or is_best:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_auroc': self.best_auroc,
                    'history': self.history,
                    'config': vars(self.config)
                }
                
                save_path = self.config.CHECKPOINTS_DIR / f'checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(checkpoint, save_path, is_best, 
                              self.config.MODELS_DIR)
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
                break
            
            print("-" * 60)
        
        # Save final history
        history_path = self.config.RESULTS_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        print("\n✓ Training completed!")
        print(f"Best AUROC: {self.best_auroc:.4f}")


def main():
    """Main training function"""
    # Initialize configuration
    config = Config()
    config.create_directories()
    config.print_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()