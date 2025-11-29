"""
Comprehensive Model Evaluation on Test Set
Includes subgroup analysis, statistical testing, and performance visualization
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import create_model
from data_preprocessing import create_data_loaders
from utils import (compute_metrics, plot_roc_curves, plot_confusion_matrices,
                  plot_metrics_comparison, save_metrics_report, load_checkpoint)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config, model_path):
        self.config = config
        self.device = config.DEVICE
        
        # Load model
        print("Loading model...")
        self.model = create_model(config, pretrained=False)
        checkpoint = load_checkpoint(model_path, self.model)
        self.model.eval()
        
        print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
        print(f"✓ Best training AUROC: {checkpoint['best_auroc']:.4f}")
    
    def evaluate_test_set(self, test_loader):
        """Evaluate model on test set"""
        print("\nEvaluating on test set...")
        
        all_predictions = []
        all_labels = []
        all_metadata = []
        inference_times = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels, metadata in tqdm(test_loader, desc='Testing'):
                images = images.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                inference_time = time.time() - start_time
                
                # Store results
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.numpy())
                all_metadata.extend(metadata)
                inference_times.append(inference_time / len(images))
        
        # Aggregate results
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        
        # Compute metrics
        metrics = compute_metrics(labels, predictions, self.config.CLASS_NAMES)
        
        # Add inference time statistics
        metrics['mean_inference_time'] = np.mean(inference_times)
        metrics['median_inference_time'] = np.median(inference_times)
        metrics['max_inference_time'] = np.max(inference_times)
        
        print(f"\n✓ Test Set Evaluation Complete")
        print(f"Mean AUROC: {metrics['mean_auroc']:.4f}")
        print(f"Mean Sensitivity: {metrics['mean_sensitivity']:.4f}")
        print(f"Mean Specificity: {metrics['mean_specificity']:.4f}")
        print(f"Mean Inference Time: {metrics['mean_inference_time']*1000:.2f}ms")
        
        return predictions, labels, metrics, all_metadata
    
    def subgroup_analysis(self, predictions, labels, metadata):
        """Perform subgroup analysis by demographics"""
        print("\nPerforming subgroup analysis...")
        
        # Convert metadata to DataFrame
        meta_df = pd.DataFrame([
            {
                'age': m['age'],
                'gender': m['gender'],
                'image_name': m['image_name']
            } for m in metadata
        ])
        
        subgroup_results = {}
        
        # Age quartile analysis
        if 'age' in meta_df.columns and meta_df['age'].notna().any():
            age_quartiles = pd.qcut(meta_df['age'].dropna(), 
                                   q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                mask = age_quartiles == quartile
                if mask.sum() > 0:
                    metrics = compute_metrics(
                        labels[mask.values], 
                        predictions[mask.values],
                        self.config.CLASS_NAMES
                    )
                    subgroup_results[f'age_{quartile}'] = metrics
        
        # Gender analysis
        if 'gender' in meta_df.columns:
            for gender in ['M', 'F']:
                mask = meta_df['gender'] == gender
                if mask.sum() > 0:
                    metrics = compute_metrics(
                        labels[mask.values],
                        predictions[mask.values],
                        self.config.CLASS_NAMES
                    )
                    subgroup_results[f'gender_{gender}'] = metrics
        
        # Save subgroup analysis
        subgroup_path = self.config.METRICS_DIR / 'subgroup_analysis.json'
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for key, value in subgroup_results.items():
            serializable_results[key] = {
                'mean_auroc': float(value['mean_auroc']),
                'mean_sensitivity': float(value['mean_sensitivity']),
                'mean_specificity': float(value['mean_specificity'])
            }
        
        with open(subgroup_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✓ Subgroup analysis saved: {subgroup_path}")
        
        return subgroup_results
    
    def plot_subgroup_comparison(self, subgroup_results, save_path):
        """Visualize subgroup analysis results"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        groups = list(subgroup_results.keys())
        auroc_scores = [subgroup_results[g]['mean_auroc'] for g in groups]
        sensitivity_scores = [subgroup_results[g]['mean_sensitivity'] for g in groups]
        specificity_scores = [subgroup_results[g]['mean_specificity'] for g in groups]
        
        # AUROC comparison
        axes[0].barh(groups, auroc_scores, color='steelblue')
        axes[0].axvline(x=0.85, color='r', linestyle='--', label='Target')
        axes[0].set_xlabel('AUROC', fontsize=12)
        axes[0].set_title('AUROC by Subgroup', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Sensitivity comparison
        axes[1].barh(groups, sensitivity_scores, color='green')
        axes[1].axvline(x=0.90, color='r', linestyle='--', label='Target')
        axes[1].set_xlabel('Sensitivity', fontsize=12)
        axes[1].set_title('Sensitivity by Subgroup', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Specificity comparison
        axes[2].barh(groups, specificity_scores, color='orange')
        axes[2].axvline(x=0.85, color='r', linestyle='--', label='Target')
        axes[2].set_xlabel('Specificity', fontsize=12)
        axes[2].set_title('Specificity by Subgroup', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Subgroup comparison plot saved: {save_path}")
    
    def analyze_cooccurrence(self, labels, predictions):
        """Analyze performance on co-occurring pathologies"""
        print("\nAnalyzing co-occurrence patterns...")
        
        # Count number of positive labels per sample
        num_positives = labels.sum(axis=1)
        
        cooccurrence_metrics = {}
        for n in range(0, 6):  # 0 to 5+ diseases
            if n < 5:
                mask = num_positives == n
                label_str = f'{n}_diseases'
            else:
                mask = num_positives >= n
                label_str = '5+_diseases'
            
            if mask.sum() > 0:
                metrics = compute_metrics(
                    labels[mask],
                    predictions[mask],
                    self.config.CLASS_NAMES
                )
                cooccurrence_metrics[label_str] = {
                    'count': int(mask.sum()),
                    'mean_auroc': float(metrics['mean_auroc']),
                    'mean_sensitivity': float(metrics['mean_sensitivity']),
                    'mean_specificity': float(metrics['mean_specificity'])
                }
        
        # Save co-occurrence analysis
        cooccur_path = self.config.METRICS_DIR / 'cooccurrence_analysis.json'
        with open(cooccur_path, 'w') as f:
            json.dump(cooccurrence_metrics, f, indent=2)
        
        print(f"✓ Co-occurrence analysis saved: {cooccur_path}")
        
        return cooccurrence_metrics
    
    def generate_all_plots(self, predictions, labels, metrics):
        """Generate all evaluation plots"""
        print("\nGenerating evaluation plots...")
        
        # ROC curves
        plot_roc_curves(
            labels, predictions, self.config.CLASS_NAMES,
            self.config.FIGURES_DIR / 'roc_curves_test.png'
        )
        
        # Confusion matrices
        plot_confusion_matrices(
            labels, predictions, self.config.CLASS_NAMES,
            self.config.FIGURES_DIR / 'confusion_matrices_test.png',
            threshold=0.5
        )
        
        # Metrics comparison
        plot_metrics_comparison(
            metrics, self.config.CLASS_NAMES,
            self.config.FIGURES_DIR / 'metrics_comparison_test.png'
        )
        
        # Save metrics report
        save_metrics_report(
            metrics, self.config.CLASS_NAMES,
            self.config.METRICS_DIR / 'test_metrics_report'
        )


def main():
    """Main evaluation function"""
    # Initialize configuration
    config = Config()
    config.create_directories()
    
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader = create_data_loaders(config)
    
    # Initialize evaluator
    model_path = config.MODELS_DIR / 'best_model.pth'
    evaluator = ModelEvaluator(config, model_path)
    
    # Evaluate on test set
    predictions, labels, metrics, metadata = evaluator.evaluate_test_set(test_loader)
    
    # Generate all plots
    evaluator.generate_all_plots(predictions, labels, metrics)
    
    # Subgroup analysis
    subgroup_results = evaluator.subgroup_analysis(predictions, labels, metadata)
    if subgroup_results:
        evaluator.plot_subgroup_comparison(
            subgroup_results,
            config.FIGURES_DIR / 'subgroup_comparison.png'
        )
    
    # Co-occurrence analysis
    cooccurrence_metrics = evaluator.analyze_cooccurrence(labels, predictions)
    
    # Print final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Set Performance:")
    print(f"  Mean AUROC:       {metrics['mean_auroc']:.4f}")
    print(f"  Mean Sensitivity: {metrics['mean_sensitivity']:.4f}")
    print(f"  Mean Specificity: {metrics['mean_specificity']:.4f}")
    print(f"  Mean F1 Score:    {metrics['mean_f1']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"\nInference Performance:")
    print(f"  Mean Time:        {metrics['mean_inference_time']*1000:.2f}ms")
    print(f"  Median Time:      {metrics['median_inference_time']*1000:.2f}ms")
    print(f"  Max Time:         {metrics['max_inference_time']*1000:.2f}ms")
    
    # Check if targets are met
    print(f"\n✓ Target AUROC (0.85): {'PASSED' if metrics['mean_auroc'] >= 0.85 else 'NOT MET'}")
    print(f"✓ Target Sensitivity (0.90): {'PASSED' if metrics['mean_sensitivity'] >= 0.90 else 'NOT MET'}")
    print(f"✓ Target Specificity (0.85): {'PASSED' if metrics['mean_specificity'] >= 0.85 else 'NOT MET'}")
    print(f"✓ Inference Time (<10s): {'PASSED' if metrics['mean_inference_time'] < 10 else 'NOT MET'}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()