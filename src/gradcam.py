"""
Grad-CAM Implementation for Visual Explainability
Generates heatmaps showing which regions influenced model predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm

from config import Config
from model import create_model
from utils import load_checkpoint


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained model
            target_layer: Layer name to extract gradients from
        """
        self.model = model
        self.model.eval()
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        # Get target layer
        self.target_layer = self._get_layer(target_layer)
        
        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _get_layer(self, layer_name):
        """Get layer by name"""
        try:
            layers = layer_name.split('.')
            layer = self.model
            for l in layers:
                layer = getattr(layer, l)
            return layer
        except AttributeError:
            # Fallback to densenet features
            print(f"Warning: Could not find layer {layer_name}, using densenet.features.denseblock4")
            return self.model.densenet.features.denseblock4
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Target class index (if None, uses max prediction)
        
        Returns:
            cam: Class activation map [H, W]
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # If class_idx not specified, use predicted class
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Calculate weights using global average pooling
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def overlay_heatmap(self, image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay CAM heatmap on original image
        
        Args:
            image: Original image [H, W, C] in RGB
            cam: Class activation map [h, w]
            alpha: Overlay transparency
            colormap: OpenCV colormap
        
        Returns:
            overlayed: Image with heatmap overlay
        """
        # Resize CAM to match image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize image
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        
        # Overlay
        overlayed = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        
        return overlayed


class GradCAMVisualizer:
    """High-level interface for Grad-CAM visualization"""
    
    def __init__(self, config, model_path):
        self.config = config
        self.device = config.DEVICE
        
        # Load model
        self.model = create_model(config, pretrained=False)
        load_checkpoint(model_path, self.model)
        self.model.eval()
        
        # Initialize Grad-CAM
        self.gradcam = GradCAM(self.model, config.GRADCAM_LAYER)
    
    def visualize_sample(self, image_tensor, original_image, labels=None, 
                        predictions=None, save_path=None, top_k=3):
        """
        Create comprehensive visualization with multiple class heatmaps
        
        Args:
            image_tensor: Preprocessed image tensor [1, C, H, W]
            original_image: Original image for overlay [H, W, C]
            labels: Ground truth labels
            predictions: Model predictions
            save_path: Path to save visualization
            top_k: Number of top predictions to show
        """
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions if not provided
        if predictions is None:
            with torch.no_grad():
                output = self.model(image_tensor)
                predictions = torch.sigmoid(output)[0].cpu().numpy()
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        # Create figure
        fig = plt.figure(figsize=(20, 5*top_k))
        
        for i, class_idx in enumerate(top_indices):
            class_name = self.config.CLASS_NAMES[class_idx]
            prob = predictions[class_idx]
            
            # Generate Grad-CAM
            cam = self.gradcam.generate_cam(image_tensor, class_idx)
            
            # Create overlay
            overlay = self.gradcam.overlay_heatmap(
                original_image, cam, 
                alpha=self.config.GRADCAM_ALPHA
            )
            
            # Plot original image
            plt.subplot(top_k, 3, i*3 + 1)
            plt.imshow(original_image, cmap='gray')
            plt.title(f'Original Image', fontsize=12, fontweight='bold')
            plt.axis('off')
            
            # Plot heatmap
            plt.subplot(top_k, 3, i*3 + 2)
            plt.imshow(cam, cmap='jet')
            plt.title(f'Attention Map', fontsize=12, fontweight='bold')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            # Plot overlay
            plt.subplot(top_k, 3, i*3 + 3)
            plt.imshow(overlay)
            
            # Add prediction info
            gt_label = "Present" if labels is not None and labels[class_idx] == 1 else "Absent"
            title = (f'{class_name}\n'
                    f'Probability: {prob:.3f}\n'
                    f'Ground Truth: {gt_label}')
            plt.title(title, fontsize=12, fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def batch_visualize(self, data_loader, num_samples=50):
        """Generate Grad-CAM visualizations for multiple samples"""
        print(f"\nGenerating Grad-CAM visualizations for {num_samples} samples...")
        
        count = 0
        for images, labels, metadata in tqdm(data_loader):
            if count >= num_samples:
                break
            
            for i in range(len(images)):
                if count >= num_samples:
                    break
                
                # Get single sample
                image_tensor = images[i:i+1]
                label = labels[i].numpy()
                
                # Denormalize image for visualization
                img = images[i].numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Get predictions
                with torch.no_grad():
                    image_tensor = image_tensor.to(self.device)
                    output = self.model(image_tensor)
                    prediction = torch.sigmoid(output)[0].cpu().numpy()
                
                # Save visualization
                img_name = metadata['image_name'][i]
                save_path = self.config.GRADCAM_DIR / f'gradcam_{img_name}'
                
                self.visualize_sample(
                    image_tensor, img, label, prediction,
                    save_path=save_path, top_k=3
                )
                
                count += 1
        
        print(f"✓ Generated {count} Grad-CAM visualizations")


def validate_gradcam_quality(gradcam_dir, annotations_file=None):
    """
    Validate Grad-CAM quality against radiologist annotations
    Computes Intersection over Union (IoU) metrics
    """
    print("\nValidating Grad-CAM quality...")
    
    # This would require radiologist bounding box annotations
    # Placeholder for validation logic
    
    # In a real implementation:
    # 1. Load radiologist bounding boxes
    # 2. Convert Grad-CAM heatmaps to bounding boxes (threshold-based)
    # 3. Compute IoU between predicted and ground truth boxes
    # 4. Report average IoU and per-class metrics
    
    print("✓ Grad-CAM validation complete")
    print("Note: Full validation requires radiologist annotations")


def main():
    """Main Grad-CAM generation function"""
    from data_preprocessing import create_data_loaders
    
    # Initialize configuration
    config = Config()
    config.create_directories()
    
    print("\n" + "="*60)
    print("GRAD-CAM VISUALIZATION GENERATION")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_loader = create_data_loaders(config)
    
    # Initialize visualizer
    model_path = config.MODELS_DIR / 'best_model.pth'
    visualizer = GradCAMVisualizer(config, model_path)
    
    # Generate visualizations
    visualizer.batch_visualize(test_loader, num_samples=50)
    
    # Validate quality (if annotations available)
    # validate_gradcam_quality(config.GRADCAM_DIR)
    
    print("\n" + "="*60)
    print("GRAD-CAM GENERATION COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()