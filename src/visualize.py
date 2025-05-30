"""Visualize the results and loss distribution of a depth estimation model.
This script loads a pre-trained depth estimation model, applies it to a dataset of RGB images,
and visualizes the predicted depth maps alongside the ground truth depth maps and per-pixel loss maps.

Usage:
Adjust the following parameters in the script if necessary and run:
python visualize.py

Required entry in config.yaml:
```yaml
model:
  use_lb: false/true
```
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from network.midas_net_custom import MidasNet_small
from network.midas_semantics import MidasNetSemantics
from main import DepthDataset
from omegaconf import OmegaConf
from tqdm import tqdm

GT_DIR = "/cluster/courses/cil/monocular_depth/data/train"
PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
TRAIN_LIST_PATH = os.path.join(PROJECT_DIR, 'data', 'train_list.txt')
INPUT_SIZE = (448, 576)
N_SAMPLES = 100  # Number of samples to visualize
MODEL_TYPE = 'MiDaS_small'  # Model type to visualize
CHECKPOINT_FILE = "best_model_midas_semantics_cross_attention_no_lb.pth"  # Model checkpoint to visualize with
OUTPUT_DIR = os.path.join(PROJECT_DIR, "visualization", CHECKPOINT_FILE.split('.')[0].replace('best_model_', ''))
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "results", CHECKPOINT_FILE)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')

# Load model
def load_model(model_type, checkpoint_path, model_cfg=None):
    if model_type == 'MiDaS_small':
        if model_cfg.dinov2_type is not None:
            model = MidasNetSemantics(None, features=64, backbone="efficientnet_lite3", exportable=True, 
                                    non_negative=True, cfg=model_cfg, blocks={'expand': True}, 
                                    dinov2_type=model_cfg.dinov2_type)
        else:
            model = MidasNet_small(None, features=64, backbone="efficientnet_lite3", exportable=True, 
                                    non_negative=True, cfg=model_cfg, blocks={'expand': True})

    checkpoint = torch.load(checkpoint_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def load_dataset():
    # Define dataset for visualization
    test_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    def target_transform(depth):
        # Resize the depth map to match input size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0), 
            size=INPUT_SIZE, 
            mode='bilinear', 
            align_corners=True
        ).squeeze()
        # Add channel dimension to match model output
        depth = depth.unsqueeze(0)
        return depth

    # Create training dataset with ground truth
    dataset = DepthDataset(
        data_dir=GT_DIR,
        list_file=TRAIN_LIST_PATH, 
        transform=test_transform,
        target_transform=target_transform,
        has_gt=True
    )
    return dataset

# Define per-pixel scale-invariant loss function
def per_pixel_scale_invariant_loss(pred, target):
    """
    Computes the per-pixel scale-invariant loss between the predicted depth and target depth.
    Both pred and target are expected to have shape (H, W) for a single sample.
    Returns a tensor of shape (H, W) containing the per-pixel loss values.
    
    The scale-invariant loss at each pixel is:
    L = (log(pred) - log(target))^2 - (mean(log(pred) - log(target)))^2
    """
    assert pred.shape == target.shape, \
        "Pred and target must have the same shape, got {} and {}".format(pred.shape, target.shape)
    assert (pred > 0).all() and (target > 0).all(), "Pred and target must be positive"

    # Compute log differences at each pixel
    log_diff = torch.log(pred) - torch.log(target)
    
    alpha = -torch.mean(log_diff)
    
    # Compute per-pixel scale-invariant loss
    per_pixel_loss = (log_diff + alpha) ** 2
    
    return per_pixel_loss

# Visualization function
def visualize_sample(rgb_image, pred_depth, gt_depth, loss_map=None, save_path=None):
    """
    Visualize a single sample in a 2x2 grid:
    - RGB image
    - Per-pixel loss map
    - Predicted depth
    - Ground truth depth
    
    Args:
        rgb_image: RGB image [3, H, W] or [H, W, 3]
        pred_depth: Predicted depth map [H, W]
        gt_depth: Ground truth depth map [H, W]
        loss_map: Per-pixel loss map [H, W] (optional)
        save_path: Path to save the visualization (optional)
    """
    # Ensure RGB image is in [H, W, 3] format
    if rgb_image.shape[0] == 3:
        rgb_image = np.transpose(rgb_image, (1, 2, 0))

    normalized_rgb = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-6)
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot RGB image
    axes[0].imshow(normalized_rgb)
    axes[0].set_title('Input RGB Image')
    axes[0].axis('off')
    
    # Plot loss map if provided, otherwise leave empty
    if loss_map is not None:
        im = axes[1].imshow(loss_map, cmap='hot')
        axes[1].set_title('Per-pixel Loss Map')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, 'No Loss Map Available', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[1].transAxes)
        axes[1].set_title('Per-pixel Loss Map')
    axes[1].axis('off')
    
    # Plot ground truth depth
    axes[2].imshow(gt_depth, cmap='plasma')
    axes[2].set_title('Ground Truth Depth')
    axes[2].axis('off')
    
    # Plot predicted depth
    axes[3].imshow(pred_depth, cmap='plasma')
    axes[3].set_title('Predicted Depth')
    axes[3].axis('off')

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_PATH)
    model_cfg = config.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model {MODEL_TYPE} from {CHECKPOINT_PATH}")
    model = load_model(MODEL_TYPE, CHECKPOINT_PATH, model_cfg)
    model = model.to(device)  # Move model to GPU
    print("Model loaded")

    print("Loading dataset...")
    dataset = load_dataset()
    print("Dataset loaded")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with torch.no_grad():
        model.eval()

        for i in tqdm(range(N_SAMPLES), desc="Visualizing samples"):
            rgb, depth_gt, _ = dataset[i]

            rgb = rgb.to(device)
            depth_gt = depth_gt.to(device)
            depth_gt = depth_gt.squeeze()
            depth_pred = model(rgb.unsqueeze(0))
            depth_pred = depth_pred.squeeze()
            rgb = rgb.cpu()
            depth_gt = depth_gt.cpu()
            depth_pred = depth_pred.cpu()

            per_pixel_si_loss = per_pixel_scale_invariant_loss(depth_pred, depth_gt)
            visualize_sample(rgb, depth_pred, depth_gt, per_pixel_si_loss, os.path.join(OUTPUT_DIR, f'sample_{i:06d}_vis.png'))

    print("Visualization saved to ", OUTPUT_DIR)