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
from util import load_dataset, load_model
from util import per_pixel_scale_invariant_loss
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import OrderedDict

GT_DIR = "/cluster/courses/cil/monocular_depth/data/train"
PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
TRAIN_LIST_PATH = os.path.join(PROJECT_DIR, 'data', 'train_list.txt')
INPUT_SIZE = (448, 576)
N_SAMPLES = 100  # Number of samples to visualize
MODEL_TYPE = 'MiDaS_small'  # Model type to visualize
CHECKPOINT_FILE = "best_model_midas_semantics_cross_attention_no_lb.pth"  # Model checkpoint to visualize with
# OUTPUT_DIR = os.path.join(PROJECT_DIR, "visualization", CHECKPOINT_FILE.split('.')[0].replace('best_model_', ''))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "visualization")
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "results", CHECKPOINT_FILE)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')


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

def save_images(rgb_image, pred_depth, gt_depth, index, model_name, loss_map=None, save_path=None):
    plt.imsave(os.path.join(save_path, f'{index:06d}_gtd.png'), gt_depth.numpy(), cmap='plasma')
    plt.imsave(os.path.join(save_path, f'{index:06d}_{model_name}_pred.png'), pred_depth.numpy(), cmap='plasma')
    if loss_map is not None:
        plt.imsave(os.path.join(save_path, f'{index:06d}_{model_name}_lmap.png'), loss_map.numpy(), cmap='hot')


if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_PATH)
    model_cfg = config.model
    usr_name = config.paths.usr_name
    output_dir = f'/home/{usr_name}/monocular-depth-estimation-cil'
    results_dir = os.path.join(output_dir, 'results')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = config.experiment.model_name
    model_type = config.model.model_type
    checkpoint_path = os.path.join(results_dir, f'best_model_{model_name}.pth')

    print(f"Loading model {model_type} from {checkpoint_path}")
    model = load_model(model_type, checkpoint_path, model_cfg)
    model = model.to(device)  # Move model to GPU
    print("Model loaded")

    print("Loading dataset...")
    dataset = load_dataset(INPUT_SIZE, GT_DIR, TRAIN_LIST_PATH)
    print("Dataset loaded")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figures_dir = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

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
            # visualize_sample(rgb, depth_pred, depth_gt, per_pixel_si_loss, os.path.join(OUTPUT_DIR, f'sample_{i:06d}_vis.png'))
            save_images(rgb, depth_pred, depth_gt, i, model_name, per_pixel_si_loss, figures_dir)

    print("Visualization saved to ", OUTPUT_DIR)