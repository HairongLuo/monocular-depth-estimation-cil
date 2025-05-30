import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torchvision import transforms
from network.midas_net_custom import MidasNet_small
from network.midas_semantics import MidasNetSemantics
from dataset import DepthDataset

def gradient_loss(pred, target):
    """
    Gradient loss for depth estimation.
    Computes the difference between gradients of predicted and target depth maps.
    
    Args:
        pred: Predicted depth map (B, 1, H, W)
        target: Ground truth depth map (B, 1, H, W)
        alpha: Weight for gradient loss
    """
    # Compute gradients in x and y directions
    pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    
    # Compute gradient loss
    dx_loss = torch.mean(torch.abs(pred_dx - target_dx))
    dy_loss = torch.mean(torch.abs(pred_dy - target_dy))
    
    return dx_loss + dy_loss

def edge_aware_loss(pred, target, rgb, beta=0.5):
    """
    Edge-aware loss for depth estimation.
    Uses RGB image gradients to weight depth gradients, helping preserve sharp edges.
    
    Args:
        pred: Predicted depth map (B, 1, H, W)
        target: Ground truth depth map (B, 1, H, W)
        rgb: RGB input image (B, 3, H, W)
        beta: Weight for edge-aware loss
    """
    # Compute RGB image gradients
    rgb_dx = torch.abs(rgb[:, :, :, :-1] - rgb[:, :, :, 1:])  # (B, 3, H, W-1)
    rgb_dy = torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :])  # (B, 3, H-1, W)
    
    # Pad gradients to match dimensions
    rgb_dx_padded = torch.nn.functional.pad(rgb_dx, (0, 1, 0, 0))  # Pad right
    rgb_dy_padded = torch.nn.functional.pad(rgb_dy, (0, 0, 0, 1))  # Pad bottom
    
    # Compute RGB gradient magnitude using padded gradients
    rgb_grad_mag = torch.sqrt(rgb_dx_padded.pow(2).mean(dim=1, keepdim=True) + 
                            rgb_dy_padded.pow(2).mean(dim=1, keepdim=True))
    
    # Normalize RGB gradient magnitude to [0, 1]
    rgb_grad_mag = (rgb_grad_mag - rgb_grad_mag.min()) / (rgb_grad_mag.max() - rgb_grad_mag.min() + 1e-6)
    
    # Compute depth gradients
    pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    
    # Pad depth gradients to match dimensions
    pred_dx_padded = torch.nn.functional.pad(pred_dx, (0, 1, 0, 0))
    pred_dy_padded = torch.nn.functional.pad(pred_dy, (0, 0, 0, 1))
    target_dx_padded = torch.nn.functional.pad(target_dx, (0, 1, 0, 0))
    target_dy_padded = torch.nn.functional.pad(target_dy, (0, 0, 0, 1))
    
    # Weight depth gradients by RGB gradient magnitude
    dx_loss = torch.mean(rgb_grad_mag * torch.abs(pred_dx_padded - target_dx_padded))
    dy_loss = torch.mean(rgb_grad_mag * torch.abs(pred_dy_padded - target_dy_padded))
    
    return beta * (dx_loss + dy_loss)

def silog_loss(pred, target, mask=None, variance_focus=0.85, epsilon=1e-6):
    """
    Scale-Invariant Logarithmic Loss (SiLog Loss), as described in the MiDaS paper.

    Args:
        pred (torch.Tensor): Predicted depth map, shape (B, 1, H, W)
        target (torch.Tensor): Ground truth depth map, shape (B, 1, H, W)
        mask (torch.Tensor, optional): Validity mask of same shape as pred/target, dtype torch.bool
        variance_focus (float): Weight for the variance term (between 0 and 1)
        epsilon (float): Small constant to avoid log(0)

    Returns:
        torch.Tensor: Scalar SiLog loss value
    """
    if pred.shape != target.shape:
        target = nn.functional.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=True)
    
    if mask is None:
        mask = (target > 0).detach()
    
    # Apply mask
    pred = pred[mask]
    target = target[mask]

    log_diff = torch.log(pred + epsilon) - torch.log(target + epsilon)

    # Mean squared log difference
    sq_log_diff = log_diff ** 2
    mean_sq_log_diff = torch.mean(sq_log_diff)

    # Squared mean log difference
    mean_log_diff = torch.mean(log_diff)
    sq_mean_log_diff = mean_log_diff ** 2

    # Final SiLog loss
    loss = mean_sq_log_diff - variance_focus * sq_mean_log_diff

    return loss

def scale_invariant_loss(pred, target, epsilon=1e-6, sqroot=False):
    """
    Computes the scale-invariant loss between the predicted depth and target depth.
    Both pred and target are expected to have shape (B, 1, H, W).
    """
    # Ensure the target is the same size as prediction.
    # print(f"pred shape: {pred.shape}")
    # print(f"target shape: {target.shape}")
    # if pred.shape != target.shape:
    #     target = nn.functional.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=True)
    assert pred.shape[-2:] == target.shape[-2:], \
        "Pred and target must have the same spatial dimensions, got {} and {}".format(pred.shape[-2:], target.shape[-2:])
    
    # Compute the logarithms.
    log_pred = torch.log(pred + epsilon)
    log_target = torch.log(target + epsilon)
    
    # Compute difference.
    diff = log_pred - log_target
    n = diff.numel() / diff.shape[0]  # number of pixels per sample.
    term1 = torch.sum(diff ** 2, dim=[1, 2, 3]) / n
    term2 = (torch.sum(diff, dim=[1, 2, 3]) ** 2) / (n ** 2)
    loss = term1 - term2
    # Match the scale-invariant loss definition in Kaggle
    if sqroot:
        loss = torch.sqrt(loss)

    return torch.mean(loss)

# Define per-pixel scale-invariant loss function
def per_pixel_scale_invariant_loss(pred, target):
    """
    DOESN'T SUPPORT BATCHING
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

def delta_thres(pred, target, thres=0.1):
    """
    Computes the delta threshold metric for depth prediction.
    Returns a boolean tensor indicating whether each pixel meets the threshold.
    """
    assert pred.shape == target.shape, \
        "Pred and target must have the same shape, got {} and {}".format(pred.shape, target.shape)

    epsilon = 1e-6
    B = pred.shape[0]

    pred = pred.view(B, -1)
    target = target.view(B, -1)

    log_pred = torch.log(pred + epsilon)
    log_target = torch.log(target + epsilon)

    scale = torch.exp(torch.mean(log_target - log_pred, dim=1, keepdim=True))  

    aligned_pred = pred * scale  

    ratio = torch.max(aligned_pred / target, target / aligned_pred)
    acc = torch.mean((ratio < thres).float(), dim=1)

    return acc.mean()  


def absolute_relative_error(pred, target):
    """
    Computes the absolute relative error between predicted and target depth.
    Returns a scalar tensor representing the mean absolute relative error.
    """
    assert pred.shape == target.shape, \
        "Pred and target must have the same shape, got {} and {}".format(pred.shape, target.shape)
    # Compute absolute relative error
    abs_rel = torch.mean(torch.abs(target - pred) / (target + 1e-6))
    return abs_rel

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

def load_dataset(input_size, train_dir, train_list_path, test_dir=None, test_list_path=None):
    """
    Loads the dataset for visualization and evaluation (No data augmentation).
    Args:
        input_size: Tuple of (height, width)
        train_dir: Path to the train directory ("/cluster/courses/cil/monocular_depth/data/train")
        train_list_path: Path to the train list file
        test_dir: Path to the test directory (optional)
        test_list_path: Path to the test list file (optional)
    Returns:
        dataset: DepthDataset object
    """
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    def target_transform(depth):
        # Resize the depth map to match input size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0), 
            size=input_size, 
            mode='bilinear', 
            align_corners=True
        ).squeeze()
        # Add channel dimension to match model output
        depth = depth.unsqueeze(0)
        return depth

    # Create training dataset with ground truth
    train_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list_path, 
        transform=test_transform,
        target_transform=target_transform,
        has_gt=True
    )
    if test_list_path and test_dir:
        test_dataset = DepthDataset(
            data_dir=test_dir,
            list_file=test_list_path,
            transform=test_transform,
            has_gt=False  # Test set has no ground truth
        )
        return train_dataset, test_dataset
    return train_dataset

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_test_predictions(model, test_loader, device, predictions_dir):
    """Generate predictions for the test set without ground truth"""
    model.eval()
    
    # Ensure predictions directory exists
    ensure_dir(predictions_dir)
    
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # Forward pass
            outputs = model(inputs).unsqueeze(1)
            
            # Resize outputs to match original input dimensions (426x560)
            outputs = nn.functional.interpolate(
                outputs,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )
            
            # Save all test predictions
            for i in range(batch_size):
                # Get filename without extension
                filename = filenames[i].split(' ')[1]
                
                # Save depth map prediction as numpy array
                depth_pred = outputs[i].cpu().squeeze().numpy()
                np.save(os.path.join(predictions_dir, f"{filename}"), depth_pred)
            
            # Clean up memory
            del inputs, outputs
        
        # Clear cache after test predictions
        torch.cuda.empty_cache()