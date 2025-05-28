import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.transforms as T
import kornia.augmentation as K
import kornia.geometry as KG
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from network.dpt_depth import DPTDepthModel
# from preprocessing.transforms import transforms
from omegaconf import OmegaConf
from network.midas_net import MidasNet
from network.midas_net_custom import MidasNet_small
from network.midas_semantics import MidasNetSemantics
from loguru import logger as guru

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = (448, 576)  # Resize to multiples of both 14 and 16
NUM_WORKERS = 4
PIN_MEMORY = True
PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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

class DepthDataset(Dataset):
    def __init__(self, data_dir, list_file, transform=None, target_transform=None, has_gt=True, extra_augmentation=False):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.has_gt = has_gt
        self.extra_augmentation = extra_augmentation
        
        # Read file list
        with open(list_file, 'r') as f:
            if has_gt:
                self.file_pairs = [line.strip().split() for line in f]
            else:
                # For test set without ground truth
                self.file_list = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.file_pairs if self.has_gt else self.file_list)
    
    def __getitem__(self, idx):
        if self.has_gt:
            rgb_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
            depth_path = os.path.join(self.data_dir, self.file_pairs[idx][1])
            
            # Load RGB image
            rgb = Image.open(rgb_path).convert('RGB')
            
            # Load depth map
            depth = np.load(depth_path).astype(np.float32)
            depth = torch.from_numpy(depth)
            
            # Apply transformations
            if self.extra_augmentation:
                rgb = F.to_tensor(rgb).unsqueeze(0)
                depth = self.target_transform(depth).unsqueeze(0)
                rgb, depth = self.transform(rgb, depth)
                rgb = rgb.squeeze(0)
                depth = depth.squeeze(0)
            else:
                if self.transform:
                    rgb = self.transform(rgb)
                
                if self.target_transform:
                    depth = self.target_transform(depth)
                else:
                    # Add channel dimension if not done by transform
                    depth = depth.unsqueeze(0)     
            
            return rgb, depth, self.file_pairs[idx][0]  # Return filename for saving predictions
        else:
            # For test set without ground truth
            rgb_path = os.path.join(self.data_dir, self.file_list[idx].split(' ')[0])
            
            # Load RGB image
            rgb = Image.open(rgb_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                rgb = self.transform(rgb)
            
            return rgb, self.file_list[idx]  # No depth, just return the filename
    
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

def combined_loss(pred, target, config, rgb=None):
    """
    Combined loss function using scale-invariant, gradient, and edge-aware losses.
    
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        config: Configuration object containing loss weights
        rgb: RGB input image (optional, required for edge-aware loss)
    """
    # Scale-invariant loss
    si_loss = scale_invariant_loss(pred, target) * config.model.loss_function.si_loss_alpha

    # Scale-Invariant Logarithmic Loss (SiLog Loss) in MiDaS paper
    slog_loss = silog_loss(pred, target, mask=(target > 0).detach(),
                            variance_focus=config.model.loss_function.silog_loss.variance_focus) \
    * config.model.loss_function.silog_loss.alpha
    
    # Gradient loss
    grad_loss = gradient_loss(pred, target) * config.model.loss_function.grad_loss_alpha
    
    # Edge-aware loss (if RGB image is provided)
    edge_loss = 0.0
    if rgb is not None:
        edge_loss = edge_aware_loss(pred, target, rgb, config.model.loss_function.edge_loss_alpha)
    
    # Combine losses
    total_loss = si_loss + slog_loss + grad_loss + edge_loss
    
    return total_loss, {
        'si_loss': si_loss.item(),
        'silog_loss': slog_loss.item(),
        'grad_loss': grad_loss.item(),
        'edge_loss': edge_loss.item() if rgb is not None else 0.0
    }

def train_model(model, train_loader, val_loader, loss_function, optimizer, device, results_dir, config):
    """Train the model and save the best based on validation metrics with early stopping
    
    Args:
        early_stopping_config: Dictionary containing early stopping parameters (patience and min_delta)
    """
    start_epoch = config.training.resume_training.resume_from_epoch if config.training.resume_training.resume else 0
    best_val_loss = float('inf')
    best_epoch = start_epoch
    train_losses = []
    val_losses = []
    num_epochs = config.training.n_epoch
    model_name = config.experiment.model_name
    early_stopping_config = config.training.early_stopping
    
    # Early stopping variables
    patience = early_stopping_config.patience
    min_delta = early_stopping_config.min_delta
    counter = 0
    early_stop = False
    
    for epoch in range(start_epoch, num_epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_losses_dict = {'si_loss': 0.0, 'grad_loss': 0.0, 'edge_loss': 0.0, 'silog_loss': 0.0}
        
        for inputs, targets, _ in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs).unsqueeze(1)
            
            # Compute combined loss with RGB input for edge-aware loss
            loss, loss_dict = loss_function(
                outputs, 
                targets, 
                config,
                rgb=inputs  # Pass RGB input for edge-aware loss
            )
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update loss tracking
            curr_train_loss = loss.item() * inputs.size(0)
            train_loss += curr_train_loss
            curr_train_losses_dict = {k: v * inputs.size(0) for k, v in loss_dict.items()}
            for k, v in curr_train_losses_dict.items():
                train_losses_dict[k] += v
            
            wandb.log({
                "iteration_train_loss": curr_train_loss,
                **{f"iteration_{k}": v for k, v in curr_train_losses_dict.items()}
            })
        
        # Average losses
        train_loss /= len(train_loader.dataset)
        for k in train_losses_dict:
            train_losses_dict[k] /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss_combined = 0.0
        val_losses_dict = {'si_loss': 0.0, 'grad_loss': 0.0, 'edge_loss': 0.0, 'silog_loss': 0.0}
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs).unsqueeze(1)
                
                # Compute combined loss with RGB input for edge-aware loss
                loss, loss_dict = loss_function(
                    outputs,
                    targets,
                    config,
                    rgb=inputs  # Pass RGB input for edge-aware loss
                )
                
                val_loss_combined += loss.item() * inputs.size(0)
                # Update loss tracking
                for k, v in loss_dict.items():
                    val_losses_dict[k] += v * inputs.size(0)
        
        # Average validation losses
        val_loss_combined /= len(val_loader.dataset)
        for k in val_losses_dict:
            val_losses_dict[k] /= len(val_loader.dataset)
        
        print(f"Train Loss: {train_loss:.4f} (SI: {train_losses_dict['si_loss']:.4f}, "
              f"Grad: {train_losses_dict['grad_loss']:.4f}, "
              f"Edge: {train_losses_dict['edge_loss']:.4f})")
        print(f"Val Loss: {val_loss_combined:.4f} (SI: {val_losses_dict['si_loss']:.4f}, "
              f"Grad: {val_losses_dict['grad_loss']:.4f}, "
              f"Edge: {val_losses_dict['edge_loss']:.4f})")

        # Early stopping check
        if val_loss_combined < best_val_loss - min_delta:
            best_val_loss = val_loss_combined
            best_epoch = epoch
            counter = 0
            # Save best model and training state
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'early_stopping_counter': counter,
                'train_loss': train_loss,
                'val_loss': val_loss_combined,
                'config': {
                    'model_name': model_name,
                    'num_epochs': num_epochs,
                    'early_stopping': {
                        'patience': patience,
                        'min_delta': min_delta
                    }
                }
            }
            torch.save(checkpoint, os.path.join(results_dir, f'best_model_{model_name}.pth'))
            print(f"New best model saved at epoch {epoch+1} with validation loss: {val_loss_combined:.4f}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                early_stop = True

        wandb.log({
            "epoch": epoch,
            "epoch_train_loss": train_loss,
            "epoch_val_loss": val_losses_dict['si_loss'],
            "epoch_val_loss_combined": val_loss_combined,
            **{f"epoch_train_{k}": v for k, v in train_losses_dict.items()},
            **{f"epoch_val_{k}": v for k, v in val_losses_dict.items()},
            "early_stopping_counter": counter,
            "early_stop_triggered": early_stop
        })
        
        print("The training time for epoch", epoch, " is: %s.\n" % (time.time() - start_time))
    
    print(f"\nBest model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")

    wandb.finish()
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(results_dir, f'best_model_{model_name}.pth'))["model_state_dict"])
    
    return model

def evaluate_model(model, val_loader, device):
    """Evaluate the model and compute metrics on validation set"""
    model.eval()
    
    mae = 0.0
    rmse = 0.0
    rel = 0.0
    delta1 = 0.0
    delta2 = 0.0
    delta3 = 0.0
    sirmse = 0.0
    
    total_samples = 0
    target_shape = None
    
    with torch.no_grad():
        for inputs, targets, filenames in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            if target_shape is None:
                target_shape = targets.shape
            

            # Forward pass
            outputs = model(inputs).unsqueeze(1)
            
            # Resize outputs to match target dimensions
            print(f"outputs shape: {outputs.shape}")
            print(f"target shape: {targets.shape}")
            outputs = nn.functional.interpolate(
                outputs,
                size=targets.shape[-2:],  # Match height and width of targets
                mode='bilinear',
                align_corners=True
            )
            
            # Calculate metrics
            abs_diff = torch.abs(outputs - targets)
            mae += torch.sum(abs_diff).item()
            rmse += torch.sum(torch.pow(abs_diff, 2)).item()
            rel += torch.sum(abs_diff / (targets + 1e-6)).item()
            
            # Calculate scale-invariant RMSE for each image in the batch
            for i in range(batch_size):
                # Convert tensors to numpy arrays
                pred_np = outputs[i].cpu().squeeze().numpy()
                target_np = targets[i].cpu().squeeze().numpy()
                
                EPSILON = 1e-6
                
                valid_target = target_np > EPSILON
                if not np.any(valid_target):
                    continue
                
                target_valid = target_np[valid_target]
                pred_valid = pred_np[valid_target]
                
                log_target = np.log(target_valid)
                
                pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON)
                log_pred = np.log(pred_valid)
                
                # Calculate scale-invariant error
                diff = log_pred - log_target
                diff_mean = np.mean(diff)
                
                # Calculate RMSE for this image
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))
            
            # Calculate thresholded accuracy
            max_ratio = torch.max(outputs / (targets + 1e-6), targets / (outputs + 1e-6))
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()
            
            # Save some sample predictions
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 5)):
                    idx = total_samples - batch_size + i
                    
                    # Convert tensors to numpy arrays
                    input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                    target_np = targets[i].cpu().squeeze().numpy()
                    output_np = outputs[i].cpu().squeeze().numpy()
                    
                    # Normalize for visualization
                    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)
                    
                    # Create visualization
                    # plt.figure(figsize=(15, 5))
                    
                    # plt.subplot(1, 3, 1)
                    # plt.imshow(input_np)
                    # plt.title("RGB Input")
                    # plt.axis('off')
                    
                    # plt.subplot(1, 3, 2)
                    # plt.imshow(target_np, cmap='plasma')
                    # plt.title("Ground Truth Depth")
                    # plt.axis('off')
                    
                    # plt.subplot(1, 3, 3)
                    # plt.imshow(output_np, cmap='plasma')
                    # plt.title("Predicted Depth")
                    # plt.axis('off')
                    
                    # plt.tight_layout()
                    # plt.savefig(os.path.join(results_dir, f"sample_{idx}.png"))
                    # plt.close()
            
            # Free up memory
            del inputs, targets, outputs, abs_diff, max_ratio
            
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Calculate final metrics using stored target shape
    total_pixels = target_shape[1] * target_shape[2] * target_shape[3]  # channels * height * width
    mae /= total_samples * total_pixels
    rmse = np.sqrt(rmse / (total_samples * total_pixels))
    rel /= total_samples * total_pixels
    sirmse = sirmse / total_samples
    delta1 /= total_samples * total_pixels
    delta2 /= total_samples * total_pixels
    delta3 /= total_samples * total_pixels
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'siRMSE': sirmse,
        'REL': rel,
        'Delta1': delta1,
        'Delta2': delta2,
        'Delta3': delta3
    }
    
    return metrics

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

def scale_invariant_loss(pred, target, epsilon=1e-6):
    """
    Computes the scale-invariant loss between the predicted depth and target depth.
    Both pred and target are expected to have shape (B, 1, H, W).
    """
    # Ensure the target is the same size as prediction.
    # print(f"pred shape: {pred.shape}")
    # print(f"target shape: {target.shape}")
    if pred.shape != target.shape:
        target = nn.functional.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=True)
    
    # Compute the logarithms.
    log_pred = torch.log(pred + epsilon)
    log_target = torch.log(target + epsilon)
    
    # Compute difference.
    diff = log_pred - log_target
    n = diff.numel() / diff.shape[0]  # number of pixels per sample.
    term1 = torch.sum(diff ** 2, dim=[1, 2, 3]) / n
    term2 = (torch.sum(diff, dim=[1, 2, 3]) ** 2) / (n ** 2)
    
    loss = torch.mean(term1 - term2)
    return loss


def init_model(configs):
    usr_name = configs.paths.usr_name
    model_cfg = configs.model
    model_type = model_cfg.model_type
    network_cfg = model_cfg.network
    os.makedirs(os.path.join(PROJECT_DIR, "pretrain_weights"), exist_ok=True)

    if model_type == "DPT_Hybrid":
        pretrain_model_path = "/home/" + usr_name + "/monocular-depth-estimation-cil/pretrain_weights/dpt_hybrid_384.pt"      # edit your path
        checkpoint_url = (
            "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt"
        )
        backbone = "vitb_rn50_384"
        # midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        model = DPTDepthModel(
            path=None,
            backbone=backbone,     # or "vitb_rn50_384" for hybrid
            non_negative=True,
        )
    elif model_type == "MiDaS":
        pretrain_model_path = "/home/" + usr_name + "/monocular-depth-estimation-cil/pretrain_weights/midas_v21_384.pt"      # edit your path
        checkpoint_url = (
            "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt"
        )
        model = MidasNet()
    elif model_type == "MiDaS_small":
        pretrain_model_path = os.path.join(PROJECT_DIR, "pretrain_weights", "midas_v21_small_256.pt")      # edit your path
        checkpoint_url = (
            "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
        )
        if model_cfg.dinov2_type is not None:
            model = MidasNetSemantics(None, features=64, backbone="efficientnet_lite3", exportable=True, 
                                    non_negative=True, cfg=network_cfg, blocks={'expand': True}, 
                                    dinov2_type=model_cfg.dinov2_type)
        else:
            model = MidasNet_small(None, features=64, backbone="efficientnet_lite3", exportable=True, 
                                 non_negative=True, cfg=network_cfg, blocks={'expand': True})

    # Check if we should resume training
    if hasattr(configs.training, 'resume_training') and configs.training.resume_training.resume:
        best_model_path = os.path.join(PROJECT_DIR, "results", f'best_model_{configs.experiment.model_name}.pth')
        if os.path.exists(best_model_path):
            print(f"Resuming training from best model: {best_model_path}")
            state_dict = torch.load(best_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict if 'state_dict' not in state_dict else state_dict['state_dict'], strict=False)
            return model
        else:
            print(f"No best model found at {best_model_path}. Loading pretrained weights instead.")

    # Load pretrained weights if not resuming
    if not os.path.exists(pretrain_model_path):
        print(f"Model weights not found at {pretrain_model_path}. Downloading...")
        os.system(f"wget -O {pretrain_model_path} {checkpoint_url}")
    state_dict = torch.load(pretrain_model_path, map_location=torch.device('cpu'))  # load to cpu before switching to DEVICE, by default cpu?

    # Load pretrained weights to model
    if model_cfg.dinov2_type is not None:
        # Only keep weights to unmodified modules
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}
        # Update the model's state dict with filtered weights
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict, strict=False)
        # Check which weights were not loaded
        missing_weights = [k for k in state_dict.keys() if k not in model_dict]
        print(f"Missing weights: {missing_weights}")
        print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} pretrained DPT weights")
    else:
        model.load_state_dict(state_dict, strict=False)        # load to cpu before switching to DEVICE, by default cpu?
    return model

CROP = min(INPUT_SIZE)

class PairAug(torch.nn.Module):
    """
    Apply the *same* geometric transform to img and depth,
    but photometric only to the RGB.
    """
    def __init__(self):
        super().__init__()
        self.resize = transforms.Compose([
            transforms.Resize(INPUT_SIZE)
        ])
        self.geo = torch.nn.Sequential(        # geo ≡ img & depth
            K.RandomResizedCrop(
                size=INPUT_SIZE, scale=(0.8, 1.25), ratio=(1.0, 1.0)
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=3.0, p=0.3,
                             resample='bilinear', align_corners=False),
        )
        self.photo = torch.nn.Sequential(      # photo ≡ img only
            K.ColorJitter(0.4, 0.4, 0.4, 0.15, p=0.8),
            K.RandomGaussianNoise(std=0.005, p=0.25),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
            K.RandomGamma(gamma=(0.9, 1.1), p=0.3),
        )
        self.norm = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def forward(self, img, depth):
        # img, depth are tensors in [0,1], shape (B,C,H,W)/(B,1,H,W)
        img = self.resize(img)
        pair = torch.cat([img, depth], dim=1)      # (B, C+1, H, W)
        pair = self.geo(pair)
        img, depth = pair[:, :3], pair[:, 3:]      # split back
        img = self.photo(img)
        img = self.norm(img)
        return img, depth

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
    config = OmegaConf.load(config_path)
    # Create output directories

    usr_name = config.paths.usr_name
    data_dir = config.paths.data_dir
    local_data_dir = f'/home/{usr_name}/monocular-depth-estimation-cil/data'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    train_list_file = os.path.join(local_data_dir, 'train_list.txt')
    test_list_file = os.path.join(local_data_dir, 'test_list.txt')
    output_dir = f'/home/{usr_name}/monocular-depth-estimation-cil'
    results_dir = os.path.join(output_dir, 'results')
    predictions_dir = os.path.join(output_dir, 'predictions')

    import time
    current_time = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = f"{config.experiment.model_name}_{current_time}"  # Use model_name from config
    ensure_dir(results_dir)
    ensure_dir(predictions_dir)

    # wandb stuff
    wandb.init(mode="disabled" if config.experiment.wandb_disable else None,
               project="MonocularDepthEstimation",
               name=f"{config.experiment.model_name}_{current_time}" if not config.training.resume_training.resume else None,
               id=config.training.resume_training.run_id if config.training.resume_training.resume else None,
               resume="allow" if config.training.resume_training.resume else None,
               config={
                   "epochs": config.training.n_epoch,
                   "batch_size": config.training.batch_size,
                   "learning_rate": LEARNING_RATE,
                   "model_name": config.experiment.model_name,
                   "run_time": current_time,  # Add timestamp to config
                   "resume_training": config.training.resume_training.resume,
                   "early_stopping": {
                       "patience": config.training.early_stopping.patience,
                       "min_delta": config.training.early_stopping.min_delta
                   }
               })
    
    # Define transforms
    # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    batch_size = config.training.batch_size
    extra_augmentation = config.augmentation

    if extra_augmentation:
        train_transform = PairAug()
    else:
        train_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create training dataset with ground truth
    train_full_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list_file, 
        transform=train_transform,
        target_transform=target_transform,
        has_gt=True,
        extra_augmentation=extra_augmentation
    )
    
    # Create test dataset without ground truth
    test_dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list_file,
        transform=test_transform,
        has_gt=False  # Test set has no ground truth
    )
    
    # Split training dataset into train and validation
    total_size = len(train_full_dataset)
    train_size = int(0.85 * total_size)  # 85% for training
    val_size = total_size - train_size    # 15% for validation
    
    # Set a fixed random seed for reproducibility
    torch.manual_seed(0)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )
    
    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Clear CUDA cache before model initialization
    torch.cuda.empty_cache()
    
    # Display GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    model = init_model(config)
    # model = nn.DataParallel(model)  # No need for DataParallel if using a single GPU 
    model = model.to(DEVICE)

    print(f"Using device: {DEVICE}")

    # Print memory usage after model initialization
    if torch.cuda.is_available():
        print(f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # TEST ON SINGLE IMAGE
    # img = train_full_dataset[0][0]
    # img = img.unsqueeze(0)
    # img = img.to(DEVICE)
    # output = model(img)
    # print(f"Output shape: {output.shape}")


    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # evaluate the best model
    # evaluate_best_model = config.opt.evaluate_best_model
    # best_model_path = os.path.join(results_dir, 'best_model.pth')
    # if evaluate_best_model:
    #     model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    #     generate_test_predictions(model, test_loader, DEVICE, predictions_dir)
    
    
    # Train the model
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, combined_loss, optimizer, DEVICE, results_dir, config)
            
    # Evaluate the model on validation set
    # print("Evaluating model on validation set...")
    # metrics = evaluate_model(model, val_loader, DEVICE)
    
    # # Print metrics
    # print("\nValidation Metrics:")
    # for name, value in metrics.items():
    #     print(f"{name}: {value:.4f}")
    
    # # Save metrics to file
    # with open(os.path.join(results_dir, 'validation_metrics.txt'), 'w') as f:
    #     for name, value in metrics.items():
    #         f.write(f"{name}: {value:.4f}\n")
    
    # Generate predictions for the test set
    print("Generating predictions for test set...")
    generate_test_predictions(model, test_loader, DEVICE, predictions_dir)
    
    print(f"Results saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")


if __name__ == "__main__":
    main()
