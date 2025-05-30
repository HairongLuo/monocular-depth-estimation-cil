from collections import OrderedDict
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
OUTPUT_DIR = os.path.join(PROJECT_DIR, "visualization")
TRAIN_LIST_PATH = os.path.join(PROJECT_DIR, 'data', 'train_list.txt')
INPUT_SIZE = (448, 576)
N_SAMPLES = 3  # Number of samples to visualize
MODEL_TYPE = 'MiDaS_small'  # Model type to visualize
CHECKPOINT_FILE = "best_model_midas_small_nolb_w_grad_loss.pth"  # Model checkpoint to visualize with
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "results", CHECKPOINT_FILE)
USE_PRETRAINED_ENCODER = False
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
N_DELTA = 3
BASE_THRES = 1.05

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k.replace("module.", "", 1)  # remove only the first "module."
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# Load model
def load_model(model_type, checkpoint_path, model_cfg=None):
    pretrain_path = "/home/lingxi/monocular-depth-estimation-cil/pretrain_weights/midas_v21_small_256.pt"
    if model_type == 'MiDaS_small':
        if model_cfg.dinov2_type is not None:
            model = MidasNetSemantics(None, features=64, backbone="efficientnet_lite3", exportable=True, 
                                    non_negative=True, cfg=model_cfg.network, blocks={'expand': True}, 
                                    dinov2_type=model_cfg.dinov2_type)
        else:
            model = MidasNet_small(None, features=64, backbone="efficientnet_lite3", exportable=True, 
                                    non_negative=True, cfg=model_cfg.network, blocks={'expand': True})

    checkpoint = torch.load(checkpoint_path)
    if USE_PRETRAINED_ENCODER:
        model.load_state_dict(torch.load(pretrain_path), strict=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        checkpoint = remove_module_prefix(checkpoint)
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

    delta =  torch.log(pred) - torch.log(target)
    alpha = torch.mean(-delta)
    calibrated_delta = torch.square(delta + alpha)
    mean_loss = torch.sqrt(torch.mean(calibrated_delta))
    return mean_loss

def delta_thres(pred, target, thres=0.1):
    """
    Computes the delta threshold metric for depth prediction.
    Returns a boolean tensor indicating whether each pixel meets the threshold.
    """
    assert pred.shape == target.shape, \
        "Pred and target must have the same shape, got {} and {}".format(pred.shape, target.shape)

    epsilon = 1e-6
    # Compute optimal scale s
    log_pred = torch.log(pred + epsilon)
    log_target = torch.log(target + epsilon)
    scale = torch.exp(torch.mean(log_target - log_pred))

    # Align prediction
    aligned_pred = pred * scale

    # Compute ratio and delta
    ratio = torch.max(aligned_pred / target, target / aligned_pred)
    acc = torch.mean((ratio < thres).float())
    return acc


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


if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_PATH)
    model_cfg = config.model
    usr_name = config.paths.usr_name
    output_dir = f'/home/{usr_name}/monocular-depth-estimation-cil'
    results_dir = os.path.join(output_dir, 'results')
    

    model_name = config.experiment.model_name
    model_type = config.model.model_type
    checkpoint_path = os.path.join(results_dir, f'best_model_{model_name}.pth')
    print(f"Loading model {model_type} from {checkpoint_path}")
    model = load_model(model_type, checkpoint_path, model_cfg)
    print("Model loaded")

    print("Loading dataset...")
    dataset = load_dataset()
    print("Dataset loaded")

    with torch.no_grad():
        model.eval()
        total_si = 0
        total_rel = 0
        total_delta = [0] * N_DELTA
        for i in tqdm(range(N_SAMPLES), desc="Visualizing samples"):
            rgb, depth_gt, _ = dataset[i]
            depth_gt = depth_gt.squeeze()
            depth_pred = model(rgb.unsqueeze(0))
            depth_pred = depth_pred.squeeze()
            si_loss = per_pixel_scale_invariant_loss(depth_pred, depth_gt)
            total_si += si_loss
            abs_rel_error = absolute_relative_error(depth_pred, depth_gt)
            total_rel += abs_rel_error
            for j in range(1, N_DELTA + 1):
                delta_thres_value = BASE_THRES ** j
                delta_thres_result = delta_thres(depth_pred, depth_gt, thres=delta_thres_value)
                total_delta[j-1] += delta_thres_result
        avg_si_loss = total_si / N_SAMPLES
        avg_rel_error = total_rel / N_SAMPLES
        avg_delta = [total / N_SAMPLES for total in total_delta]
        print(f"Average Scale-Invariant Loss: {avg_si_loss.item()}")
        print(f"Average Absolute Relative Error: {avg_rel_error.item()}")
        for j in range(1, N_DELTA + 1):
            print(f"Average Delta {BASE_THRES ** j} Threshold: {avg_delta[j-1].item()}")