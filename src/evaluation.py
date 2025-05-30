from collections import OrderedDict
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from network.midas_net_custom import MidasNet_small
from network.midas_semantics import MidasNetSemantics
from util import scale_invariant_loss, absolute_relative_error, delta_thres
from main import DepthDataset
from omegaconf import OmegaConf
from tqdm import tqdm

GT_DIR = "/cluster/courses/cil/monocular_depth/data/train"
PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
TRAIN_LIST_PATH = os.path.join(PROJECT_DIR, 'data', 'train_list.txt')
INPUT_SIZE = (448, 576)
N_SAMPLES = 1000  # Number of samples to visualize
# Configure these variables in config.yaml
# MODEL_TYPE = 'MiDaS_small'  # Model type to visualize
# CHECKPOINT_FILE = "best_model_midas_small_nolb.pth"  # Model checkpoint to visualize with
# CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "results", CHECKPOINT_FILE)
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


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    model = model.to(device)  # Move model to GPU
    print("Model loaded")

    print("Loading dataset...")
    dataset = load_dataset()
    print("Dataset loaded")

    # Create a DataLoader for batch processing
    batch_size = 32  # Adjust this based on your GPU memory
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    with torch.no_grad():
        model.eval()
        total_si = 0
        total_rel = 0
        total_delta = [0] * N_DELTA
        samples_processed = 0
        
        for batch_idx, (rgb_batch, depth_gt_batch, _) in enumerate(tqdm(dataloader, desc="Evaluating batches")):
            if samples_processed >= N_SAMPLES:
                break
                
            # Move batch to GPU
            rgb_batch = rgb_batch.to(device)
            depth_gt_batch = depth_gt_batch.to(device)
            
            # Forward pass
            depth_pred_batch = model(rgb_batch)
            if depth_pred_batch.dim() == 3:
                depth_pred_batch = depth_pred_batch.unsqueeze(1)
            
            # Calculate metrics for the batch
            si_loss = scale_invariant_loss(depth_pred_batch, depth_gt_batch)
            total_si += si_loss * len(rgb_batch)
            
            abs_rel_error = absolute_relative_error(depth_pred_batch, depth_gt_batch)
            total_rel += abs_rel_error * len(rgb_batch)
            
            for j in range(1, N_DELTA + 1):
                delta_thres_value = BASE_THRES ** j
                delta_thres_result = delta_thres(depth_pred_batch, depth_gt_batch, thres=delta_thres_value)
                total_delta[j-1] += delta_thres_result * len(rgb_batch)
            
            samples_processed += len(rgb_batch)
            if samples_processed > N_SAMPLES:
                # Adjust the last batch's contribution
                excess = samples_processed - N_SAMPLES
                total_si -= si_loss * excess
                total_rel -= abs_rel_error * excess
                for j in range(N_DELTA):
                    total_delta[j] -= delta_thres_result * excess
                samples_processed = N_SAMPLES

        # Calculate final averages
        avg_si_loss = total_si / samples_processed
        avg_rel_error = total_rel / samples_processed
        avg_delta = [total / samples_processed for total in total_delta]
        print(f"Average Scale-Invariant Loss: {avg_si_loss.item()}")
        print(f"Average Absolute Relative Error: {avg_rel_error.item()}")
        for j in range(1, N_DELTA + 1):
            print(f"Average Delta {BASE_THRES ** j} Threshold: {avg_delta[j-1].item()}")