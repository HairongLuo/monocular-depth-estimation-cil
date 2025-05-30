"""
Generate predictions and corresponding csv file 
using a trained model specified in config.yaml.
"""
import os
from util import load_dataset, load_model, generate_test_predictions
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

TRAIN_DATA_DIR = "/cluster/courses/cil/monocular_depth/data/train"
TEST_DATA_DIR = "/cluster/courses/cil/monocular_depth/data/test"
PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
TRAIN_LIST_PATH = os.path.join(PROJECT_DIR, 'data', 'train_list.txt')
TEST_LIST_PATH = os.path.join(PROJECT_DIR, 'data', 'test_list.txt')
INPUT_SIZE = (448, 576)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, 'predictions')
config = OmegaConf.load(CONFIG_PATH)
model_cfg = config.model
model_name = config.experiment.model_name
model_type = config.model.model_type
checkpoint_path = os.path.join(RESULTS_DIR, f'best_model_{model_name}.pth')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model {model_type} from {checkpoint_path}...")
    model = load_model(model_type, checkpoint_path, model_cfg)
    model = model.to(device)  # Move model to GPU
    print("Model loaded")

    print("Loading dataset...")
    _, test_dataset = load_dataset(INPUT_SIZE, TRAIN_DATA_DIR, TRAIN_LIST_PATH, TEST_DATA_DIR, TEST_LIST_PATH)
    print("Dataset loaded")
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    generate_test_predictions(model, test_loader, device, PREDICTIONS_DIR)
    print("Predictions generated and saved to:", PREDICTIONS_DIR)

    # invoke create_prediction_csv.py to generate the CSV file
    os.system(f"python {os.path.join(PROJECT_DIR, 'create_prediction_csv.py')}")
    print("CSV file created with predictions.")