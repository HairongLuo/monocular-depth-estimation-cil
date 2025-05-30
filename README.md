# Monocular Depth Estimation

## Overview

Project of Computational Intelligence Lab at ETH Zürich, 2025.

Authors: Xichong Ling, Yingzhe Liu, Hairong Luo

## Training

To train our model, configure the training in `configs/config.yaml`, including data directory, result directory, training epochs, model type, use of features and weights of loss components, etc. Then run:
```bash
cd src
python main.py
```
The trained model will be saved at the folder specified by `results_dir`.

## Evaluation

To check the number of trainable parameters and evaluate models using metrics including Scale-Invariant RMSE, Absolute Relative Error, and Delta, configure the model in `configs/config.yaml` to specify the model and checkpoint to load, then run:
```bash
cd src
python evaluation.py
``` 

## Visualization

To visualize the predicted depth map and error map, configure the model to load in `configs/config.yaml` and specify the checkpoint path and other parameters in `src/visualize.py`, then run:
```bash
cd src
python visualize.py
```
The visualization will be saved in `visualization/model_name`.


## Submission Generation

To generate submission for the Kaggle competition, configure the model and checkpoint to load in `configs/config.yaml`, then run:
```bash
cd src
python generate_predictions.py
```
The predictions of test set will be saved in `predictions`, and the encoded result will be saved as `predictions.csv` in the project directory.

## Test Run Pipeline

When running the provided `monocular-depth-example-notebook.ipynb`, you should use `data/train_list.txt` and `data/test_list.txt` specified as `local_data_dir` instead of the ones under the official data location, or it runs into error.