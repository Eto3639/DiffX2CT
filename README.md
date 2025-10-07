# 3D CT Image Generation from 2D DRRs using Diffusion Model

This project implements a diffusion model to generate 3D CT scan volumes from a pair of 2D Digitally Reconstructed Radiographs (DRRs). The model leverages a sophisticated U-Net architecture with a conditioning encoder and is designed for distributed training across multiple GPUs.

## Key Features

- **Diffusion Model:** Generates high-quality 3D CT volumes from 2D DRR inputs.
- **Model Parallelism:** The U-Net model is distributed across three GPUs to handle the large memory requirements of 3D volumetric data processing.
- **Hyperparameter Optimization with Optuna:** Integrated with Optuna for automated hyperparameter searching to find the best settings for learning rate, loss ratios, gradient clipping, and more.
- **Robust and Resumable Training:**
    - **Resumable Studies:** Training progress is saved to a SQLite database, allowing hyperparameter searches to be resumed if interrupted.
    - **Gradient Clipping:** Implements dynamic gradient clipping to prevent gradient explosion and stabilize training.
    - **Trial Pruning:** Automatically stops and prunes Optuna trials that result in NaN/Inf loss, saving computational resources.
- **Configuration-driven:** All major settings, including data paths, model choices, and Optuna search spaces, are managed via a central `config.yml` file.

## Project Structure

```
.
├── DiffX2CT/
│   ├── train.py                # Main script for training with Optuna
│   ├── visualization.py        # Script to generate visualizations from a checkpoint
│   ├── models.py               # DistributedUNet wrapper for model parallelism
│   ├── custom_models/
│   │   ├── unet.py             # Core U-Net model architecture
│   │   └── conditioning_encoder.py # Encoder for DRR conditions
│   ├── data_utils.py           # PyTorch Dataset and data loading utilities
│   └── utils.py                # Utility functions (e.g., config loading)
├── config.yml                  # Central configuration file for all settings
├── run_training.sh             # Shell script to run the training in a Docker container
└── run_visualize.sh            # Shell script to run visualization in a Docker container
```

## Setup

### 1. Docker Environment
This project is designed to run inside a Docker container. Ensure you have Docker and the NVIDIA Container Toolkit installed. The required Docker image is specified in the `run_*.sh` scripts (e.g., `monai_env:v2.3`).

### 2. Weights & Biases (W&B)
The project uses W&B for logging and visualization.
- Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).
- Open `run_training.sh` and `run_visualize.sh` and set your `WANDB_API_KEY`.

```bash
# In run_training.sh and run_visualize.sh
WANDB_API_KEY="YOUR_WANDB_API_KEY"
```

### 3. Data
Update the `config.yml` file to point to the directories containing your preprocessed CT and DRR data.

```yaml
# In config.yml
DATA:
  PT_DATA_DIR: "/path/to/your/CT_pt_files"
  DRR_DIR: "/path/to/your/DRR_mod_files"
```

## Usage

### Training with Optuna
To start the hyperparameter search, run the `run_training.sh` script. You can specify a name for your study, which allows you to resume it later if it gets interrupted.

```bash
# Start a new study or resume an existing one named "my-first-study"
./run_training.sh --study_name my-first-study
```

- The script will execute `DiffX2CT/train.py` inside the Docker container.
- Optuna will create a `my-first-study.db` file in the project root to store the study's progress.
- Training checkpoints for each trial will be saved in the `checkpoints/trial_<number>/` directory.
- At the end of the study, the best hyperparameters found will be printed to the console.

### Visualization
To generate and save visualizations from a trained model checkpoint, use the `run_visualize.sh` script. You need to provide the path to the checkpoint directory of a specific trial.

```bash
# Example: Visualize the model from trial 0
./run_visualize.sh --checkpoint_dir ./checkpoints/trial_0/
```

- The script executes `DiffX2CT/visualization.py`.
- It will load the model weights from the specified directory.
- The generated MPR (Multi-Planar Reconstruction) images will be saved in the checkpoint directory under a `visualizations/` subfolder.
- The output will also be logged to W&B.