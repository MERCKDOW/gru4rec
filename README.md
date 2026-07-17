
# GRU4Rec Pipeline & Training Environment

This repository provides a complete MLOps workflow for training and deploying [Gru4Rec_Pytorch_Official](https://github.com/hidasib/GRU4Rec_Pytorch) models. It encompasses the entire lifecycle from data preparation to automated pipeline scheduling.

## 📁 Repository Structure

```text
gru4rec/
├── Gru4Rec_Pytorch_Official/  # Original repo
├── sql/                       # Queries for dataset creation
├── pipeline/                  # Infrastructure & Training
│   ├── Dockerfile             # Docker environment configuration
│   ├── gru4rec_train.py       # Training script
│   └── gru4rec_config         # Training configuration file
└── notebooks/                 # Master notebook for workflow orchestration
    └── orchestrator.ipynb

🚀 Workflow Overview
The core of this project is the orchestrator.ipynb notebook, which manages the lifecycle of the model from image creation to production scheduling.
| Step | Action | Description |
|---|---|---|
| 1 | Clone | Synchronizes the Gru4Rec_Pytorch_Official repository. |
| 2 | Dockerize | Builds the Docker image (using /pipeline/Dockerfile) and pushes to the registry. |
| 3 | Custom Job | Defines and executes a one-off custom training job. |
| 4 | Pipeline | Builds and executes the full training/deployment pipeline. |
| 5 | Schedule | Configures and enables automated pipeline scheduling. |
🛠 Components
1. GRU4Rec Implementation
We utilize the Gru4Rec_Pytorch_Official repository as the engine for our session-based recommendations.
2. Dataset Creation (/sql)
Located in the /sql directory, these queries define the feature engineering and data extraction processes required to generate the input tensors for the GRU4Rec model.
3. Training Routine (/pipeline)
The /pipeline directory contains the core training logic and environment configuration:
 * gru4rec_train.py: The main execution script for model training.
 * gru4rec_config: Configuration file for model hyperparameters and settings.
 * Dockerfile: Defines the environment with all dependencies (PyTorch, CUDA, etc.) to ensure reproducible training runs.
Built for end-to-end ML production.

