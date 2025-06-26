# Neural Network Library

A modular implementation of various neural network architectures from scratch using PyTorch tensors. This project focuses on demonstrating the underlying mechanics of each architecture with an emphasis on educational clarity.

## Overview

This project provides implementations of different neural network models including:

- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Transformer [Work in Progress]

## Models

### Multi-Layer Perceptron (MLP)
- **Architecture**: Fully connected feed-forward neural network with customizable hidden layers
- **Features**: Variable layer sizes, multiple activation function support
- **Use Cases**: Tabular data classification, regression tasks, function approximation
- **Example**: `python main/main_mlp.py --mode train` (stress tensor classification)

### Convolutional Neural Network (CNN)
- **Architecture**: Custom convolutional and pooling layers integrated with fully connected layers
- **Features**: Feature extraction through convolution, spatial hierarchy learning
- **Use Cases**: Image recognition, object detection, medical imaging analysis
- **Example**: `python main/main_cnn.py --mode train` (pet image classification)

### Recurrent Neural Network (RNN)
- **Architecture**: Sequential processing network with memory capabilities
- **Features**: Stateful/stateless training, autoregressive capabilities, temporal dependency modeling
- **Use Cases**: Text generation, stock price prediction, sensor data analysis
- **Example**: `python main/main_rnn.py --mode train` (sine wave prediction)

### Long Short-Term Memory (LSTM)
- **Architecture**: RNN variant with gating mechanisms for long-range dependencies
- **Features**: Forget/input/output gates, autoregressive capabilities, improved gradient flow
- **Use Cases**: Language modeling, speech recognition, anomaly detection in time series
- **Example**: `python main/main_auto_regressive_lstm.py --mode train` (sine wave prediction)

### Transformer (Work in Progress)
- **Architecture**: Attention-based architecture with self-attention mechanisms
- **Features**: Parallel processing, multi-head attention, positional encoding
- **Use Cases**: Machine translation, text summarization
- **Example**: Currently under development

## Project Structure

- **`archive/`**: Contains older versions or experimental code.
- **`backup/`**: Stores backups of training runs, logs, and model weights for good performing models.
- **`config/`**: YAML configuration files for each model (e.g., [`auto_regressive_lstm.yml`](config/auto_regressive_lstm.yml), [`rnn.yml`](config/rnn.yml)).
- **`data/`**: Toy datasets used for training and testing examples (e.g., `pets/`, `text/`).
- **`documents/`**: Project-related documents.
- **`figures/`**: Saved plots and figures from model training or analysis.
- **`logs/`**: Logging directory, including training loss curves and configuration snapshots.
- **`main/`**: Main executable scripts for training and testing models (e.g., [`main_auto_regressive_lstm.py`](main/main_auto_regressive_lstm.py)).
- **`models/`**: Saved model weights and architectures.
- **`params/`**: Potentially for hyperparameter sets or specific model parameters.
- **`src/`**: Source code for the neural network layers, activations, and core logic.
- **`utils/`**: Utility scripts for data processing, logging, plotting, etc.
- **[`changes.txt`](changes.txt)**: A log of significant changes and development milestones.
- **[`todo.txt`](todo.txt)**: A list of tasks and planned features.
- **[`requirements.txt`](requirements.txt)**: Python package dependencies.
- **[`run.sh`](run.sh)**: A bash script to simplify running training and inference.
- **[`train.sh`](train.sh)**: Dedicated training script for streamlined model training.
- **[`test.sh`](test.sh)**: Dedicated testing script for model evaluation and inference.

## Installation

1.  Clone the repository.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running Models

There are multiple ways to run the models:

1.  **Using individual main scripts:**

    Each model has its own main script in the `main/` directory. You can run them directly, specifying the mode (train/test) and other options. Configuration for these scripts is managed through corresponding YAML files in the `config/` directory.

    ```bash
    # Example: Train a new MLP model
    python main/main_mlp.py --mode train

    # Example: Test with a pretrained CNN model
    python main/main_cnn.py --mode test --pretrained

    # Example: Train an autoregressive LSTM
    python main/main_auto_regressive_lstm.py --mode train
    ```
    Refer to the specific `main_<model_type>.py` script and its associated `config/<model_type>.yml` file for more details on configurable parameters.

2.  **Using the shell scripts:**

    - **[`run.sh`](run.sh)**: Runs both training and testing sequentially
    ```bash
    # Usage: ./run.sh <model_name> [pretrained]
    
    # Example: Run the rnn model (assumes rnn.yml config exists)
    ./run.sh rnn
    
    # Example: Run with pretrained weights
    ./run.sh rnn pretrained
    ```
    
    - **[`train.sh`](train.sh)**: Runs training only
    ```bash
    # Usage: ./train.sh <model_name> [pretrained]
    
    # Example: Train a specific model
    ./train.sh lstm
    
    # Example: Train using pretrained weights
    ./train.sh lstm pretrained
    ```
    
    - **[`test.sh`](test.sh)**: Runs testing/inference only
    ```bash
    # Usage: ./test.sh <model_name>
    
    # Example: Test a trained model
    ./test.sh cnn
    ```
    
    **Arguments:**
    - `<model_name>` (required): The model type to run (e.g., `rnn`, `lstm`, `cnn`, `mlp`)
    - `pretrained` (optional): Use pretrained weights for training (only available for `run.sh` and `train.sh`)
    
    Ensure the scripts have execute permissions (`chmod +x *.sh`). The scripts expect the model name as an argument, which corresponds to the main script file and its configuration file.

## Logging and Backups
- Training progress, including loss curves and configuration files, is logged in the `logs/` directory.

## Development Notes
- For a loosely maintained history of changes and features, see [`changes.txt`](changes.txt).
- For a loosely maintained list of planned features and tasks, refer to [`todo.txt`](todo.txt).