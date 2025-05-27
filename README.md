# Neural Network Library

A modular implementation of various neural network architectures from scratch using PyTorch tensors. This project focuses on demonstrating the underlying mechanics of each architecture with an emphasis on educational clarity.

## Overview

This project provides implementations of different neural network models including:

- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)

The models are designed to be modular and extensible.

## Models

### Multi-Layer Perceptron (MLP)
- Fully connected feed-forward neural network.
- Customizable architecture with variable hidden layers.
- Support for different activation functions.
- Example: Matrix symmetry classification.

### Convolutional Neural Network (CNN)
- Custom implementation of convolutional and pooling layers.
- Integrated with MLP for the fully connected layers.
- Example: Pet (cat/dog) image classification.

### Recurrent Neural Network (RNN)
- Capable of processing sequential data and time series.
- Supports stateful and stateless training/inference.
- Implemented with auto-regressive capabilities.
- Example: Sine wave prediction, text generation.

### Long Short-Term Memory (LSTM)
- An advanced type of RNN, designed to better capture long-range dependencies.
- Implemented with auto-regressive capabilities.
- Example: Sine wave prediction (see `config/auto_regressive_lstm.yml` and `main/main_auto_regressive_lstm.py`).

## Project Structure

- **`archive/`**: Contains older versions or experimental code.
- **`backup/`**: Stores backups of training runs, logs, and model weights for good performing models.
- **`config/`**: YAML configuration files for each model (e.g., [`auto_regressive_lstm.yml`](config/auto_regressive_lstm.yml), [`rnn.yml`](config/rnn.yml)).
- **`data/`**: Datasets used for training and testing (e.g., `pets/`, `text/`).
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

## Installation

1.  Clone the repository.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running Models

There are two primary ways to run the models:

1.  **Using individual main scripts:**

    Each model has its own main script in the `main/` directory. You can run them directly, specifying the mode (train/test) and other options. Configuration for these scripts is typically managed through corresponding YAML files in the `config/` directory.

    ```bash
    # Example: Train a new MLP model
    python main/main_mlp.py --mode train

    # Example: Test with a pretrained CNN model
    python main/main_cnn.py --mode test --pretrained

    # Example: Train an auto-regressive LSTM
    python main/main_auto_regressive_lstm.py --mode train
    ```
    Refer to the specific `main_<model_type>.py` script and its associated `config/<model_type>.yml` file for more details on configurable parameters.

2.  **Using the `run.sh` script:**

    The [`run.sh`](run.sh) script provides a convenient way to execute training and inference. It typically handles the selection of the correct main script and configuration file.

    ```bash
    # Example: Run the rnn model (assumes rnn.yml config exists)
    ./run.sh rnn
    ```
    Ensure the script has execute permissions (`chmod +x run.sh`). The script expects the model name as an argument, which corresponds to the main script file (e.g., `rnn` for `main_rnn.py`) and its configuration file (`rnn.yml`).

## Logging and Backups
- Training progress, including loss curves and configuration files, is logged in the `logs/` directory.
- Successful training runs and model checkpoints are often backed up in the `backup/` directory.

## Development Notes
- For a detailed history of changes and features, see [`changes.txt`](changes.txt).
- For planned features and tasks, refer to [`todo.txt`](todo.txt).