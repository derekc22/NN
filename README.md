# Neural Network Library

A modular implementation of various neural network architectures from scratch using PyTorch tensors.

## Overview

This project provides implementations of different neural network models including:

- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)

Each model is implemented with a focus on modularity, extensibility, and educational clarity. The code demonstrates the underlying mechanics of each architecture without relying heavily on PyTorch's high-level APIs.

## Models

### Multi-Layer Perceptron (MLP)
- Fully connected feed-forward neural network
- Customizable architecture with variable hidden layers
- Support for different activation functions
- Matrix symmetry classification example

### Convolutional Neural Network (CNN)
- Custom implementation of convolutional and pooling layers
- Integrated with MLP for the fully connected layers
- Pet (cat/dog) image classification example

### Recurrent Neural Network (RNN)
- Time series processing
- Sine wave prediction example
- Sequential data handling

## Project Structure

## Usage

### Running Models

Each model has its own main script and can be run using the following command format:

```bash
python main_<model_type>.py --mode train
python main_<model_type>.py --mode train --pretrained
python main_<model_type>.py --mode test
```


# Train a new MLP model
python main_mlp.py --mode train

# Test with a pretrained CNN model
python main_cnn.py --mode test --pretrained