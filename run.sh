#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if a model name was provided
if [ -z "$1" ]; then
  echo "Error: No model name provided."
  echo "Usage: $0 <model_name> [pretrained]"
  exit 1
fi

MODEL_NAME="$1"
CONFIG_PATH="config/${MODEL_NAME}.yml"
PRETRAINED_FLAG=""

# Check if the second argument is 'pretrained'
if [ "$2" == "pretrained" ]; then
  PRETRAINED_FLAG="--pretrained"
fi

echo "Starting ${MODEL_NAME} training..."
python3 -m main.main_${MODEL_NAME} --config "${CONFIG_PATH}" --mode train ${PRETRAINED_FLAG}

echo "Training completed. Starting ${MODEL_NAME} testing..."
python3 -m main.main_${MODEL_NAME} --config "${CONFIG_PATH}" --mode test

echo "Testing completed."
