# neural-net

Neural network models implemented from scratch in Python

## Setup

- Clone the repository
  ```bash
  git clone git@github.com:derekc22/neural-net.git
  cd neural-net
  ```

- Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

## Models

- Multi-layer perceptron
- Convolutional neural network
- Recurrent neural network
- Long short-term memory network
- Transformer, work in progress

Planned:
- Physics-informed neural network
- Residual network
- Vision transformer

## Usage

- Model configuration files are stored in `config/`
- Main scripts are stored in `main/`
- Model implementations are stored in `models/`
- Core layers, activations, and utilities are stored in `src/`

## Run

- Train an MLP
  ```bash
  python main/main_mlp.py --mode train
  ```

- Test a pretrained CNN
  ```bash
  python main/main_cnn.py --mode test --pretrained
  ```

- Train an autoregressive LSTM
  ```bash
  python main/main_auto_regressive_lstm.py --mode train
  ```

## Scripts

- Run training and testing
  ```bash
  ./run.sh <model_name> [pretrained]
  ```

- Run training only
  ```bash
  ./train.sh <model_name> [pretrained]
  ```

- Run testing only
  ```bash
  ./test.sh <model_name>
  ```

Arguments:
- `<model_name>`: Model to run, e.g. `mlp`, `cnn`, `rnn`, or `lstm`
- `pretrained`: Optional flag for loading saved weights where supported

## Output

- Training logs are saved in `logs/`
- Model parameters are saved in `params/`
- Figures are saved in `figures/`
- Backup runs are saved in `backup/`

## Notes

- `changes.txt` contains a loose development history
- `todo.txt` contains planned work
