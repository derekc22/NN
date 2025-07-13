import torch
from utils.data import *
from utils.rnn_utils import *
from utils.lstm_utils import *
from utils.logger import load_config
from models.lstm import LSTM
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Set run options')
parser.add_argument('--config', type=str, help='Specify config location')
parser.add_argument('--mode', type=str, help='Specify "train" or "test"')
parser.add_argument('--pretrained', action="store_true", help='Specify if model is pretrained')
args = parser.parse_args()

config = load_config(args.config)
log_id = config['log_id']

system = config["system"]
device = system["device"]
save_fpath = system["save_fpath"]

specifications = config["specifications"]
stateful = specifications["stateful"]
autoregressive = specifications["autoregressive"]

architecture = config["architecture"]

mode = args.mode
pretrained = args.pretrained

freq = 5
amp = 1
time_steps = 10
T = 2*np.pi

# Training mode
if mode == "train":
    train_config = config["train"]
    train_dataset_size = train_config["train_dataset_size"]
    epochs = train_config["epochs"]
    show_plot = train_config["show_loss_plot"]
    hyperparameters = config["hyperparameters"]

    if pretrained:
        lstm = LSTM(
            pretrained=True,
            training=True,
            device=device,
            params=fetch_lstm_params_from_file(device, save_fpath),
            hyperparameters=hyperparameters,
            save_fpath=save_fpath,
            specifications=specifications
        )
        
    else:
        lstm = LSTM(
            pretrained=False,
            training=True,
            device=device,
            architecture=architecture,
            hyperparameters=hyperparameters,
            save_fpath=save_fpath,
            specifications=specifications
        )
        
    t, X = gen_sine_wave(time_steps, freq, amp, T, train_dataset_size, vary_dt=False, vary_phase=False, add_noise=False)
    # t, X = gen_decaying_sine_wave(time_steps, -1, amp, T, train_dataset_size, vary_dt=False, vary_phase=False, add_noise=False)
    data_batch = X[:, :-1, :]
    label_batch = X[:, 1:, :]

    epoch_plt, loss_plt = lstm.train(
        data=data_batch, 
        target=label_batch, 
        epochs=epochs, 
        save_params=True
    )
    
    if epoch_plt and show_plot:
        plot_training_results(epoch_plt, loss_plt, log_id)

# Testing mode

else:
    test_config = config["test"]
    test_dataset_size = test_config["test_dataset_size"]
    show_results = test_config["show_results"]

    t, X = gen_sine_wave(time_steps, freq, amp, T, test_dataset_size, vary_dt=False, vary_phase=False, add_noise=False)
    # t, X = gen_decaying_sine_wave(time_steps, -1, amp, T, test_dataset_size, vary_dt=False, vary_phase=False, add_noise=False)
    t = t[:, :-1, :]
    data_batch = X[:, :-1, :]
    label_batch = X[:, 1:, :]

    lstm = LSTM(
        pretrained=True,
        training=False,
        device=device,
        params=fetch_lstm_params_from_file(device, save_fpath),
        specifications=specifications
    )

    prediction_batch = lstm.inference(data_batch)
    plot_regression_results(t, prediction_batch, label_batch, log_id, show_results)






