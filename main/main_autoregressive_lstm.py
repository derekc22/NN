import torch
from utils.data import *
from utils.rnn_utils import *
from utils.lstm_utils import *
from utils.logger import load_config
from models.lstm import LSTM
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Set run options')
parser.add_argument('--config', type=str, help='Specify config location')
parser.add_argument('--mode', type=str, help='Specify "train" or "test"')
parser.add_argument('--pretrained', action="store_true", help='Specify if model is pretrained')
args = parser.parse_args()


config = load_config(args.config)
log_id = config['log_id']

specs = config["specs"]
device_type = specs["device_type"] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = args.mode #specs["mode"]
pretrained = args.pretrained #specs["pretrained"]
input_feature_count = specs["input_feature_count"]
# time_steps = specs["time_steps"]
stateful = specs["stateful"]
autoregressive = specs["autoregressive"]
teacher_forcing = specs["teacher_forcing"]

parameters_fpath = config["parameters_fpath"]
architecture = config["architecture"]

freq = 5
amp = 1
time_steps = 150
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
            device_type=device_type,
            hyperparameters=hyperparameters,
            model_params=fetchLSTMParametersFromFile(device_type, parameters_fpath),
            stateful=stateful,
            autoregressive=autoregressive,
            teacher_forcing=teacher_forcing,
            save_fpath=parameters_fpath,
        )
        
    else:
        lstm = LSTM(
            pretrained=False,
            training=True,
            device_type=device_type,
            hyperparameters=hyperparameters,
            architecture=architecture,
            input_feature_count=input_feature_count,
            stateful=stateful,
            autoregressive=autoregressive,
            teacher_forcing=teacher_forcing,
            save_fpath=parameters_fpath,
        )
        
    t, X = genSineWave(time_steps, freq, amp, T, train_dataset_size, vary_dt=False, vary_phase=False, add_noise=False)
    # t, X = genDecayingSineWave(time_steps, -1, amp, T, train_dataset_size, vary_dt=False, vary_phase=False, add_noise=False)
    data_batch = X[:, :-1, :]
    label_batch = X[:, 1:, :]


    (epoch_plt, loss_plt) = lstm.train(data_batch, label_batch, epochs, save_params=True)
    if epoch_plt and show_plot:
        plotTrainingResults(epoch_plt, loss_plt, log_id)

# Testing mode

else:
    test_config = config["test"]
    test_dataset_size = test_config["test_dataset_size"]
    show_results = test_config["show_results"]

    t, X = genSineWave(time_steps, freq, amp, T, test_dataset_size, vary_dt=False, vary_phase=False, add_noise=False)
    # t, X = genDecayingSineWave(time_steps, -1, amp, T, test_dataset_size, vary_dt=False, vary_phase=False, add_noise=False)
    t = t[:, :-1, :]
    data_batch = X[:, :-1, :]
    label_batch = X[:, 1:, :]


    lstm = LSTM(
        pretrained=True,
        training=False,
        device_type=device_type,
        model_params=fetchLSTMParametersFromFile(device_type, parameters_fpath),
        stateful=stateful,
        autoregressive=autoregressive
    )

    prediction_batch = lstm.inference(data_batch)
    plotRegressionResults(t, prediction_batch, label_batch, log_id, show_results)






