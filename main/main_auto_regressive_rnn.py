import torch
from utils.data import *
from utils.rnn_utils import *
from models.rnn import RNN
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Run the simulation')
parser.add_argument('--config', type=str, help='Specify config location')
parser.add_argument('--mode', type=str, help='Specify "train" or "test"')
parser.add_argument('--pretrained', action="store_true", help='Specify if model is pretrained')
args = parser.parse_args()


config = load_config(args.config)

specs = config["specs"]
device_type = specs["device_type"] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = args.mode #specs["mode"]
pretrained = args.pretrained #specs["pretrained"]
input_feature_count = specs["input_feature_count"]
# time_steps = specs["time_steps"]
stateful = specs["stateful"]
auto_regressive = specs["auto_regressive"]
teacher_forcing = specs["teacher_forcing"]

parameters_fpath = config["parameters_fpath"]
architecture = config["architecture"]

freq = 2
amp = 1
time_steps = 500
T = 2*np.pi

# Training mode
if mode == "train":
    train_config = config["train"]
    train_dataset_size = train_config["train_dataset_size"]
    epochs = train_config["epochs"]
    show_plot = train_config["show_loss_plot"]
    hyperparameters = config["hyperparameters"]

    if pretrained:
        rnn = RNN(
            pretrained=True,
            training=True,
            device_type=device_type,
            hyperparameters=hyperparameters,
            model_params=fetchRNNParametersFromFile(device_type, parameters_fpath),
            stateful=stateful,
            # num_sequences=train_dataset_size,
            auto_regressive=auto_regressive,
            teacher_forcing=teacher_forcing,
        )
        
    else:
        rnn = RNN(
            pretrained=False,
            training=True,
            device_type=device_type,
            hyperparameters=hyperparameters,
            architecture=architecture,
            input_feature_count=input_feature_count,
            stateful=stateful,
            # num_sequences=train_dataset_size,
            auto_regressive=auto_regressive,
            teacher_forcing=teacher_forcing,
            save_fpath=parameters_fpath,
        )
        
    t, X = genSineWave(time_steps, freq, amp, T, train_dataset_size, vary_dt=False, vary_phase=True, add_noise=True)
    # t, X = genDecayingSineWave(time_steps, -1, amp, T, train_dataset_size, vary_dt=False, vary_phase=True, add_noise=True)
    # data_batch = torch.empty_like(X)
    # data_batch[:, 0, :] = X[:, 0, :]
    data_batch = X
    label_batch = X


    (epoch_plt, loss_plt) = rnn.train(data_batch, label_batch, epochs, save_params=True)
    if epoch_plt and show_plot:
        plotTrainingResults(epoch_plt, loss_plt)

# Testing mode

else:
    test_config = config["test"]
    test_dataset_size = test_config["test_dataset_size"]

    t, X = genSineWave(time_steps, freq, amp, T, test_dataset_size, vary_dt=False, vary_phase=True, add_noise=True)
    # t, X = genDecayingSineWave(time_steps, -1, amp, T, test_dataset_size, vary_dt=False, vary_phase=True, add_noise=True)
    # data_batch = torch.empty_like(X)
    # data_batch[:, 0, :] = X[:, 0, :]
    data_batch = X
    label_batch = X

    rnn = RNN(
        pretrained=True,
        training=False,
        device_type=device_type,
        model_params=fetchRNNParametersFromFile(device_type, parameters_fpath),
        stateful=stateful,
        # num_sequences=test_dataset_size,
        auto_regressive=auto_regressive
    )

    prediction_batch = rnn.inference(data_batch)#.detach()
    printRegressionResults(t, prediction_batch, label_batch)






