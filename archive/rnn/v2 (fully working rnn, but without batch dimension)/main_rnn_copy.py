import torch
from utils.data import *
from utils.rnn_utils import *
from models.rnn_copy import RNN
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
input_feature_count = specs["input_feature_count"]

parameters = config["parameters_fpath"]
architecture = config["architecture"]

freq = 1
amp = 1
time_steps = 100
T = 10*np.pi

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
            model_params=fetchRNNParametersFromFile(device_type, parameters),
            stateful=stateful
            # time_steps=time_steps,
        )
        
    else:
        rnn = RNN(
            pretrained=False,
            training=True,
            device_type=device_type,
            hyperparameters=hyperparameters,
            architecture=architecture,
            input_feature_count=input_feature_count,
            stateful=stateful
            # time_steps=time_steps,
        )
        
        
    t, X = genSineWave(time_steps, freq, amp, T, add_noise=False)
    data_batch = t
    label_batch = X
    # plt.plot(data_batch.squeeze(), label_batch.squeeze())
    # plt.show()

    (epoch_plt, loss_plt) = rnn.train(data_batch, label_batch, epochs, save_params=True)
    if epoch_plt and show_plot:
        plotTrainingResults(epoch_plt, loss_plt)

# Testing mode

else:
    test_config = config["test"]
    test_dataset_size = test_config["test_dataset_size"]

    t, X = genSineWave(time_steps, freq, amp, T, add_noise=False)
    data_batch = t
    label_batch = X

    rnn = RNN(
        pretrained=True,
        training=False,
        device_type=device_type,
        model_params=fetchRNNParametersFromFile(device_type, parameters),
        stateful=stateful
        # time_steps=time_steps,
    )

    prediction_batch = rnn.inference(data_batch).squeeze().detach()
    printRegressionResults(prediction_batch, label_batch)
    # print(prediction_batch)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data_batch.squeeze(), label_batch.squeeze(), label='Ground Truth', color='C0')
    ax2.plot(data_batch.squeeze(), prediction_batch, label='Prediction', color='C1')
    ax1.set_ylabel('Ground Truth')
    ax2.set_ylabel('Prediction')
    fig.legend()
    plt.show()



