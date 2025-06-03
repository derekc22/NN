import torch
from utils.data import *
from utils.rnn_utils import *
from utils.logger import load_config
from models.rnn import RNN
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Set run options')
parser.add_argument('--config', type=str, help='Specify config location')
parser.add_argument('--mode', type=str, help='Specify "train" or "test"')
parser.add_argument('--pretrained', action="store_true", help='Specify if model is pretrained')
args = parser.parse_args()


config = load_config(args.config)

specs = config["specs"]
device_type = specs["device_type"] 
mode = args.mode 
pretrained = args.pretrained 
input_feature_count = specs["input_feature_count"]
stateful = specs["stateful"]

parameters_fpath = config["parameters_fpath"]
architecture = config["architecture"]


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
            batch_size=train_dataset_size,
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
            batch_size=train_dataset_size,
            save_fpath=parameters_fpath,
        )



    data_batch = torch.load("data/text/embeddings/data_batch.pth")[:train_dataset_size]
    label_batch = torch.load("data/text/embeddings/label_batch.pth")[:train_dataset_size]
        
    (epoch_plt, loss_plt) = rnn.train(data_batch, label_batch, epochs, save_params=True)
    if epoch_plt and show_plot:
        plotTrainingResults(epoch_plt, loss_plt)

# Testing mode

else:
    test_config = config["test"]
    test_dataset_size = test_config["test_dataset_size"]


    rnn = RNN(
        pretrained=True,
        training=False,
        device_type=device_type,
        model_params=fetchRNNParametersFromFile(device_type, parameters_fpath),
        stateful=stateful,
        batch_size=test_dataset_size,
    )

    data_batch = torch.load("data/text/embeddings/data_batch.pth")[:test_dataset_size]
    prediction_batch = rnn.inference(data_batch)

    decoded_sentence = decodeParagraph(prediction_batch, input_feature_count)
    print(decoded_sentence)







