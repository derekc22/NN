import torch
from utils.data import *
from utils.rnn_utils import *
from utils.logger import load_config
from models.rnn import RNN
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
input_feature_count = architecture["input_feature_count"]

mode = args.mode
pretrained = args.pretrained

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
            device=device,
            params=fetch_rnn_params_from_file(device, save_fpath),
            hyperparameters=hyperparameters,
            save_fpath=save_fpath,
            specifications=specifications
        )
        
    else:
        rnn = RNN(
            pretrained=False,
            training=True,
            device=device,
            architecture=architecture,
            hyperparameters=hyperparameters,
            save_fpath=save_fpath,
            specifications=specifications
        )

    data_batch = torch.load("data/text/embeddings/data_batch.pth")[:train_dataset_size]
    label_batch = torch.load("data/text/embeddings/label_batch.pth")[:train_dataset_size]
        
    epoch_plt, loss_plt = rnn.train(
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

    rnn = RNN(
        pretrained=True,
        training=False,
        device=device,
        params=fetch_rnn_params_from_file(device, save_fpath),
        specifications=specifications
    )

    data_batch = torch.load("data/text/embeddings/data_batch.pth")[:test_dataset_size]
    prediction_batch = rnn.inference(data_batch)

    decoded_sentence = decode_paragraph(prediction_batch, input_feature_count)
    print(decoded_sentence)







