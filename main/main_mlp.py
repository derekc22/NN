import torch
from utils.data import *
from utils.mlp_utils import *
from utils.logger import load_config
from models.mlp import MLP
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
    
    data_batch, label_batch = gen_matrix_stack(
        train_dataset_size, 
        int(input_feature_count**(1/2))
    )

    if pretrained:
        mlp = MLP(
            pretrained=True,
            training=True,
            device=device,
            params=fetch_mlp_params_from_file(device, save_fpath),
            hyperparameters=hyperparameters,
            save_fpath=save_fpath,
            specifications=specifications
        )
    else:
        mlp = MLP(
            pretrained=False,
            training=True,
            device=device,
            architecture=architecture,
            hyperparameters=hyperparameters,
            save_fpath=save_fpath,
            specifications=specifications
        )

    epoch_plt, loss_plt = mlp.train(
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

    data_batch, label_batch = gen_matrix_stack(test_dataset_size, int(input_feature_count**(1/2)))

    mlp = MLP(
        pretrained=True,
        training=False,
        device=device,
        params=fetch_mlp_params_from_file(device, save_fpath),
        specifications=specifications
    )

    prediction_batch = mlp.inference(data_batch)
    print_classification_results(test_dataset_size, prediction_batch, label_batch)

