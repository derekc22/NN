import torch
from utils.data import *
from utils.mlp_utils import *
from models.mlp import MLP
import argparse

parser = argparse.ArgumentParser(description='Run the simulation')
parser.add_argument('--config', type=str, help='Specify config location')
parser.add_argument('--mode', type=str, help='Specify "train" or "test"')
parser.add_argument('--pretrained', action="store_true", help='Specify if model is pretrained')
args = parser.parse_args()

config = load_config(args.config)

specs = config["specs"]
device_type = specs["device_type"] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = mode = args.mode #specs["mode"]
pretrained = args.pretrained #specs["pretrained"]
input_feature_count = specs["input_feature_count"]

parameters_fpath = config["parameters_fpath"]
architecture = config["architecture"]


# Training mode
if mode == "train":
    train_config = config["train"]
    train_dataset_size = train_config["train_dataset_size"]
    epochs = train_config["epochs"]
    show_plot = train_config["show_loss_plot"]
    hyperparameters = config["hyperparameters"]
    
    (data_batch, label_batch) = genMatrixStack(train_dataset_size, int(input_feature_count**(1/2)))
    # print(data_batch.shape)
    # print(label_batch.shape)

    if pretrained:
        mlp = MLP(
            pretrained=True,
            training=True,
            device_type=device_type,
            hyperparameters=hyperparameters,
            model_params=fetchMLPParametersFromFile(device_type, parameters_fpath),
        )
    else:
        mlp = MLP(
            pretrained=False,
            training=True,
            device_type=device_type,
            hyperparameters=hyperparameters,
            architecture=architecture,
            input_feature_count=input_feature_count,
            save_fpath=parameters_fpath,
        )

    (epoch_plt, loss_plt) = mlp.train(data_batch, label_batch, epochs, save_params=True)
    if epoch_plt and show_plot:
        plotTrainingResults(epoch_plt, loss_plt)

# Testing mode
else:
    test_config = config["test"]
    test_dataset_size = test_config["test_dataset_size"]
    show_images = test_config["show_results"]

    (data_batch, label_batch) = genMatrixStack(test_dataset_size, int(input_feature_count**(1/2)))

    mlp = MLP(
        pretrained=True,
        training=False,
        device_type=device_type,
        model_params=fetchMLPParametersFromFile(device_type, parameters_fpath),
    )

    prediction_batch = mlp.inference(data_batch)
    printClassificationResults(test_dataset_size, prediction_batch, label_batch)

