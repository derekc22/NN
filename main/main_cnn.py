
from utils.data import *
from utils.cnn_utils import *
from utils.mlp_utils import fetchMLPParametersFromFile
from models.cnn import CNN
import argparse

parser = argparse.ArgumentParser(description='Run the simulation')
parser.add_argument('--config', type=str, help='Specify config location')
parser.add_argument('--mode', type=str, help='Specify "train" or "test"')
parser.add_argument('--pretrained', action="store_true", help='Specify if model is pretrained')
args = parser.parse_args()


config = load_config(args.config)

specs = config["specs"]
multi_out = specs["multi_out"]
device_type = specs["device_type"] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = args.mode #mode = specs["mode"]
pretrained = args.pretrained #specs["pretrained"]
input_data_dim = tuple(specs["input_data_dim"])
color_channels, img_height, img_width = input_data_dim

parameters = config["parameters"]
architecture = config["architecture"]


# Training mode
if mode == "train":
    train_config = config["train"]
    train_dataset_size = train_config["train_dataset_size"]
    epochs = train_config["epochs"]
    show_plot = train_config["show_loss_plot"]
    hyperparameters = config["hyperparameters"]
    cnn_hyperparameters = hyperparameters["cnn_hyperparameters"]
    mlp_hyperparameters = hyperparameters["mlp_hyperparameters"]
    
    (img_batch, label_batch) = genPetImageStack(
        train_dataset_size, 
        use="train",
        device_type=device_type,
        img_height=img_height,
        img_width=img_width,
        color_channels=color_channels,
        multi_out=multi_out,
        save=False
    )

    if pretrained:
        cnn = CNN(
            pretrained=True,
            training=True,
            device_type=device_type,
            hyperparameters=cnn_hyperparameters,
            mlp_hyperparameters=mlp_hyperparameters,
            cnn_model_params=fetchCNNParametersFromFile(device_type, parameters["cnn_parameters_fpath"]),
            mlp_model_params=fetchMLPParametersFromFile(device_type, parameters["mlp_parameters_fpath"])
        )
    else:
        cnn = CNN(
            pretrained=False,
            training=True,
            device_type=device_type,
            hyperparameters=cnn_hyperparameters,
            mlp_hyperparameters=mlp_hyperparameters,
            cnn_architecture=architecture["cnn_architecture"],
            mlp_architecture=architecture["mlp_architecture"],
            input_data_dim=input_data_dim
        )

    (epoch_plt, loss_plt) = cnn.train(img_batch, label_batch, epochs, save_params=True)
    if epoch_plt and show_plot:
        plotTrainingResults(epoch_plt, loss_plt)

# Testing mode
else:
    test_config = config["test"]
    test_dataset_size = test_config["test_dataset_size"]
    show_images = test_config["show_results"]

    (img_batch, label_batch) = genPetImageStack(
        test_dataset_size,
        use="test",
        device_type=device_type,
        img_height=img_height,
        img_width=img_width,
        color_channels=color_channels,
        multi_out=multi_out,
        save=False
    )

    cnn = CNN(
        pretrained=True,
        training=False,
        device_type=device_type,
        cnn_model_params=fetchCNNParametersFromFile(device_type, parameters["cnn_parameters_fpath"]),
        mlp_model_params=fetchMLPParametersFromFile(device_type, parameters["mlp_parameters_fpath"]),
    )

    prediction_batch = cnn.inference(img_batch)

    if multi_out:
        pass
        # printPetInferenceResultsMultiOut(
        #     dataset_size=test_dataset_size,
        #     img_batch=img_batch,
        #     label_batch=label_batch,
        #     prediction_batch=prediction_batch,
        #     color_channels=color_channels,
        #     show_images=show_images
        # )
    else:
        printPetInferenceResults(
            dataset_size=test_dataset_size,
            img_batch=img_batch,
            label_batch=label_batch,
            prediction_batch=prediction_batch,
            color_channels=color_channels,
            show_images=show_images
        )


