
from utils.data import *
from utils.cnn_utils import *
from utils.mlp_utils import fetch_mlp_params_from_file
from utils.logger import load_config
from models.cnn import CNN
import argparse

parser = argparse.ArgumentParser(description='Set run options')
parser.add_argument('--config', type=str, help='Specify config location')
parser.add_argument('--mode', type=str, help='Specify "train" or "test"')
parser.add_argument('--pretrained', action="store_true", help='Specify if model is pretrained')
args = parser.parse_args()


config = load_config(args.config)
log_id = config['log_id']

specs = config["specs"]
multi_out = specs["multi_out"]
device_type = specs["device_type"] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = args.mode #mode = specs["mode"]
pretrained = args.pretrained #specs["pretrained"]
input_data_dim = tuple(specs["input_data_dim"])
color_channels, img_height, img_width = input_data_dim

parameters = config["parameters"]
cnn_parameters_fpath = parameters["cnn_parameters_fpath"]
mlp_parameters_fpath = parameters["mlp_parameters_fpath"]
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
    
    img_batch, label_batch = gen_pet_img_stack(
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
            cnn_model_params=fetch_cnn_params_from_file(device_type, cnn_parameters_fpath),
            mlp_model_params=fetch_mlp_params_from_file(device_type, mlp_parameters_fpath),
            cnn_save_fpath=cnn_parameters_fpath,
            mlp_save_fpath=mlp_parameters_fpath,
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
            input_data_dim=input_data_dim,
            cnn_save_fpath=cnn_parameters_fpath,
            mlp_save_fpath=mlp_parameters_fpath,
        )

    epoch_plt, loss_plt = cnn.train(
        data=img_batch, 
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

    img_batch, label_batch = gen_pet_img_stack(
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
        cnn_model_params=fetch_cnn_params_from_file(device_type, parameters["cnn_parameters_fpath"]),
        mlp_model_params=fetch_mlp_params_from_file(device_type, parameters["mlp_parameters_fpath"]),
    )

    prediction_batch = cnn.inference(img_batch)

    if multi_out:
        pass
        # print_pet_inference_resultsMultiOut(
        #     dataset_size=test_dataset_size,
        #     img_batch=img_batch,
        #     label_batch=label_batch,
        #     prediction_batch=prediction_batch,
        #     color_channels=color_channels,
        #     show_results=show_results
        # )
    else:
        print_pet_inference_results(
            dataset_size=test_dataset_size,
            img_batch=img_batch,
            label_batch=label_batch,
            prediction_batch=prediction_batch,
            color_channels=color_channels,
            show_images=show_results
        )


