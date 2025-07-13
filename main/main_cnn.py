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

system = config["system"]
device = system.get("device")
cnn_save_fpath = system.get("save_fpath")
mlp_save_fpath = system.get("mlp_save_fpath")

specifications = config["specifications"]
multi_out = specifications["multi_out"]

architecture = config["architecture"]
cnn_architecture = architecture["cnn_architecture"]
mlp_architecture = architecture["mlp_architecture"]
color_channels, img_height, img_width = tuple(cnn_architecture["input_data_dim"])

mode = args.mode
pretrained = args.pretrained


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
        dataset_size=train_dataset_size, 
        use="train",
        device=device,
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
            device=device,
            params=fetch_cnn_params_from_file(device, cnn_save_fpath),
            mlp_params=fetch_mlp_params_from_file(device, mlp_save_fpath),
            hyperparameters=cnn_hyperparameters,
            mlp_hyperparameters=mlp_hyperparameters,
            save_fpath=cnn_save_fpath,
            mlp_save_fpath=mlp_save_fpath,
            specifications=specifications
        )
    else:
        cnn = CNN(
            pretrained=False,
            training=True,
            device=device,
            architecture=cnn_architecture,
            mlp_architecture=mlp_architecture,
            hyperparameters=cnn_hyperparameters,
            mlp_hyperparameters=mlp_hyperparameters,
            save_fpath=cnn_save_fpath,
            mlp_save_fpath=mlp_save_fpath,
            specifications=specifications
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
        dataset_size=test_dataset_size,
        use="test",
        device=device,
        img_height=img_height,
        img_width=img_width,
        color_channels=color_channels,
        multi_out=multi_out,
        save=False
    )

    cnn = CNN(
        pretrained=True,
        training=False,
        device=device,
        params=fetch_cnn_params_from_file(device, cnn_save_fpath),
        mlp_params=fetch_mlp_params_from_file(device, mlp_save_fpath),
        specifications=specifications
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
        