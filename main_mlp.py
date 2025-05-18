import torch
from utils.data import *
from models.mlp import MLP



config = load_config("config/mlp.yml")

specs = config["specs"]
input_feature_count = specs["input_feature_count"]
device_type = specs["device_type"] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = specs["mode"]
pretrained = specs["pretrained"]

parameters = config["parameters"]
architecture = config["architecture"]


# Training mode
if mode == "train":
    train_config = config["train"]
    train_dataset_size = train_config["train_dataset_size"]
    epochs = train_config["epochs"]
    show_plot = train_config["show_loss_plot"]
    hyperparameters = config["hyperparameters"]
    mlp_hyperparameters = hyperparameters["mlp_hyperparameters"]
    
    (dataBatch, labelBatch) = genMatrixStack(train_dataset_size, int(input_feature_count**(1/2)))
    # print(dataBatch.shape)
    # print(labelBatch.shape)

    if pretrained:
        mlp = MLP(
            pretrained=True,
            training=True,
            device_type=device_type,
            hyperparameters=mlp_hyperparameters,
            mlp_model_params=fetchMLPParametersFromFile(device_type, parameters["mlp_parameters_fpath"])
        )
    else:
        mlp = MLP(
            pretrained=False,
            training=True,
            device_type=device_type,
            hyperparameters=mlp_hyperparameters,
            mlp_architecture=architecture["mlp_architecture"],
            input_feature_count=input_feature_count
        )

    (epochPlt, lossPlt) = mlp.train(dataBatch, labelBatch, epochs=epochs, save_params=True)
    if epochPlt and show_plot:
        plotTrainingResults(epochPlt, lossPlt)

# Testing mode
else:
    test_config = config["test"]
    test_dataset_size = test_config["test_dataset_size"]
    show_images = test_config["show_results"]

    (dataBatch, labelBatch) = genMatrixStack(test_dataset_size, int(input_feature_count**(1/2)))

    mlp = MLP(
        pretrained=True,
        training=False,
        device_type=device_type,
        mlp_model_params=fetchMLPParametersFromFile(device_type, parameters["mlp_parameters_fpath"]),
    )

    predictionBatch = mlp.inference(dataBatch)
    printInferenceResults(test_dataset_size, predictionBatch, labelBatch)

