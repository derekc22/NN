import torch
from utils.data import *
from src.recurrent_cell import RecurrentCell
from models.rnn import RNN

# torch.manual_seed(42)

config = load_config(args.config)

specs = config["specs"]
device_type = specs["device_type"] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = specs["mode"]
pretrained = specs["pretrained"]
input_feature_count = specs["input_feature_count"]

parameters = config["parameters_fpath"]
architecture = config["architecture"]


# Training mode
if mode == "train":
    train_config = config["train"]
    train_dataset_size = train_config["train_dataset_size"]
    epochs = train_config["epochs"]
    show_plot = train_config["show_loss_plot"]
    hyperparameters = config["hyperparameters"]

    if pretrained:
        pass
        # rnn = RNN(
        #     pretrained=True,
        #     training=True,
        #     device_type=device_type,
        #     hyperparameters=hyperparameters,
        #     model_params=fetchMLPParametersFromFile(device_type, parameters)
        # )
        
    else:

        rnn = RNN(
            pretrained=False,
            training=True,
            device_type=device_type,
            hyperparameters=hyperparameters,
            architecture=architecture,
            input_feature_count=input_feature_count
        )
        X = torch.rand(architecture["time_steps"], specs["input_feature_count"])
        print(rnn.layers[0].feed(X))

    # (epoch_plt, loss_plt) = rnn.train(data_batch, label_batch, epochs=epochs, save_params=True)
    # if epoch_plt and show_plot:
    #     plotTrainingResults(epoch_plt, loss_plt)

# # Testing mode
# else:
#     test_config = config["test"]
#     test_dataset_size = test_config["test_dataset_size"]
#     show_images = test_config["show_results"]

#     (data_batch, label_batch) = genMatrixStack(test_dataset_size, int(input_feature_count**(1/2)))

#     mlp = MLP(
#         pretrained=True,
#         training=False,
#         device_type=device_type,
#         model_params=fetchMLPParametersFromFile(device_type, parameters),
#     )

#     prediction_batch = mlp.inference(data_batch)
#     printClassificationResults(test_dataset_size, prediction_batch, label_batch)




