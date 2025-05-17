import torch
from utils.data import *
from models.cnn import CNN
import argparse






def setArgs():
    parser = argparse.ArgumentParser(description="Set CNN arguments")

    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--pretrained", action="store_true", help="Specify if model used for training is pretrained")
    parser.add_argument("--dataset", type=int, help="Set dataset size")
    parser.add_argument("--epochs", type=int, help="Set number of epochs")
    parser.add_argument("--plot", action="store_true", help="Show training plot")

    parser.add_argument("--show", action="store_true", help="Show results as images")

    args = parser.parse_args()
    return args



args = setArgs()
multiOut = False
imgHeight = 64
imgWidth = 64
colorChannels = 1
deviceType = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if not args.train:

    datasetSize = 50


    (imgBatch, labelBatch) = genPetImageStack(datasetSize, use="test", device_type=deviceType, img_height=imgHeight, img_width=imgWidth,color_channels=colorChannels, multi_out=multiOut, save=False)
    # (imgBatch, labelBatch) = loadPetImageStack()

    cnn = CNN(pretrained=True, training=False, device_type=deviceType, cnn_model_params=fetchCNNParametersFromFile(deviceType, "./params/paramsCNN"), mlp_model_params=fetchMLPParametersFromFile(deviceType, "./params/paramsMLP"))

    predictionBatch = cnn.inference(imgBatch)

    if multiOut:
        printPetInferenceResultsMultiout(dataset_size=datasetSize, img_batch=imgBatch, label_batch=labelBatch, prediction_batch=predictionBatch, color_channels=colorChannels, show_images=args.show)
    else:
        printPetInferenceResults(dataset_size=datasetSize, img_batch=imgBatch, label_batch=labelBatch, prediction_batch=predictionBatch, color_channels=colorChannels, show_images=args.show)



elif args.train:

    isConvLayer =          [True, False, True, False]
    filterCounts =         [2, 2, 4, 4] # THE NUMBER OF FILTERS IN A POOLING KERNEL MUST MATCH THE NUMBER OF FILTERS IN THE PRECEEDING CONVOLUTIONAL LAYER KERNEL
    kernelShapes =         [(5, 5), (2, 2), (3, 3), (2, 2)]
    kernelStrides =        [1, 2, 1, 2]
    CNNactivationFunctions =  ["leakyReLU", "none", "leakyReLU", "none"]

    neuronCounts =        [1]
    MLPactivationFunctions = ["sigmoid"]

    CNNmodelConfig = {
        "is_conv_layer": isConvLayer,
        "filter_counts": filterCounts,
        "kernel_shapes": kernelShapes,
        "kernel_strides": kernelStrides,
        "CNN_activation_functions": CNNactivationFunctions
    }
    MLPmodelConfig = {
        "neuron_counts": neuronCounts,
        "MLP_activation_functions": MLPactivationFunctions
    }

    CNNHyperParameters = {
        "learn_rate": 0.01,
        "batch_size": 24,
        "loss_func": "BCELoss",
        "reduction": "mean",
        "optimizer": "adam",
        "lambda_L2": 1e-6,
        "dropout_rate": None
    }
    MLPHyperParameters = {
        "learn_rate": 0.01,
        "optimizer": "adam",
        "lambda_L2": 1e-6,
        "dropout_rate": None
    }








    datasetSize = args.dataset
    # torch.cuda.empty_cache()


    (imgBatch, labelBatch) = genPetImageStack(datasetSize, use="train", device_type=deviceType, img_height=imgHeight, img_width=imgWidth, color_channels=colorChannels, multi_out=multiOut, save=False)

    if not args.pretrained: cnn = CNN(pretrained=False, training=True, device_type=deviceType, hyperparameters=CNNHyperParameters, mlp_hyperparameters=MLPHyperParameters, input_data_dim=(colorChannels, imgHeight, imgWidth), cnn_model_config=CNNmodelConfig, mlp_model_config=MLPmodelConfig)
    else: cnn = CNN(pretrained=True, training=True, device_type=deviceType, hyperparameters=CNNHyperParameters, mlp_hyperparameters=MLPHyperParameters, cnn_model_params=fetchCNNParametersFromFile(deviceType, "./params/paramsCNN"), mlp_model_params=fetchMLPParametersFromFile(deviceType, "./params/paramsMLP"))



    (epochPlt, lossPlt) = cnn.train(imgBatch, labelBatch, epochs=args.epochs, save_params=True)
    plotTrainingResults(epochPlt, lossPlt) if epochPlt and args.plot else None

