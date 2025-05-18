import torch
import numpy as np
import torch.nn as nn
from src.layer import Layer


class RecurrentCell(Layer):

    def __init__(self, pretrained, device_type, **kwargs):
        self.nonlinearity = kwargs.get("nonlinearity")
        self.index = int(kwargs.get("index"))

        self.device_type = device_type


        if not pretrained:
            input_size = kwargs.get("input_size")
            neuron_count = kwargs.get("neuron_count")
            stddev = np.sqrt(2 / input_size)

            self.wxh = torch.normal(0, stddev, size=(input_size, neuron_count), device_type=torch.float32) # He Initialization
            self.whh = torch.normal(0, stddev, size=(input_size, neuron_count), device_type=torch.float32) # He Initialization
            self.why = torch.normal(0, stddev, size=(input_size, neuron_count), device_type=torch.float32) # He Initialization


        else:
            pass
