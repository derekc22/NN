import torch
import numpy
import torch.nn as nn
from src.layer import Layer


class RecurrentCell(Layer):

    def __init__(self, pretrained, device_type, **kwargs):
        self.nonlinearity = kwargs.get("nonlinearity")
        self.index = int(kwargs.get("index"))

        self.device_type = device_type


        if not pretrained:
            self.


        else:
            pass
