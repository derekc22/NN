import torch
import numpy as np
import torch.nn as nn
from src.layer import Layer


class DenseLayer(Layer):

  def __init__(self, pretrained, device_type, **kwargs):

    self.nonlinearity = kwargs.get("nonlinearity")
    self.index = int(kwargs.get("index"))


    self.device_type = device_type


    if not pretrained:

      input_count = kwargs.get("input_count")
      neuron_count = kwargs.get("neuron_count")

      ##### Initialize weights
      # self.weights = torch.rand(size=(neuron_count, input_count), dtype=torch.float32)  # Random Initialization

      # stddev = np.sqrt(2 / (input_count + neuron_count))
      # self.weights = torch.normal(0, stddev, size=(neuron_count, input_count), dtype=torch.float32, device=self.device_type)  # Xavier Initialization

      stddev = np.sqrt(2 / input_count)
      # self.weights = torch.normal(0, stddev, size=(neuron_count, input_count), dtype=torch.float32)  # He Initialization
      self.weights = torch.normal(0, stddev, size=(input_count, neuron_count), dtype=torch.float32, device=self.device_type)  # He Initialization

      # self.biases = torch.zeros(size=(neuron_count, 1), dtype=torch.float32, device=self.device_type)
      self.biases = torch.zeros(neuron_count, dtype=torch.float32, device=self.device_type)

    else:
      self.weights = kwargs.get("pretrained_weights").to(device=self.device_type)
      self.biases = kwargs.get("pretrained_biases").to(device=self.device_type)



    self.weights.requires_grad_()
    self.biases.requires_grad_()





  def __repr__(self):
    return (f"__________________________________________\n"
            f"MLP Layer {self.index}\nWeights:\n{self.weights}\nBiases:\n{self.biases}\nActivation:\n{self.nonlinearity}\n"
            f"__________________________________________")



  def feed(self, x):
    # print(self.weights.shape)
    # print(x.shape)
    # print(self.biases)
    z = torch.matmul(x, self.weights) + self.biases

    # ####### TESTING THIS ############################
    if x.size(dim=0) > 1:
      bn1 = nn.BatchNorm1d(num_features=self.weights.size(dim=1), dtype=torch.float32, device=self.device_type)
      # z = bn1(z.T).T
      z = bn1(z)
    # ####### TESTING THIS ############################


    y = self.activate(z, self.nonlinearity)


    return y