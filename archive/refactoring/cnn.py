import os
from src.network import Network
from models.mlp import MLP
from cnn_layer import CNNLayer
import torch

class CNN(Network):

  def __init__(self, pretrained, device_type, training, **kwargs):

    super().__init__(model_type="cnn", training=training, kwargs=kwargs)

    self.device_type = torch.device(device_type)
    self.save_fpath = kwargs.get("cnn_save_fpath")
    mlp_save_fpath = kwargs.get("mlp_save_fpath")

    if not pretrained:
      architecture = kwargs.get("cnn_architecture")
      self.checkConfig(architecture=architecture)
      self.layers = self.buildLayers(architecture=architecture)

      mlp_input_feature_count = self.calcMLPInputSize(kwargs.get("input_data_dim"))
      self.MLP = MLP(pretrained=False, device_type=self.device_type, training=training, input_feature_count=mlp_input_feature_count, architecture=kwargs.get("mlp_architecture"), hyperparameters=kwargs.get("mlp_hyperparameters"), save_fpath=mlp_save_fpath)
    else:
      self.layers = self.loadLayers(kwargs.get("cnn_model_params"))
      self.MLP = MLP(pretrained=True, device_type=self.device_type, training=training, model_params=kwargs.get("mlp_model_params"), hyperparameters=kwargs.get("mlp_hyperparameters"), save_fpath=mlp_save_fpath)

    if not (self.layers and self.MLP.layers):
      raise ValueError("CNN or MLP layers are uninitialized!")
    self.num_layers = len(self.layers)

    if training and self.optimizer:
      self.setOptimizer()




  def loadLayers(self, model_params):

    layers = [CNNLayer(
      pretrained=True, 
      device_type=self.device_type, 
      type=layer_type, 
      pretrained_kernels=kernels, 
      pretrained_biases=biases, 
      nonlinearity=nonlinearity,
      kernel_stride=stride, 
      index=index) for (layer_type, kernels, biases, nonlinearity, stride, index) in model_params.values()]

    return layers


  def buildLayers(self, architecture):
    layer_types = architecture.get("type")
    filter_counts = architecture.get("filter_counts")
    kernel_shapes = architecture.get("kernel_shapes")
    kernel_strides = architecture.get("kernel_strides")
    activation_functions = architecture.get("activation_functions")
    num_layers = len(layer_types)

    layers = [CNNLayer(
      pretrained=False, 
      device_type=self.device_type, 
      type=layer_types[i], 
      filter_count=filter_counts[i],
      kernel_shape=kernel_shapes[i], 
      kernel_stride=kernel_strides[i], 
      nonlinearity=activation_functions[i], 
      index=i+1) for i in range(num_layers)]

    return layers



  def saveParameters(self):
    os.makedirs(f"{self.save_fpath}", exist_ok=True)
    for layer in self.layers:
      layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
      torch.save(layer.kernels, f"{self.save_fpath}/cnn_layer_{layer.index}_kernels_{layer.nonlinearity}_{layer.type}_{layer.kernel_stride}.pth")
      torch.save(layer.biases, f"{self.save_fpath}/cnn_layer_{layer.index}_biases_{layer.nonlinearity}_{layer.type}_{layer.kernel_stride}.pth")

    self.MLP.saveParameters()



  def calcMLPInputSize(self, input_data_dim):

    print("calculating MLP input size.......")

    input_data_dim = (1, ) + input_data_dim

    dummy_data = torch.empty(size=input_data_dim, device=self.device_type)
    dummy_MLP_input = self.forward(dummy_data, training=True, dummy=True)
    dummy_mlp_input_feature_count = dummy_MLP_input.size(dim=1)
    return dummy_mlp_input_feature_count






  def forward(self, curr_input, training, dummy=False):  # maybe recursion would work for this?

    for layer in self.layers:
      if layer.type == "convolutional":

        curr_input = layer.convolve(curr_input)

        # if self.dropout_rate:
        #     curr_input = self.spatialDropout(curr_input)

        # if training and not dummy and self.dropout_rate: #layer != self.layers[-1]:
        #     curr_input = layer.convolve(curr_input, dropout_rate=self.dropout_rate)
        # else:
        #     curr_input = layer.convolve(curr_input)


      else:
        curr_input = layer.maxpool(curr_input)


    curr_input_batch_size = curr_input.size(dim=0)
    # flattened_feature_map = curr_input.view(curr_input_batch_size, -1).to(torch.float32).T
    flattened_feature_map = curr_input.view(curr_input_batch_size, -1).to(torch.float32)
    # print(flattened_feature_map.shape)


    if dummy:
      return flattened_feature_map
    else:
      return self.MLP.forward(flattened_feature_map, training=training)




  # def spatialDropout(self, curr_input):

  #   drop_count = int(self.dropout_rate * curr_input.size(dim=1))

  #   dropout_indices = torch.randint(low=0, high=curr_input.size(dim=1), size=(drop_count, ))

  #   curr_input[:, dropout_indices, :, :] = 0

  #   # print(f"curr_input_dim = {curr_input.size()}. dim 1 = {curr_input.size(dim=1)}")
  #   # print(dropout_indices)
  #   # print(curr_input)

  #   return curr_input







  def backprop(self, loss):

    self.zerograd()
    self.MLP.zerograd()


    loss.backward()

    with torch.no_grad():

      if not self.optimizer:
        self.update()
      else:
        self.t += 1
        self.optimizerUpdate()


      if not self.MLP.optimizer:
        self.MLP.update()
      else:
        self.MLP.t += 1
        self.MLP.optimizerUpdate()
