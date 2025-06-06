import torch
from src.network import Network
from src.dense import DenseLayer
import os


class MLP(Network):

  def __init__(self, pretrained, training, device_type, **kwargs):

    super().__init__(model_type="mlp", training=training, kwargs=kwargs)

    self.device_type = torch.device(device_type)

    if not pretrained:
      architecture = kwargs.get("architecture")
      self.input_feature_count = kwargs.get("input_feature_count")
      self.checkConfig(architecture=architecture)
      self.layers = self.buildLayers(architecture=architecture) #or mlp_architecture.get("input_data_dim"))
      self.save_fpath = kwargs.get("save_fpath")
    else:
      self.layers = self.loadLayers(model_params=kwargs.get("model_params"))

    if not self.layers:
      raise ValueError("Layers are uninitialized!")
    self.num_layers = len(self.layers)
    
    if training and self.optimizer:
      self.setOptimizer()





  def loadLayers(self, model_params):
    layers = [DenseLayer(
      pretrained=True, 
      device_type=self.device_type, 
      pretrained_weights=weights, 
      pretrained_biases=biases, 
      nonlinearity=nonlinearity, 
      index=index) for (weights, biases, nonlinearity, index) in model_params.values()]
    
    return layers


  def buildLayers(self, architecture):

    neuron_counts = architecture.get("neuron_counts")
    activation_functions = architecture.get("activation_functions")

    neuron_counts.insert(0, self.input_feature_count)

    layers = [DenseLayer(
      pretrained=False, 
      device_type=self.device_type, 
      input_count=neuron_counts[i],
      neuron_count=neuron_counts[i+1], 
      nonlinearity=activation_functions[i], 
      # index=i+2) for i in range(len(neuron_counts)-1)]
      index=i+1) for i in range(len(neuron_counts)-1)]

    return layers


  def saveParameters(self):
    os.makedirs(f"{self.save_fpath}", exist_ok=True)
    for layer in self.layers:
      layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
      torch.save(layer.weights, f"{self.save_fpath}/layer_{layer.index}_weights_{layer.nonlinearity}.pth")
      torch.save(layer.biases, f"{self.save_fpath}/layer_{layer.index}_biases_{layer.nonlinearity}.pth")





  def forward(self, curr_input, training):
    for layer in self.layers:
      
      curr_input = layer.feed(curr_input)

      if training and self.dropout_rate and layer != self.layers[-1]:
        curr_input = self.dropout(curr_input)

    return curr_input.squeeze()




  def dropout(self, curr_input):
    # print("FC DROPOUT")

    drop_count = int(self.dropout_rate * curr_input.numel())
    dropout_row_indices = torch.randint(low=0, high=curr_input.size(dim=0), size=(drop_count,))
    dropout_col_indices = torch.randint(low=0, high=curr_input.size(dim=1), size=(drop_count,))

    curr_input[dropout_row_indices, dropout_col_indices] = 0

    return curr_input






  def backprop(self, loss):

    self.zerograd()

    loss.backward()

    with torch.no_grad():

      if not self.optimizer:
          self.update()
      else:
        self.t += 1
        self.optimizerUpdate()



