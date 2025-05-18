import torch
from src.network import Network
from src.dense import DenseLayer


class MLP(Network):

  def __init__(self, pretrained, training, device_type, **kwargs):

    super().__init__(model_type="MLP", training=training, kwargs=kwargs)

    self.device_type = device_type

    if not pretrained:
      mlp_architecture = kwargs.get("mlp_architecture")
      self.checkConfig(architecture=mlp_architecture)
      self.layers = self.buildLayers(mlp_architecture=mlp_architecture, input_feature_count=kwargs.get("input_feature_count") or mlp_architecture.get("input_data_dim"))
    else:
      self.layers = self.loadLayers(mlp_model_params=kwargs.get("mlp_model_params"))

    self.num_layers = len(self.layers)

    if training and self.optimizer:
      self.setOptimizer()

    if not self.layers:
      raise ValueError("Layers are uninitialized!")




  def loadLayers(self, mlp_model_params):
    layers = [DenseLayer(
      pretrained=True, 
      device_type=self.device_type, 
      pretrained_weights=weights, 
      pretrained_biases=biases, 
      nonlinearity=nonlinearity, 
      index=index) for (weights, biases, nonlinearity, index) in mlp_model_params.values()]
    
    return layers


  def buildLayers(self, mlp_architecture, input_feature_count):

    neuron_counts = mlp_architecture.get("neuron_counts")
    activation_functions = mlp_architecture.get("mlp_activation_functions")

    neuron_counts.insert(0, input_feature_count)

    layers = [DenseLayer(
      pretrained=False, 
      device_type=self.device_type, 
      input_count=neuron_counts[i],
      neuron_count=neuron_counts[i+1], 
      nonlinearity=activation_functions[i], 
      index=i+2) for i in range(len(neuron_counts)-1)]

    return layers


  def saveParameters(self):
    for layer in self.layers:
      layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
      torch.save(layer.weights, f"./params/paramsMLP/layer_{layer.index}_weights_{layer.nonlinearity}.pth")
      torch.save(layer.biases, f"./params/paramsMLP/layer_{layer.index}_biases_{layer.nonlinearity}.pth")





  def forward(self, curr_input, training):
    for layer in self.layers:
      
      curr_input = layer.feed(curr_input)

      if training and self.dropout_rate and layer != self.layers[-1]:
        curr_input = self.dropout(curr_input)

    return curr_input.squeeze()




  def dropout(self, curr_input):
    # print("FC DROPOUT")

    drop_count = int(self.dropout_rate * curr_input.numel())
    dropout_row_indicies = torch.randint(low=0, high=curr_input.size(dim=0), size=(drop_count,))
    dropout_col_indicies = torch.randint(low=0, high=curr_input.size(dim=1), size=(drop_count,))

    curr_input[dropout_row_indicies, dropout_col_indicies] = 0

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



