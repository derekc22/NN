import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import time

class Network:

  def __init__(self, model_type, training, kwargs):

    self.model_type = model_type
    self.epoch = 0 # temp hack (should ideally be under the 'if training' block) for the rnn implementation. update later
    self.epochs = -1 # temp hack (should ideally be under the 'if training' block) for the rnn implementation. update later

    if training:
      hyperparameters = kwargs.get("hyperparameters")
      self.learn_rate = hyperparameters.get("learn_rate")
      self.batch_size = hyperparameters.get("batch_size")
      self.loss_func = hyperparameters.get("loss_func")
      self.reduction = hyperparameters.get("reduction")
      self.optimizer = hyperparameters.get("optimizer")
      self.lambda_L2 = hyperparameters.get("lambda_L2")  # regularization strength which controls how much the L2 penalty influences the loss. Larger λ values increase the regularization effect.
      self.dropout_rate = hyperparameters.get("dropout_rate")
      self.grad_clip_norm  = hyperparameters.get('grad_clip_norm')
      self.grad_clip_value = hyperparameters.get('grad_clip_value')





  def _collect_parameters(self):
    """
    Walk self.layers and return a flat list of all weight Tensors whose
    .grad you want to clip. Adapt these attribute names to match your layers.
    """
    params = []
    for layer in self.layers:
      if self.model_type == "cnn" and layer.is_conv_layer:
        params.extend([layer.kernels, layer.biases])
      elif self.model_type == "mlp":
        params.extend([layer.weights, layer.biases])
      elif self.model_type == "rnn":
        params.append(layer.wxh)
        if layer.type == "hidden":
          params.extend([layer.whh, layer.bh])
        elif layer.type == "output":
          params.append(layer.by)
    return params


  def clipGradients(self):
    """
    Apply PyTorch’s native clipping to the gradients in-place.
    Call immediately after loss.backward().
    """
    params = self._collect_parameters()
    # print(len(params))

    # grads = [p.grad for p in params if p.grad is not None]
    # total_grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
    # print("Before clip, grad norm:", total_grad_norm.item())
    
    if self.grad_clip_norm is not None:
      clip_grad_norm_(params, self.grad_clip_norm)
    if self.grad_clip_value is not None:
      clip_grad_value_(params, self.grad_clip_value)

    # grads = [p.grad for p in params if p.grad is not None]
    # total_grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
    # print("After clip, grad norm:", total_grad_norm.item())





  def setOptimizer(self):

    # match self.optimizer:

    if self.optimizer == "adam":
      self.t = 0
      self.weight_moment_list = [[0, 0]]*self.num_layers
      self.bias_moment_list = [[0, 0]]*self.num_layers
      self.other_moment_list = [[0, 0]]*self.num_layers




  def inference(self, data):
    with torch.no_grad(): # new
      return self.forward(data, training=False)





  def train(self, data, target, epochs, save_params=True):

    epoch_plt = []
    loss_plt = []
    self.epochs = epochs

    # epochs = (epochs or (data.size(dim=0/self.batch_size)))

    for epoch in range(epochs): #range(1, int(epochs+1)):
      start = time.time()
      
      data_batch, target_batch = self.batch(data, target)
      pred_batch = self.forward(data_batch, training=True)
      # print(f"pred: {pred_batch.shape}")

      loss = getattr(self, self.loss_func)(pred_batch, target_batch)

      if self.lambda_L2:
        loss += self.L2Regularization()
      self.backprop(loss)


      epoch_plt.append(epoch)
      loss_plt.append(loss.item())
      print(f"OG epoch = {epoch+1}, OG loss = {loss}, OG time:{time.time()-start}")
      print(f"__________________________________________")
      
      self.epoch = epoch


    if save_params:
      self.saveParameters()
      
    return epoch_plt, loss_plt





  def reduce(self, x):
    if self.reduction == "mean":
      return x.mean()
    elif self.reduction == "sum":
      return x.sum()





  # def batch(self, data, target):
  #   batch_indices = torch.randperm(n=data.size(dim=0))[:self.batch_size]  # stochastic

  #   data_batch = data[batch_indices]

  #   target_batch = target.T[batch_indices].T

  #   return data_batch, target_batch


  def batch(self, data, target):
    # print( "yoooo", self.batch_size)
    if self.batch_size:
      # print("batched")
      # batch_indices = torch.randperm(n=data.size(dim=0))[:self.batch_size]  # stochastic
      batch_indices = torch.randperm(n=data.shape[0])[:self.batch_size]  # stochastic

      data_batch = data[batch_indices]
      target_batch = target[batch_indices]
      return data_batch, target_batch

    
    return data, target





  def CCELoss(self, pred_batch, target_batch):

    epsilon = 1e-8
    pred_batch = torch.clamp(pred_batch, epsilon, 1 - epsilon)
    errs = torch.mul(target_batch, torch.log(pred_batch))

    cce_loss = -torch.sum(errs, dim=0)  # CCE (Categorical Cross Entropy) Loss
    cce_loss_reduced = self.reduce(cce_loss)

    return cce_loss_reduced




  def CCELossNN(self, data, target):
    criterion = nn.CrossEntropyLoss()

    pred = self.forward(data).reshape(1, -1)
    target = torch.tensor([torch.nonzero(target == 1.0)[:, 0].item()])
    return criterion(pred, target)



  def BCEWithLogitsLoss(self, pred_batch, target_batch):

    max_val = torch.clamp(pred_batch, min=0)
    stable_log_exp = max_val + torch.log(1 + torch.exp(-torch.abs(pred_batch)))

    errs = stable_log_exp - torch.mul(target_batch, pred_batch)

    bce_with_logits_loss = torch.sum(errs, dim=0)  # BCE with logits loss (DO NOT USE SIGMOID ACTIVATION)
    bce_with_logits_loss_reduced = self.reduce(bce_with_logits_loss)

    return bce_with_logits_loss_reduced







  def BCELoss(self, pred_batch, target_batch):
    # print(pred_batch.shape)
    # print(target_batch.shape)
    epsilon = 1e-8
    pred_batch = torch.clamp(pred_batch, epsilon, 1 - epsilon)
    errs = target_batch * torch.log(pred_batch) + (1 - target_batch) * torch.log(1 - pred_batch)

    bce_loss = -(1/self.batch_size)*torch.sum(errs, dim=0)  # BCE (Binary Cross Entropy) Loss
    bce_loss_reduced = self.reduce(bce_loss)

    return bce_loss_reduced



  def FocalBCELoss(self, pred_batch, target_batch):

    """Hyperparameter"""
    alpha = 0.7  # positive class weighting
    gamma = 2  #
    epsilon = 1e-8

    pred_batch = torch.clamp(pred_batch, epsilon, 1 - epsilon)
    errs = alpha * torch.pow((1-pred_batch), gamma) * (target_batch * torch.log(pred_batch) + (1-target_batch) * torch.log(1-pred_batch))

    focal_bce_loss = -(1/self.batch_size)*torch.sum(errs, dim=0)  # Focal BCE (Binary Cross Entropy) Loss
    focal_bce_loss_reduced = self.reduce(focal_bce_loss)

    return focal_bce_loss_reduced



  def MSELoss(self, pred_batch, target_batch):
    
    errs = (pred_batch - target_batch)**2
    mse_loss = (1/self.batch_size)*torch.sum(errs, dim=0) if self.batch_size else (1/pred_batch.shape[0])*torch.sum(errs, dim=0) # MSE (Mean Squared Error) Loss
    mse_loss_reduced = self.reduce(mse_loss)

    return mse_loss_reduced
  
  def SSELoss(self, pred_batch, target_batch):
    errs = (pred_batch - target_batch)**2
    sse_loss = torch.sum(errs, dim=0)  # SSE (Sum Squared Error) Loss
    sse_loss_reduced = self.reduce(sse_loss)

    return sse_loss_reduced



  def update(self):

    for layer in self.layers:

      if self.model_type == "cnn" and layer.is_conv_layer:
        layer.kernels -= self.learn_rate * layer.kernels.grad
        layer.biases -= self.learn_rate * layer.biases.grad

      elif self.model_type == "mlp":
        layer.weights -= self.learn_rate * layer.weights.grad
        layer.biases -= self.learn_rate * layer.biases.grad

      elif self.model_type == "rnn":
        layer.wxh -= self.learn_rate * layer.wxh.grad
        if layer.type == "hidden":
          layer.whh -= self.learn_rate * layer.whh.grad
          layer.bh -= self.learn_rate * layer.bh.grad
        elif layer.type == "output":
          layer.by -= self.learn_rate * layer.by.grad



  def adam(self, layer_index, gt, param_type, *args):

    moment_list = self.weight_moment_list if param_type == "weight" else self.bias_moment_list if param_type == "bias" else self.other_moment_list

    mt_1, vt_1 = moment_list[layer_index]

    """Hyperparameter"""
    beta1 = 0.9  # first moment estimate decay rate (smaller = more aggressive)
    beta2 = 0.999  # second moment estimate decay rate (smaller = more aggressive)
    epsilon = 1e-8

    mt = beta1*mt_1 + (1-beta1)*gt
    vt = beta2*vt_1 + (1-beta2)*gt**2
    mt_hat = mt/(1-beta1**self.t)
    vt_hat = vt/(1-beta2**self.t)

    moment_list[layer_index] = [mt, vt]

    adam_grad = (self.learn_rate*mt_hat)/(torch.sqrt(vt_hat) + epsilon)

    # print(mt_1.shape)
    # print(gt.shape)
    # print(mt_1)
    # print(gt)

    return adam_grad




  def optimizerUpdate(self):

    optimizer_func = getattr(self, self.optimizer)

    for layer in self.layers:

      if self.model_type == "cnn" and layer.is_conv_layer:
        # layer_index = self.layers.index(layer)
        layer_index = layer.index-1 # why am i not just doing this???
        layer.kernels -= optimizer_func(layer_index=layer_index, gt=layer.kernels.grad, param_type="weight")
        layer.biases -= optimizer_func(layer_index=layer_index, gt=layer.biases.grad, param_type="bias")
        # print(layer.kernels)

      elif self.model_type == "mlp":
        # layer_index = self.layers.index(layer)
        layer_index = layer.index-1 # why am i not just doing this???
        layer.weights -= optimizer_func(layer_index=layer_index, gt=layer.weights.grad, param_type="weight")
        layer.biases -= optimizer_func(layer_index=layer_index, gt=layer.biases.grad, param_type="bias")

      elif self.model_type == "rnn":
        # layer_index = self.layers.index(layer)
        layer_index = layer.index-1 # why am i not just doing this???
        layer.wxh -= optimizer_func(layer_index=layer_index, gt=layer.wxh.grad, param_type="weight")
        if layer.type == "hidden":
          layer.whh -= optimizer_func(layer_index=layer_index, gt=layer.whh.grad, param_type="other")
          layer.bh -= optimizer_func(layer_index=layer_index, gt=layer.bh.grad, param_type="bias")
        elif layer.type == "output":
          layer.by -= optimizer_func(layer_index=layer_index, gt=layer.by.grad, param_type="bias")






  def L2Regularization(self):

      weight_sum = 0

      for layer in self.layers:
        if self.model_type == "cnn" and layer.is_conv_layer:
          weight_sum += (torch.sum(layer.kernels ** 2))
        elif self.model_type == "mlp":
          weight_sum += (torch.sum(layer.weights ** 2))
        elif self.model_type == "rnn":
          weight_sum += (torch.sum(layer.wxh ** 2))


      regularization = self.lambda_L2*weight_sum
      # print(regularization)

      return regularization







  def checkConfig(self, architecture):

    config_lengths = [len(v) for k, v in architecture.items()]
    all_same_length = all(config_length == config_lengths[0] for config_length in config_lengths)

    if not all_same_length:
      raise IndexError(f"{self.model_type} Configuration Error. Recheck sizes of configuration objects: {config_lengths}")






  def printLayers(self):
    for layer in self.layers:
      print(layer)

    if self.model_type == "cnn":
      for layer in self.MLP.layers:
        print(layer)



  def zerograd(self):

    for layer in self.layers:

      if self.model_type == "cnn" and layer.is_conv_layer:
        layer.kernels.grad = None
        layer.biases.grad = None

      elif self.model_type == "mlp":
        layer.weights.grad = None
        layer.biases.grad = None

      elif self.model_type == "rnn":
        layer.wxh.grad = None
        if layer.type == "hidden":
          layer.whh.grad = None
          layer.bh.grad = None
        elif layer.type == "output":
          layer.by.grad = None