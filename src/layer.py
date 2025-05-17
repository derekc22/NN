import torch

class Layer:


  @staticmethod
  def reLU(k):
    return k * (k > 0)

  @staticmethod
  def leakyReLU(k):
    alpha = 0.01
    k[k < 0] *= alpha
    return k

  @staticmethod
  def sigmoid(k):
    return 1/(1 + torch.exp(-k))



  @staticmethod
  def softmax(k):
    k = k - torch.max(k)  # Subtract the max value from the logits to avoid overflow
    exp_k = torch.exp(k)
    return exp_k / torch.sum(exp_k)



  @staticmethod
  def none(k):
    return k


  def activate(self, z, nonlinearity):
    return getattr(self, nonlinearity)(z)