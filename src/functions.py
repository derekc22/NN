import torch



def tanh(k):
  return torch.tanh(k)


def reLU(k):
  return k * (k > 0)


def leakyReLU(k):
  alpha = 0.01
  k[k < 0] *= alpha
  return k


def sigmoid(k):
  return 1/(1 + torch.exp(-k))



def softmax(k):
  k = k - torch.max(k)  # Subtract the max value from the logits to avoid overflow
  exp_k = torch.exp(k)
  return exp_k / torch.sum(exp_k)


def none(k):
  return k


def activate(z, nonlinearity):
  return globals()[nonlinearity](z)
