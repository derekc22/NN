import torch
import numpy as np


def GELU(k):
    arg = np.sqrt(2/np.pi) * (k + 0.044715 * torch.pow(k, 3))
    return 0.5 * k * (1 + tanh(arg))  


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


def softmax(k, dim):
    k = k - torch.max(k, dim=dim, keepdim=True).values  # Subtract the max value from the logits to avoid overflow
    exp_k = torch.exp(k)
    return exp_k / torch.sum(exp_k, dim=dim, keepdim=True)


def none(k):
    return k


def activate(z, activation):
    return globals()[activation](z)
