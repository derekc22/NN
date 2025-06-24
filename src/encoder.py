import torch
import numpy as np
import torch.nn as nn
from src.functions import softmax 
from src.dense import DenseLayer


class Encoder:

    def __init__(self, pretrained, device_type, **kwargs):

        self.device_type = device_type
        self.d_model = kwargs.get("embedding_size") #  AKA model dimension AKA d_model
        self.h = kwargs.get("num_heads")
        
        assert (self.d_model > self.h and self.d_model % self.h == 0)
        self.dk = int(self.d_model/self.h)

        stddev_model = np.sqrt(2 / self.d_model)

        if not pretrained:
            self.WQ = torch.normal(
                0, stddev_model,
                size=(self.h, self.d_model, self.dk),
                dtype=torch.float32,
                device=self.device_type)
            
            self.WK = torch.normal(
                0, stddev_model,
                size=(self.h, self.d_model, self.dk),
                dtype=torch.float32,
                device=self.device_type)
            
            self.WV = torch.normal(
                0, stddev_model,
                size=(self.h, self.d_model, self.dk),
                dtype=torch.float32,
                device=self.device_type)
            
            self.WO = torch.normal(
                0, stddev_model,
                size=(self.h * self.dk, self.d_model), # self.h * self.dk = self.d_model
                dtype=torch.float32,
                device=self.device_type)
            
            ff_neuron_count = kwargs.get("ff_neuron_count") # typically 2048 or 4096
            ff_nonlinearity = kwargs.get("ff_nonlinearity") # typically reLU or GELU
            
            self.ffLayers = [
                DenseLayer(
                    pretrained=pretrained,
                    device_type=device_type,
                    nonlinearity=ff_nonlinearity,
                    input_count=self.d_model,
                    neuron_count=ff_neuron_count,
                    index=1
                ),
                DenseLayer(
                    pretrained=pretrained,
                    device_type=device_type,
                    nonlinearity="none",
                    input_count=ff_neuron_count,
                    neuron_count=self.d_model,
                    index=2
                )
            ]

        self.WQ.requires_grad_()
        self.WK.requires_grad_()
        self.WV.requires_grad_()
        self.WO.requires_grad_()

        self.ln_1 = nn.LayerNorm(self.d_model)
        self.ln_2 = nn.LayerNorm(self.d_model)



    def computeQKV(self, X, WQ, WK, WV):
        Q = X @ WQ
        K = X @ WK
        V = X @ WV
        return Q, K, V


    def attention(self, Q, K, V):
        return softmax( (Q @ K.mT) / self.dk**0.5 ) @ V

    def computeHead(self, X):
        # X is ( batch x n/sequenceLength/numTokens x d_model )

        Q, K, V = torch.vmap(
            self.computeQKV, 
            in_dims=(None, 0, 0, 0)
            )(X, self.WQ, self.WK, self.WV)
        
        H_ = self.attention(Q, K, V)
        H = H_.reshape(-1, self.h * self.dk)

        return H


    def multiHeadedAttention(self, X):
        H = torch.vmap(
            self.computeHead,
            in_dims=(0)
            )(X)

        MH_A = H @ self.WO

        return MH_A


    def addNorm(self, X, Y, ln):
        Z = X + Y
        ZNorm = ln(Z)
        return ZNorm


    def feedForward(self, curr_input):
        for layer in self.ffLayers:
            curr_input = layer.feed(curr_input)
        return curr_input



    def feed(self, X):
        MH_A = self.multiHeadedAttention(X)
        
        ZNorm1 = self.addNorm(X, MH_A, self.ln_1)

        batch_size, seq_len = X.shape[:2]
        ZFF = self.feedForward(
            ZNorm1.reshape(-1, self.d_model)
            ).reshape(batch_size, seq_len, self.d_model)
        
        ZNorm2 = self.addNorm(ZNorm1, ZFF, self.ln_2)
        
        return ZNorm2

        



if __name__ == "__main__":
    batch_size_ = 5
    sequence_len = 3
    embedding_size = 8
    x = torch.rand(size=(batch_size_, sequence_len, embedding_size))
    # x = torch.rand(size=(seq_len, embedding_size))

    encoder = Encoder(
        pretrained=False, 
        device_type="cpu", 
        embedding_size=embedding_size, 
        num_heads=2,
        ff_neuron_count=64,
        ff_nonlinearity="GELU"
    )
    encoder.feed(x)
    # encode.computeHead(x)
