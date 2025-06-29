import torch
import numpy as np
import torch.nn as nn
from utils.functions import softmax 
from src.dense import DenseLayer
torch.manual_seed(42)

class Encoder:

    def __init__(self, pretrained, device_type, **kwargs):

        self.device_type = device_type

        if not pretrained:
            
            self.d_model = kwargs.get("embedding_size") #  AKA model dimension AKA d_model
            self.h = kwargs.get("num_heads")
            
            assert (self.d_model > self.h and self.d_model % self.h == 0)
            self.dk = int(self.d_model/self.h)

            stddev_model = np.sqrt(2 / self.d_model)
            
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
        # print("tester")
        # print(Q.grad_fn, Q.is_contiguous())
        # print((K.mT).grad_fn, (K.mT).is_contiguous())
        # print((Q @ K.mT).grad_fn, (Q @ K.mT).is_contiguous())
        # exit()
        return Q, K, V


    def attention(self, Q, K, V):
        return softmax( (Q @ K.mT) / self.dk**0.5, dim=-1 ) @ V

    def computeHeads(self, X):
        # X : (batch, seq_len, d_model)

        # project once per head
        Q, K, V = torch.vmap(
            self.computeQKV,
            in_dims=(None, 0, 0, 0)
        )(X, self.WQ, self.WK, self.WV)          # (h, batch, seq_len, dk)

        # scaled-dot-product attention, routed through the original helper
        H_ = self.attention(Q, K, V)        # (h, batch, seq_len, dk)

        # reorder to (batch, seq_len, h, dk) then flatten heads
        H__ = H_.permute(1, 2, 0, 3)    # (batch, seq_len, h, dk)
        batch_size, seq_len = X.shape[:2]
        H = H__.reshape(batch_size, seq_len, self.d_model)   # (batch, seq_len, d_model)
        
        return H


    def multiHeadedAttention(self, X):
        # X : (batch, seq_len, d_model)
        H = self.computeHeads(X)                   # (batch, seq_len, d_model)
        return H @ self.WO                        # (batch, seq_len, d_model)



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
        print(MH_A.shape)
        print(MH_A)
        exit()
        
        ZNorm1 = self.addNorm(X, MH_A, self.ln_1)

        batch_size, seq_len = X.shape[:2]
        ZFF = self.feedForward(
            ZNorm1.reshape(-1, self.d_model)
            ).reshape(batch_size, seq_len, self.d_model)
        
        ZNorm2 = self.addNorm(ZNorm1, ZFF, self.ln_2)
        
        return ZNorm2

        



if __name__ == "__main__":
    batch_size_ = 2
    sequence_len = 3
    embedding_size = 4
    num_heads = 2
    
    x = torch.rand(size=(batch_size_, sequence_len, embedding_size))
    # x = torch.rand(size=(seq_len, embedding_size))

    encoder = Encoder(
        pretrained=False, 
        device_type="cpu", 
        embedding_size=embedding_size, 
        num_heads=num_heads,
        ff_neuron_count=3,
        ff_nonlinearity="GELU"
    )
    encoder.feed(x)
    # encode.computeHead(x)
