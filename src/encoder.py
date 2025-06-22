import torch
import numpy as np
import torch.nn as nn
from functions import softmax 


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

        self.WQ.requires_grad_()
        self.WK.requires_grad_()
        self.WV.requires_grad_()
        self.WO.requires_grad_()

        self.ln_1 = nn.LayerNorm(self.d_model)


    # @staticmethod
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

        MH_a = H @ self.WO

        print(MH_a.shape)

        return MH_a


    def addNorm(self, X, MH_a):
        A = X + MH_a
        Anorm = self.ln_1(A)
        return Anorm


        



if __name__ == "__main__":
    batch_size = 5
    seq_len = 3
    embedding_size = 8
    x = torch.rand(size=(batch_size, seq_len, embedding_size))
    # x = torch.rand(size=(seq_len, embedding_size))

    encode = Encoder(False, device_type="cpu", embedding_size=embedding_size, num_heads=2)
    encode.multiHeadedAttention(x)
