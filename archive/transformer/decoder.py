import torch
import numpy as np
import torch.nn as nn
from utils.functions import softmax 
from src.dense import DenseLayer
torch.manual_seed(42)

class Decoder:

    def __init__(self, pretrained, device_type, **kwargs):

        self.device_type = device_type
        self.type = kwargs.get("type")
        ff_nonlinearity = kwargs.get("ff_nonlinearity") # typically reLU or GELU
        

        if not pretrained:
            
            self.d_model = kwargs.get("embedding_size") # should be the same as the encoder
            self.h = kwargs.get("num_heads") # may be different from the encoder
            
            assert (self.d_model > self.h and self.d_model % self.h == 0)
            self.dk = int(self.d_model/self.h)

            stddev_model = np.sqrt(2 / self.d_model)
            
            self.WQ_masked = torch.normal(
                0, stddev_model,
                size=(self.h, self.d_model, self.dk),
                dtype=torch.float32,
                device=self.device_type)
            
            self.WK_masked = torch.normal(
                0, stddev_model,
                size=(self.h, self.d_model, self.dk),
                dtype=torch.float32,
                device=self.device_type)
            
            self.WV_masked = torch.normal(
                0, stddev_model,
                size=(self.h, self.d_model, self.dk),
                dtype=torch.float32,
                device=self.device_type)
            
            self.WO_masked = torch.normal(
                0, stddev_model,
                size=(self.h * self.dk, self.d_model), # self.h * self.dk = self.d_model
                dtype=torch.float32,
                device=self.device_type)
            

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
            
            
            if self.type == "output":
                V = kwargs.get("vocab_size")
                self.linear = torch.normal(
                    0, stddev_model,
                    size=(self.d_model, V),
                    dtype=torch.float32,
                    device=self.device_type)


        self.WQ_masked.requires_grad_()
        self.WK_masked.requires_grad_()
        self.WV_masked.requires_grad_()
        self.WO_masked.requires_grad_()
        self.WQ.requires_grad_()
        self.WK.requires_grad_()
        self.WV.requires_grad_()
        self.WO.requires_grad_()

        self.ln_1 = nn.LayerNorm(self.d_model)
        self.ln_2 = nn.LayerNorm(self.d_model)
        self.ln_3 = nn.LayerNorm(self.d_model)



    def computeQKV(self, X, WQ, WK, WV):
        Q = X @ WQ
        K = X @ WK
        V = X @ WV
        return Q, K, V
    
    def computeQKV2(self, X, X_encoder, WQ, WK, WV):
        Q = X @ WQ
        K = X_encoder @ WK
        V = X_encoder @ WV
        return Q, K, V
    
    
    def maskedAttention(self, Q, K, V):
        seq_len = Q.shape[1]
        M = torch.tril(torch.full((seq_len, seq_len), -torch.inf))
        return softmax( (Q @ K.mT + M) / self.dk**0.5, dim=-1 ) @ V
    
    def attention2(self, Q, K, V):
        return softmax( (Q @ K.mT) / self.dk**0.5, dim=-1 ) @ V


    def computeMaskedHead(self, X):
        # X is ( batch x n/sequenceLength/numTokens x d_model )

        Q, K, V = torch.vmap(
            self.computeQKV, 
            in_dims=(None, 0, 0, 0)
            )(X, self.WQ_masked, self.WK_masked, self.WV_masked)
        
        H_ = self.maskedAttention(Q, K, V)
        # H = H_.reshape(-1, self.h * self.dk)
        seq_len = X.shape[0]
        H = H_.permute(1, 0, 2).reshape(seq_len, self.h * self.dk)

        return H

    def computeHead2(self, X, X_encoder):
        # X is ( batch x n/sequenceLength/numTokens x d_model )

        Q, K, V = torch.vmap(
            self.computeQKV2, 
            in_dims=(None, None, 0, 0, 0)
            )(X, X_encoder, self.WQ, self.WK, self.WV)
        
        H_ = self.attention2(Q, K, V)
        # H = H_.reshape(-1, self.h * self.dk)
        seq_len = X.shape[0]
        H = H_.permute(1, 0, 2).reshape(seq_len, self.h * self.dk)

        return H

    
    def maskedMultiHeadedAttention(self, X):
        H = torch.vmap(
            self.computeMaskedHead,
            in_dims=(0)
            )(X)

        MMH_A = H @ self.WO_masked

        return MMH_A

    def multiHeadedAttention2(self, X, X_encoder):
        H = torch.vmap(
            self.computeHead2,
            in_dims=(0, 0)
            )(X, X_encoder)

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


    def linearFeed(self, ZNorm3):
        return ZNorm3 @ self.linear


    def feed(self, X, X_encoder):
        MMH_A = self.maskedMultiHeadedAttention(X)
        # print(MMH_A)
        # exit()
        
        ZNorm1 = self.addNorm(X, MMH_A, self.ln_1)

        MH_A = self.multiHeadedAttention2(ZNorm1, X_encoder)
        
        ZNorm2 = self.addNorm(ZNorm1, MH_A, self.ln_2)
        
        batch_size, seq_len = X.shape[:2]
        ZFF = self.feedForward(
            ZNorm2.reshape(-1, self.d_model)
            ).reshape(batch_size, seq_len, self.d_model)
        
        ZNorm3 = self.addNorm(ZNorm2, ZFF, self.ln_3)
        
        # print(ZNorm3)
        
        return ZNorm3

        



if __name__ == "__main__":
    batch_size_ = 2
    sequence_len = 3
    embedding_size = 4
    num_heads = 2
    xe = torch.rand(size=(batch_size_, sequence_len, embedding_size))
    xd = torch.rand(size=(batch_size_, sequence_len, embedding_size))

    from src.encoder import Encoder
    encoder = Encoder(
        pretrained=False, 
        device_type="cpu", 
        embedding_size=embedding_size, 
        num_heads=num_heads,
        ff_neuron_count=3,
        ff_nonlinearity="GELU"
    )
    decoder = Decoder(
        pretrained=False, 
        device_type="cpu", 
        embedding_size=embedding_size, 
        num_heads=num_heads,
        ff_neuron_count=3,
        ff_nonlinearity="GELU"
    )
    
    out = encoder.feed(xe)
    print(out)    
    exit()

    decoder.feed(xd, out)
    
# if __name__ == "__main__":
#     batch_size_ = 5
#     sequence_len = 3
#     embedding_size = 8
#     x = torch.rand(size=(batch_size_, sequence_len, embedding_size))

#     decoder = Decoder(
#         pretrained=False, 
#         device_type="cpu", 
#         embedding_size=embedding_size, 
#         num_heads=2,
#         ff_neuron_count=3,
#         ff_nonlinearity="GELU"
#     )
    
#     decoder.feed(x)
