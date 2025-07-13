import torch
import numpy as np
import torch.nn as nn
from utils.functions import softmax 
from src.dense import DenseLayer
from models.mlp import MLP
# torch.manual_seed(42)

class Encoder:

    def __init__(self, pretrained, **kwargs):

        self.index = int(kwargs.get("index"))
        self.device = kwargs.get("device")
        self.component = "encoder"

        if not pretrained:
            
            self.d_model = kwargs.get("d_model") #  AKA model dimension AKA d_model
            self.h = kwargs.get("num_heads")
            
            assert (self.d_model > self.h and self.d_model % self.h == 0)
            self.dk = int(self.d_model/self.h)
            self.sqrt_dk = np.sqrt(self.dk)

            stddev = np.sqrt(2 / self.d_model)

            self.WQKV = torch.normal(
                0, stddev / np.sqrt(3.0),
                size=(self.d_model, 3 * self.h * self.dk), # self.h * self.dk = self.d_model
                dtype=torch.float32,
                device=self.device)

            self.WO = torch.normal(
                0, stddev,
                size=(self.h * self.dk, self.d_model), # self.h * self.dk = self.d_model
                dtype=torch.float32,
                device=self.device)
            
            self.ff = MLP(**kwargs.get("ff_config"))
            
        else:
            pass
        
        self.WQKV.requires_grad_()
        self.WO.requires_grad_()

        self.ln_1 = nn.LayerNorm(self.d_model)
        self.ln_2 = nn.LayerNorm(self.d_model)

        self.padding_mask = kwargs.get("padding_mask")


    # def compute_QKV(self, X):
    #     # b = batch_dim, s = seq_len, e = embedding_dim, h = num_heads, k = dk
    #     Q = torch.einsum("bse,hek->bhsk", X, self.WQ)
    #     K = torch.einsum("bse,hek->bhsk", X, self.WK)
    #     V = torch.einsum("bse,hek->bhsk", X, self.WV)
    #     return Q, K, V
    
    def compute_QKV(self, X):
        # b = batch_dim, s = seq_len, d = d_model, h = num_heads, k = 3 * h * d
        QKV = torch.einsum('bsd, dk->bsk', X, self.WQKV)  # single Einsum = one GEMM
        Q, K, V = QKV.chunk(3, dim=-1)
        batch_size, seq_len = X.shape[:2]
        Q = Q.view(batch_size, seq_len, self.h, self.dk).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.h, self.dk).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.h, self.dk).transpose(1, 2)
        return Q, K, V


    # def attention(self, Q, K, V):
    #     KT = K.transpose(-1, -2)
    #     return softmax( (Q @ KT) / self.dk**0.5, dim=-1 ) @ V
    
    def attention(self, Q, K, V):
        KT = K.transpose(-1, -2)
        scores = (Q @ KT) / self.sqrt_dk

        if self.padding_mask is not None:
            # padding_mask shape: (batch, seq_len)
            # expand to (batch, 1, 1, seq_len) to broadcast over heads and queries
            expanded_mask = self.padding_mask.unsqueeze(1).unsqueeze(2)
            # scores = scores.masked_fill(expanded_mask == 0, -torch.inf)
            scores = scores.masked_fill(expanded_mask == 0, torch.finfo(scores.dtype).min)

        return softmax(scores, dim=-1) @ V


    def compute_heads(self, X):
        # X : (batch, seq_len, d_model)

        Q, K, V = self.compute_QKV(X)
    
        H_ = self.attention(Q, K, V)        # (batch, h, seq_len, dk)

        # reorder to (batch, seq_len, h, dk) then flatten heads
        H__ = H_.permute(0, 2, 1, 3)    # (batch, seq_len, h, dk)
        batch_size, seq_len = X.shape[:2]
        H = H__.reshape(batch_size, seq_len, self.d_model)   # (batch, seq_len, d_model)
        
        return H


    def multiheaded_attention(self, X):
        # X : (batch, seq_len, d_model)
        H = self.compute_heads(X)                   # (batch, seq_len, d_model)
        return H @ self.WO                        # (batch, seq_len, d_model)



    def add_norm(self, X, Y, ln):
        Z = X + Y
        ZNorm = ln(Z)
        return ZNorm


    # def feed_forward(self, curr_input):
    #     for layer in self.ff_layers:
    #         curr_input = layer.feed(curr_input)
    #     return curr_input



    def feed(self, X):
        MH_A = self.multiheaded_attention(X)
        
        ZNorm1 = self.add_norm(X, MH_A, self.ln_1)

        batch_size, seq_len = X.shape[:2]
        # ZFF = self.feed_forward(
        #     ZNorm1.reshape(-1, self.d_model)
        #     ).reshape(batch_size, seq_len, self.d_model)
        ZFF = self.ff.forward(
            ZNorm1.reshape(-1, self.d_model), training=True
            ).reshape(batch_size, seq_len, self.d_model)
        
        ZNorm2 = self.add_norm(ZNorm1, ZFF, self.ln_2)
        # print(ZNorm2.shape)
        # print(ZNorm2)
        # exit()
        
        return ZNorm2


# if __name__ == "__main__":
#     batch_size_ = 2
#     sequence_len = 3
#     d_model = 4
#     num_heads = 2
    
#     x = torch.rand(size=(batch_size_, sequence_len, d_model))
#     # x = torch.rand(size=(seq_len, d_model))

#     encoder = Encoder(
#         pretrained=False, 
#         device="cpu", 
#         d_model=d_model, 
#         num_heads=num_heads,
#         ff_neuron_count=3,
#         ff_nonlinearity="GELU"
#     )
#     encoder.feed(x)
