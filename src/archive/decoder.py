import torch
import numpy as np
import torch.nn as nn
from utils.functions import softmax 
from src.dense import DenseLayer
from models.mlp import MLP
# torch.manual_seed(42)

class Decoder:

    def __init__(self, pretrained, device_type, **kwargs):

        self.index = int(kwargs.get("index"))
        self.device_type = device_type
        self.type = kwargs.get("type")
        self.component = "decoder"
        

        if not pretrained:
            
            self.d_model = kwargs.get("embedding_size") # should be the same as the encoder
            self.h = kwargs.get("num_heads") # may be different from the encoder
            
            assert (self.d_model > self.h and self.d_model % self.h == 0)
            self.dk = int(self.d_model/self.h)
            self.sqrt_dk = self.dk**0.5

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
            ff_nonlinearity = kwargs.get("ff_nonlinearity") # typically reLU or GELU
            
            # self.ff_layers = [
            #     DenseLayer(
            #         pretrained=pretrained,
            #         device_type=device_type,
            #         nonlinearity=ff_nonlinearity,
            #         input_count=self.d_model,
            #         neuron_count=ff_neuron_count,
            #         index=1
            #     ),
            #     DenseLayer(
            #         pretrained=pretrained,
            #         device_type=device_type,
            #         nonlinearity="none",
            #         input_count=ff_neuron_count,
            #         neuron_count=self.d_model,
            #         index=2
            #     )
            # ]
            
            self.ff = MLP(pretrained=False, 
                          device_type=self.device_type, 
                          training=True, 
                          input_feature_count=self.d_model, 
                          architecture=kwargs.get("ff_architecture"), 
                          hyperparameters=kwargs.get("ff_hyperparameters"), 
                          save_fpath=kwargs.get("ff_save_fpath")
                        )
            
            if self.type == "output":
                V = kwargs.get("vocab_size")
                self.linear = torch.normal(
                    0, stddev_model,
                    size=(self.d_model, V),
                    dtype=torch.float32,
                    device=self.device_type)
                self.linear.requires_grad_()


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

        self.padding_mask = kwargs.get("padding_mask")
        self.combined_mask = None


    def compute_QKV(self, X):
        # b = batch_dim, s = seq_len, e = embedding_dim, h = num_heads, k = dk
        Q = torch.einsum("bse,hek->bhsk", X, self.WQ_masked)
        K = torch.einsum("bse,hek->bhsk", X, self.WK_masked)
        V = torch.einsum("bse,hek->bhsk", X, self.WV_masked)
        return Q, K, V
    
    def compute_QKV2(self, X, X_encoder):
        # b = batch_dim, s = seq_len, e = embedding_dim, h = num_heads, k = dk
        Q = torch.einsum("bse,hek->bhsk", X, self.WQ)
        K = torch.einsum("bse,hek->bhsk", X_encoder, self.WK)
        V = torch.einsum("bse,hek->bhsk", X_encoder, self.WV)
        return Q, K, V
    
    
    # def masked_attention(self, Q, K, V):
    #     batch_size = Q.shape[0]
    #     seq_len = Q.shape[2]
    #     M = torch.tril(torch.full((batch_size, self.h, seq_len, seq_len), -torch.inf))
    #     KT = K.transpose(-1, -2)
    #     return softmax( (Q @ KT + M) / self.dk**0.5, dim=-1 ) @ V
    
    # def masked_attention(self, Q, K, V):
    #     batch_size, _, seq_len, _ = Q.shape
    #     causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device_type)).bool()
    #     causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    #     if self.padding_mask is not None:
    #         padding_mask = self.padding_mask.unsqueeze(1).unsqueeze(2).bool()  # (B, 1, 1, S)
    #         combined_mask = padding_mask & causal_mask
    #     else:
    #         combined_mask = causal_mask

    #     KT = K.transpose(-1, -2)
    #     scores = (Q @ KT) / self.dk**0.5
    #     scores = scores.masked_fill(combined_mask == 0, -torch.inf)
    #     return softmax(scores, dim=-1) @ V
    
    def masked_attention(self, Q, K, V):
        KT = K.transpose(-1, -2)
        scores = (Q @ KT) / self.sqrt_dk
        # scores = scores.masked_fill(self.combined_mask == 0, -torch.inf)
        scores = scores.masked_fill(self.combined_mask == 0, torch.finfo(scores.dtype).min)
        return softmax(scores, dim=-1) @ V

    
    # def attention2(self, Q, K, V):
    #     KT = K.transpose(-1, -2)
    #     return softmax( (Q @ KT) / self.dk**0.5, dim=-1 ) @ V
    
    def attention2(self, Q, K, V):
        KT = K.transpose(-1, -2)
        scores = (Q @ KT) / self.sqrt_dk

        if self.padding_mask is not None:
            # padding_mask shape: (batch, seq_len)
            # expand to (batch, 1, 1, seq_len) to broadcast over heads and queries
            expanded_mask = self.padding_mask.unsqueeze(1).unsqueeze(2)
            # scores = scores.masked_fill(expanded_mask == 0, -torch.inf)
            scores = scores.masked_fill(expanded_mask == 0, torch.finfo(scores.dtype).min)

        return softmax(scores, dim=-1) @ V

    def compute_masked_heads(self, X):
        # X : (batch, seq_len, d_model)

        Q, K, V = self.compute_QKV(X)

        H_ = self.masked_attention(Q, K, V)        # (batch, h, seq_len, dk)

        # reorder to (batch, seq_len, h, dk) then flatten heads
        H__ = H_.permute(0, 2, 1, 3)    # (batch, seq_len, h, dk)
        batch_size, seq_len = X.shape[:2]
        H = H__.reshape(batch_size, seq_len, self.d_model)   # (batch, seq_len, d_model)
        
        return H
    
    def compute_heads2(self, X, X_encoder):
        # X : (batch, seq_len, d_model)

        Q, K, V = self.compute_QKV2(X, X_encoder)

        H_ = self.attention2(Q, K, V)        # (batch, h, seq_len, dk)

        # reorder to (batch, seq_len, h, dk) then flatten heads
        H__ = H_.permute(0, 2, 1, 3)    # (batch, seq_len, h, dk)
        batch_size, seq_len = X.shape[:2]
        H = H__.reshape(batch_size, seq_len, self.d_model)   # (batch, seq_len, d_model)
        
        return H
    
    
    def masked_multiheaded_attention(self, X):
        # X : (batch, seq_len, d_model)
        H = self.compute_masked_heads(X)      # (batch, seq_len, d_model)
        return H @ self.WO_masked                     # (batch, seq_len, d_model)
    
    def multiheaded_attention2(self, X, X_encoder):
        # X : (batch, seq_len, d_model)
        H = self.compute_heads2(X, X_encoder)      # (batch, seq_len, d_model)
        return H @ self.WO                        # (batch, seq_len, d_model)


    def add_norm(self, X, Y, ln):
        Z = X + Y
        ZNorm = ln(Z)
        return ZNorm


    # def feed_forward(self, curr_input):
    #     for layer in self.ff_layers:
    #         curr_input = layer.feed(curr_input)
    #     return curr_input


    def linear_feed(self, ZNorm3):
        return ZNorm3 @ self.linear


    def gen_mask(self, X):
        _, seq_len, _ = X.shape
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device_type)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)    
        if self.padding_mask is not None:
            padding_mask = self.padding_mask.unsqueeze(1).unsqueeze(2).bool()  # (B, 1, 1, S)
            self.combined_mask = padding_mask & causal_mask
        else:
            self.combined_mask = causal_mask    


    def feed(self, X, X_encoder):
        
        self.gen_mask(X)

        MMH_A = self.masked_multiheaded_attention(X)
        
        ZNorm1 = self.add_norm(X, MMH_A, self.ln_1)

        MH_A = self.multiheaded_attention2(ZNorm1, X_encoder)
        
        ZNorm2 = self.add_norm(ZNorm1, MH_A, self.ln_2)
        
        batch_size, seq_len = X.shape[:2]
        # ZFF = self.feed_forward(
        #     ZNorm2.reshape(-1, self.d_model)
        #     ).reshape(batch_size, seq_len, self.d_model)
        ZFF = self.ff.forward(
            ZNorm2.reshape(-1, self.d_model), training=True
            ).reshape(batch_size, seq_len, self.d_model)
        
        ZNorm3 = self.add_norm(ZNorm2, ZFF, self.ln_3)
        
        return ZNorm3

        



if __name__ == "__main__":
    batch_size_ = 2
    sequence_len = 3
    embedding_size = 4
    num_heads = 2
    xe = torch.rand(size=(batch_size_, sequence_len, embedding_size))
    xd = torch.rand(size=(batch_size_, sequence_len, embedding_size))

    from encoder import Encoder
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
    
    # out = encoder.feed(xe)
    # print(out)
    # exit()    

    # decoder.feed(xd, out)
    
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
