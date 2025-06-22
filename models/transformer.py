import os
from src.network import Network
import torch
import jax
from src.encoder import Encoder
from src.decoder import Decoder





class Transformer(Network):

    def __init__(self, pretrained, device_type, training, **kwargs):
        
        super().__init__(model_type="transformer", training=training, kwargs=kwargs)

        self.device_type = torch.device(device_type)




def buildTransformer(self, architecture):

    layers = [
    
    Encoder(
        pretrained=False,
        device_type=self.device_type,
    )


    ]






def forward(curr_input):

    for encoder in self.encoders:
        curr_input = encoder.multiHeadedSelfAttention(curr_input)

    for decoder in self.decoders:
        curr_input = decoder.




# if __name__ == "__main__":
#     a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
#     b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
#     print(a)
#     print(b)
#     print(a * b)  # or torch.mul(a, b)
#     print(a @ b)
#     # Output: tensor([[ 5.0, 12.0], [21.0, 32.0]])




