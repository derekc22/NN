import os
from src.network import Network
import torch




class Transformer(Network):

    def __init__(self, pretrained, device_type, training, **kwargs):
        
        super().__init__(model_type="transformer", training=training, kwargs=kwargs)

        self.device_type = torch.device(device_type)
        self.num_heads = kwargs.get("num_heads")






def multiHeadedSelfAttention(curr_input):
    Q_ = self.WQ @ curr_input
    K_ = self.WK @ curr_input
    V_ = self.WV @ curr_input



def forward(curr_input):

    multiHeadedSelfAttention(curr_input)



# if __name__ == "__main__":
#     a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
#     b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
#     print(a)
#     print(b)
#     print(a * b)  # or torch.mul(a, b)
#     print(a @ b)
#     # Output: tensor([[ 5.0, 12.0], [21.0, 32.0]])




