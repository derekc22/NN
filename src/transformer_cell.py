import torch
import numpy as np


class TransformerCell():

    def __init__(self, pretrained, device_type, **kwargs):

        self.device_type = device_type
        self.embedding_size = kwargs.get("embedding_size")

        stddev_embedding = np.sqrt(2 / (self.embedding_size))

        if not pretrained:
            self.WQ = torch.normal(
                0, stddev_embedding,
                size=(self.embedding_size, self.embedding_size),
                dtype=torch.float32,
                device=self.device_type)
            
            self.WK = torch.normal(
                0, stddev_embedding,
                size=(self.embedding_size, self.embedding_size),
                dtype=torch.float32,
                device=self.device_type)
            
            self.WV = torch.normal(
                0, stddev_embedding,
                size=(self.embedding_size, self.embedding_size),
                dtype=torch.float32,
                device=self.device_type)
