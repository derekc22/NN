import os
from src.network import Network
import torch
from src.encoder import Encoder
from src.decoder import Decoder
from utils.functions import softmax





class Transformer(Network):

    def __init__(self, pretrained, device_type, training, **kwargs):
        
        super().__init__(model_type="transformer", training=training, kwargs=kwargs)

        self.device_type = torch.device(device_type)
        self.type = kwargs.get("type", "both") # options are: encoder, decoder, both (ie default option)

        if not pretrained:
            architecture = kwargs.get("architecture")
            self.input_feature_count = kwargs.get("input_feature_count")
            self.layers, self.encoders, self.decoders = self.build_layers(architecture=architecture)

        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.set_optimizer()



    def build_layers(self, architecture):
        
        num_heads = architecture.get("num_heads")
        ff_neuron_count = architecture.get("ff_neuron_count")
        ff_activation_fn = architecture.get("ff_activation_fn")
        depth = architecture.get("depth")
        vocab_size = architecture.get("vocab_size")

        encoders = [
            Encoder(
            pretrained=False,
            device_type=self.device_type,
            embedding_size=self.input_feature_count,
            num_heads=num_heads,
            ff_neuron_count=ff_neuron_count,
            ff_nonlinearity=ff_activation_fn,
            index=i+1) for i in range(depth)
        ]
        
        decoders = [
            
            Decoder(
            pretrained=False,
            device_type=self.device_type,
            type="hidden",
            embedding_size=self.input_feature_count,
            num_heads=num_heads,
            ff_neuron_count=ff_neuron_count,
            ff_nonlinearity=ff_activation_fn,
            index=i+1) for i in range(depth-1) ] + [
            
            Decoder(
            pretrained=False,
            device_type=self.device_type,
            type="output",
            embedding_size=self.input_feature_count,
            num_heads=num_heads,
            ff_neuron_count=ff_neuron_count,
            ff_nonlinearity=ff_activation_fn,
            vocab_size=vocab_size,
            index=depth)
        
        ]
        
        layers = encoders + decoders
        
        return layers, encoders, decoders





    def forward(self, curr_input, training, **kwargs):
        
        if self.type == "encoder":
            return self.forward_encoder_only(curr_input)
        
        out_embedding = kwargs.get("target")
        
        if self.type == "decoder":
            return self.forward_decoder_only(out_embedding)    
            
        return self.forward_encoder_decoder(curr_input, out_embedding)

        

    def forward_encoder_decoder(self, curr_input, out_embedding):
        
        for encoder in self.encoders:
            curr_input = encoder.feed(curr_input)
            
        X_encoder = curr_input
        curr_input = out_embedding
            
        for decoder in self.decoders:
            curr_input = decoder.feed(curr_input, X_encoder)
            
        ZNorm3 = curr_input # store final decoder output 
        
        logits = softmax(decoder.linear_feed(ZNorm3), dim=-1) # pass final decoder output to linear block and take softmax
        
        return logits
        
        
    def forward_encoder_only(self, curr_input) -> None:
        pass

    def forward_decoder_only(self, curr_input) -> None:
        pass
        


        
        
        




    # if __name__ == "__main__":
    #     a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    #     b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    #     print(a)
    #     print(b)
    #     print(a * b)  # or torch.mul(a, b)
    #     print(a @ b)
    #     # Output: tensor([[ 5.0, 12.0], [21.0, 32.0]])




