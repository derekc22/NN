import os
from src.network import Network
import torch
from src.encoder import Encoder
from src.decoder import Decoder





class Transformer(Network):

    def __init__(self, pretrained, device_type, training, **kwargs):
        
        super().__init__(model_type="transformer", training=training, kwargs=kwargs)

        self.device_type = torch.device(device_type)
        self.type = kwargs.get("type", "both") # options are: encoder, decoder, both (ie default option)

        if not pretrained:
            architecture = kwargs.get("architecture")
            self.input_feature_count = kwargs.get("input_feature_count")
            self.layers, self.encoders, self.decoders = self.buildLayers(architecture=architecture)

        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.setOptimizer()



    def buildLayers(self, architecture):
        
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





    def forward(self, curr_input, training):
        if self.type == "encoder":
            return self.forwardEncoderOnly(curr_input)
        if self.type == "decoder":
            return self.forwardDecoderOnly(curr_input)
        return self.forwardEncoderDecoder(curr_input)

        

    def forwardEncoderDecoder(self, curr_input):
        
        X_decoder = curr_input.detach().clone()

        for encoder in self.encoders:
            print(curr_input.shape)
            # import time
            # time.sleep(5)
            curr_input = encoder.feed(curr_input)
            
        X_encoder = curr_input
        curr_input = X_decoder
            
        for decoder in self.decoders:
            curr_input = decoder.feed(curr_input, X_encoder)
            
        ZNorm3 = curr_input # store final decoder output 
        
        out = decoder.linearFeed(ZNorm3) # pass final decoder output to linear block
        
        return out
        
        
    def forwardEncoderOnly(self, curr_input) -> None:
        pass

    def forwardDecoderOnly(self, curr_input) -> None:
        pass
        


        
        
        




    # if __name__ == "__main__":
    #     a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    #     b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    #     print(a)
    #     print(b)
    #     print(a * b)  # or torch.mul(a, b)
    #     print(a @ b)
    #     # Output: tensor([[ 5.0, 12.0], [21.0, 32.0]])




