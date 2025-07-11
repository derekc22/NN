import os
from src.network import Network
import torch
from src.encoder import Encoder
from src.decoder import Decoder
from utils.functions import softmax
from typing_extensions import override





class Transformer(Network):

    def __init__(self, pretrained, device_type, training, **kwargs):
        
        super().__init__(model_type="transformer", training=training, **kwargs)

        self.device_type = torch.device(device_type)
        self.type = kwargs.get("type", "encoder-decoder") # options are: encoder, decoder, encoder-decoder (ie default option)
        self.tokenizer = kwargs.get("tokenizer")
        self.save_fpath = kwargs.get("save_fpath")
        ff_save_fpath = kwargs.get("ff_save_fpath")

        if not pretrained:
            architecture = kwargs.get("architecture")
            self.input_feature_count = kwargs.get("input_feature_count")
            self.layers, self.encoders, self.decoders = self.build_layers(architecture, kwargs.get("ff_architecture"), kwargs.get("ff_hyperparameters"), ff_save_fpath)
            
        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.set_optimizer()



    def build_layers(self, architecture, ff_architecture, ff_hyperparameters, ff_save_fpath):
        
        num_heads = architecture.get("num_heads")
        # ff_neuron_count = architecture.get("ff_neuron_count")
        # ff_activation_fn = architecture.get("ff_activation_fn")
        depth = architecture.get("depth")
        vocab_size = architecture.get("vocab_size")
        encoder_padding_mask = architecture.get("encoder_padding_mask")
        decoder_padding_mask = architecture.get("decoder_padding_mask")
        
        encoders = [
            Encoder(
            pretrained=False,
            device_type=self.device_type,
            embedding_size=self.input_feature_count,
            num_heads=num_heads,
            # ff_neuron_count=ff_neuron_count,
            # ff_nonlinearity=ff_activation_fn,
            ff_architecture=ff_architecture,
            ff_hyperparameters=ff_hyperparameters,
            ff_save_fpath=ff_save_fpath,
            padding_mask=encoder_padding_mask,
            index=i+1) for i in range(depth)
        ]
        
        decoders = [
            
            Decoder(
            pretrained=False,
            device_type=self.device_type,
            type="hidden",
            embedding_size=self.input_feature_count,
            num_heads=num_heads,
            # ff_neuron_count=ff_neuron_count,
            # ff_nonlinearity=ff_activation_fn,
            ff_architecture=ff_architecture,
            ff_hyperparameters=ff_hyperparameters,
            ff_save_fpath=ff_save_fpath,
            padding_mask=decoder_padding_mask,
            index=i+1) for i in range(depth-1) ] + [
            
            Decoder(
            pretrained=False,
            device_type=self.device_type,
            type="output",
            embedding_size=self.input_feature_count,
            num_heads=num_heads,
            # ff_neuron_count=ff_neuron_count,
            # ff_nonlinearity=ff_activation_fn,
            ff_architecture=ff_architecture,
            ff_hyperparameters=ff_hyperparameters,
            ff_save_fpath=ff_save_fpath,
            padding_mask=decoder_padding_mask,
            vocab_size=vocab_size,
            index=depth)
        
        ]
        
        layers = encoders + decoders
        
        return layers, encoders, decoders


    # def save_parameters(self):
    #     os.makedirs(f"{self.save_fpath}", exist_ok=True)
    #     for layer in self.layers:
    #         layer.index = str(layer.index).zfill(2)
    #         torch.save(layer.WQ, f"{self.save_fpath}/layer_{layer.index}_WQ.pth")
    #         torch.save(layer.WK, f"{self.save_fpath}/layer_{layer.index}_WK.pth")
    #         torch.save(layer.WQ, f"{self.save_fpath}/layer_{layer.index}_WQ.pth")
    #         torch.save(layer.WQ, f"{self.save_fpath}/layer_{layer.index}_WQ.pth")

    #         if layer.type == "output":
    #             torch.save(layer.why, f"{self.save_fpath}/layer_{layer.index}_why_{layer.why_nonlinearity}.pth")
    #             torch.save(layer.by, f"{self.save_fpath}/layer_{layer.index}_by.pth")


    def forward(self, training, **kwargs):
        
        src_emb = kwargs.get("src_emb")
        
        if self.type == "encoder":
            return self.forward_encoder_only(src_emb, training)
        
        tgt_emb = kwargs.get("tgt_emb")
        
        if self.type == "decoder":
            return self.forward_decoder_only(tgt_emb, training)    
            
        return self.forward_encoder_decoder(src_emb, tgt_emb, training)

        

    def forward_encoder_decoder(self, curr_input, tgt_emb, training):
        
        for encoder in self.encoders:
            curr_input = encoder.feed(curr_input)
            
        X_encoder = curr_input
        curr_input = tgt_emb
            
        for decoder in self.decoders:
            curr_input = decoder.feed(curr_input, X_encoder)
        ZNorm3 = curr_input # store final decoder output 
        
        logits = self.decoders[-1].linear_feed(ZNorm3) # pass final decoder output to linear block and take softmax
        
        if training:
            return logits.view(-1, self.tokenizer.vocab_size)
        return softmax(logits, dim=-1) # probs
        
        
        
    def forward_encoder_only(self, curr_input) -> None:
        pass

    def forward_decoder_only(self, curr_input) -> None:
        pass
        


        
        
    @override
    def train(self, epochs, save_params=True, **kwargs):
        
        # src_emb, tgt_emb, labels = (
        #     kwargs.get("src_emb"),  
        #     kwargs.get("tgt_emb"), 
        #     kwargs.get("labels")
        # )
        
        labels = kwargs.get("labels")
        
        epoch_plt = []
        loss_plt = []
        self.epochs = epochs

        if not self.batch_size: self.batch_size = labels.shape[0]

        for epoch in range(epochs):
            
            self.epoch = epoch+1

            # data_batch, target_batch = self.batch(data, target)
            pred = self.forward(training=True, **kwargs)
            
            loss = getattr(self, self.loss_func)(pred, labels.flatten())

            # if self.lambda_L2:
                # loss += self.l2_regularization()
            self.backprop(loss)


            epoch_plt.append(epoch)
            loss_plt.append(loss.item())
            print(f"epoch = {epoch+1}, loss = {loss}")
            print(f"__________________________________________")
            

        # if save_params:
            # self.save_parameters()
            
        return epoch_plt, loss_plt


    @override
    def backprop(self, loss):

        self.zerograd()
        for layer in self.layers:
            layer.ff.zerograd()


        loss.backward()

        with torch.no_grad():

            if not self.optimizer:
                self.update()
            else:
                self.t += 1
                self.optimizer_update()


            if not self.layers[0].ff.optimizer:
                for layer in self.layers:
                    layer.ff.update()

            else:
                for layer in self.layers:
                    layer.ff.t += 1
                    layer.ff.optimizer_update()




