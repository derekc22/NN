import os
from src.network import Network
import torch
from src.encoder import Encoder
from src.decoder import Decoder
from utils.functions import softmax
from typing_extensions import override





class Transformer(Network):

    def __init__(self, pretrained, training, device, **kwargs):
                
        super().__init__(model_type="transformer", training=training, **kwargs)
        
        self.device = torch.device(device)
        self.save_fpath = kwargs.get("save_fpath")
        
        specifications = kwargs.get("specifications")
        self.type = specifications.get("type") # options are: encoder, decoder, encoder-decoder (ie default option)
        self.tokenizer = specifications.get("tokenizer")
        
        if not pretrained:
            self.layers, self.encoders, self.decoders = self.build_layers(
                kwargs.get("architecture"), 
                self.init_ff(pretrained, training, **kwargs)
            )
            
        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.set_optimizer()


    def init_ff(self, pretrained, training, **kwargs):
        if not pretrained:
            ff_architecture = kwargs.get("ff_architecture")
            ff_architecture["input_feature_count"] = kwargs.get("architecture").get("d_model")
            return {
                "pretrained": False,
                "training": training,
                "device": self.device,
                "architecture": ff_architecture,
                "hyperparameters": kwargs.get("ff_hyperparameters"),
                "save_fpath": kwargs.get("save_fpath"),
                "specifications": None,
            }


    def build_layers(self, architecture, ff_config):
        
        num_heads = architecture.get("num_heads")
        depth = architecture.get("depth")
        vocab_size = self.tokenizer.vocab_size
        encoder_padding_mask = architecture.get("encoder_padding_mask")
        decoder_padding_mask = architecture.get("decoder_padding_mask")
        d_model = architecture.get("d_model")
        
        
        encoders = [
            Encoder(
            pretrained=False,
            device=self.device,
            d_model=d_model,
            num_heads=num_heads,
            padding_mask=encoder_padding_mask,
            ff_config=ff_config,
            index=i+1) for i in range(depth)
        ]
        
        decoders = [
            
            Decoder(
            pretrained=False,
            device=self.device,
            type="hidden",
            d_model=d_model,
            num_heads=num_heads,
            padding_mask=decoder_padding_mask,
            ff_config=ff_config,
            index=i+1) for i in range(depth, 2*depth-1) ] + [
            
            Decoder(
            pretrained=False,
            device=self.device,
            type="output",
            d_model=d_model,
            num_heads=num_heads,
            padding_mask=decoder_padding_mask,
            vocab_size=vocab_size,
            ff_config=ff_config,
            index=2*depth)
        
        ]
        
        layers = encoders + decoders
        
        return layers, encoders, decoders


    def save_parameters(self, qualifier=""):
        save_fpath = f"{self.save_fpath}/{qualifier}"
        os.makedirs(save_fpath, exist_ok=True)
        for layer in self.layers:
            layer.index = str(layer.index).zfill(2)
                
            if layer.component == "encoder":
                torch.save(layer.WQKV, f"{save_fpath}/encoder_{layer.index}_WQKV.pth")
            if layer.component == "decoder":
                torch.save(layer.WQKV_masked, f"{save_fpath}/decoder_{layer.index}_WQKV_masked.pth")
                torch.save(layer.WO_masked, f"{save_fpath}/decoder_{layer.index}_WO_masked.pth")
                torch.save(layer.WQ, f"{save_fpath}/decoder_{layer.index}_WQ.pth")
                torch.save(layer.WKV, f"{save_fpath}/decoder_{layer.index}_WKV.pth")
                if layer.type == "output":
                    layer.linear.grad = None
                    torch.save(layer.linear, f"{save_fpath}/decoder_{layer.index}_linear.pth")
            torch.save(layer.WO, f"{save_fpath}/{layer.component}_{layer.index}_WO.pth") 
            layer.ff.save_parameters(qualifier=f"ff/{layer.component}_{layer.index}")




    def forward_inference(self, **kwargs):

        src_emb  = kwargs.get("src_emb")

        device      = src_emb.device
        batch_size  = src_emb.size(0)

        bos_id = (self.tokenizer.cls_token_id
                  if self.tokenizer.cls_token_id is not None
                  else self.tokenizer.bos_token_id)
        eos_id = (self.tokenizer.sep_token_id
                  if self.tokenizer.sep_token_id is not None
                  else self.tokenizer.eos_token_id)

        # ------------------------------------------------------------------ #
        # 2. Encode source sequence once
        # ------------------------------------------------------------------ #
        curr_input = src_emb
        for encoder in self.encoders:
            curr_input = encoder.feed(curr_input)
        X_encoder = curr_input                                            # (B, S, d_model)

        # ------------------------------------------------------------------ #
        # 3. Initialise decoder with BOS
        # ------------------------------------------------------------------ #
        decoder_input_ids = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=device
        )                                                                 # (B, 1)

        generated = [decoder_input_ids]                                   # list of tensors

        # ------------------------------------------------------------------ #
        # 4. Autoregressive loop
        # ------------------------------------------------------------------ #
        for _ in range(max_len - 1):

            tgt_emb = self.tgt_embedding(decoder_input_ids)               # (B, T, d_model)

            curr_input = tgt_emb
            for decoder in self.decoders:
                curr_input = decoder.feed(curr_input, X_encoder)
            ZNorm3 = curr_input                                           # (B, T, d_model)

            # Project only the last time step
            step_logits = self.decoders[-1].linear_feed(ZNorm3[:, -1:, :])  # (B, 1, V)
            next_id     = step_logits.argmax(dim=-1)                        # (B, 1)

            generated.append(next_id)

            # Stop early if every sequence just produced EOS
            if (next_id == eos_id).all():
                break

            decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=1)  # (B, T+1)

        # ------------------------------------------------------------------ #
        # 5. Concatenate and return
        # ------------------------------------------------------------------ #
        return torch.cat(generated, dim=1)                                # (B, L_generated)

        

    def forward(self, training, **kwargs):
        if training:
            return self.forward_train(**kwargs)
        return self.forward_inference(**kwargs)


    def forward_train(self, **kwargs):
        
        src_emb = kwargs.get("src_emb")
        
        if self.type == "encoder":
            return self.forward_train_encoder_only(src_emb)
        
        tgt_emb = kwargs.get("tgt_emb")
        
        if self.type == "decoder":
            return self.forward_train_decoder_only(tgt_emb)    
            
        return self.forward_train_encoder_decoder(src_emb, tgt_emb)

        

    def forward_train_encoder_decoder(self, curr_input, tgt_emb):
        
        for encoder in self.encoders:
            curr_input = encoder.feed(curr_input)
            
        X_encoder = curr_input
        curr_input = tgt_emb
            
        for decoder in self.decoders:
            curr_input = decoder.feed(curr_input, X_encoder)
        ZNorm3 = curr_input # store final decoder output 
        
        logits = self.decoders[-1].linear_feed(ZNorm3) # pass final decoder output to linear block and take softmax
        
        return logits.view(-1, self.tokenizer.vocab_size)
        
        
        
    def forward_train_encoder_only(self, curr_input) -> None:
        pass

    def forward_train_decoder_only(self, curr_input) -> None:
        pass
        


    @override
    def inference(self, **kwargs):
        with torch.no_grad():
            return self.forward(training=False, **kwargs)

        
        
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
            

        if save_params:
            self.save_parameters()
            
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




