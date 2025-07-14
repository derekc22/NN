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
                kwargs.get("architecture"), specifications,
                self.init_ff(pretrained, training, **kwargs))
        else:
            self.layers, self.encoders, self.decoders = self.load_layers(
                training, kwargs.get("params"), specifications)
            
        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.set_optimizer()





    def load_layers(self, training, params, specifications):

        encoder_params = params["encoder"]
        decoder_params = params["decoder"]
        
        encoder_padding_mask = specifications.get("encoder_padding_mask")
        decoder_padding_mask = specifications.get("decoder_padding_mask")

        encoders = [
            Encoder(
            pretrained=True,
            device=self.device,
            pretrained_WQKV=WQKV,
            pretrained_WO=WO,
            num_heads=num_heads,
            ff_config=self.init_ff(pretrained=True, training=training, ff_params=ff_params),
            padding_mask=encoder_padding_mask,
            index=index) for (WQKV, WO, num_heads, ff_params, index) in encoder_params.values()
        ]
        
        decoders = [
            Decoder(
            pretrained=True,
            device=self.device,
            type="hidden",
            pretrained_WQKV_masked=WQKV_masked,
            pretrained_WO_masked=WO_masked,
            pretrained_WQ=WQ,
            pretrained_WKV=WKV,
            pretrained_WO=WO,
            num_heads=num_heads,
            ff_config=self.init_ff(pretrained=True, training=training, ff_params=ff_params),
            padding_mask=decoder_padding_mask,
            index=index) for (WQKV_masked, WO_masked, WQ, WKV, WO, num_heads, ff_params, index) in list(decoder_params.values())[:-1] ] + [
            
            Decoder(
            pretrained=True,
            device=self.device,
            type="output",
            pretrained_WQKV_masked=WQKV_masked,
            pretrained_WO_masked=WO_masked,
            pretrained_WQ=WQ,
            pretrained_WKV=WKV,
            pretrained_WO=WO,
            pretrained_linear=linear,
            num_heads=num_heads,
            ff_config=self.init_ff(pretrained=True, training=training, ff_params=ff_params),
            padding_mask=decoder_padding_mask,
            index=index) for (WQKV_masked, WO_masked, WQ, WKV, WO, num_heads, ff_params, linear, index) in [list(decoder_params.values())[-1]]
        ]
        
        layers = encoders + decoders
        
        return layers, encoders, decoders


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
                "save_fpath": kwargs.get("ff_save_fpath"),
                "specifications": None
            }
        else:
        
            return {
                "pretrained": True,
                "training": training,
                "device": self.device,
                "params": kwargs.get("ff_params"),
                "hyperparameters": kwargs.get("ff_hyperparameters"),
                "save_fpath": kwargs.get("ff_save_fpath"),
                "specifications": None
            }



    def build_layers(self, architecture, specifications, ff_config):
        
        num_heads = architecture.get("num_heads")
        depth = architecture.get("depth")
        d_model = architecture.get("d_model")
        encoder_padding_mask = specifications.get("encoder_padding_mask")
        decoder_padding_mask = specifications.get("decoder_padding_mask")
        
        
        encoders = [
            Encoder(
            pretrained=False,
            device=self.device,
            d_model=d_model,
            num_heads=num_heads,
            ff_config=ff_config,
            padding_mask=encoder_padding_mask,
            index=i+1) for i in range(depth)
        ]
        
        decoders = [
            Decoder(
            pretrained=False,
            device=self.device,
            type="hidden",
            d_model=d_model,
            num_heads=num_heads,
            ff_config=ff_config,
            padding_mask=decoder_padding_mask,
            index=i+1) for i in range(depth, 2*depth-1) ] + [
            
            Decoder(
            pretrained=False,
            device=self.device,
            type="output",
            d_model=d_model,
            num_heads=num_heads,
            ff_config=ff_config,
            vocab_size=self.tokenizer.vocab_size,
            padding_mask=decoder_padding_mask,
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
                    torch.save(layer.linear, f"{save_fpath}/decoder_{layer.index}_linear.pth")
            torch.save(layer.WO, f"{save_fpath}/{layer.component}_{layer.index}_WO_h_{layer.h}.pth") 
            layer.ff.save_parameters(qualifier=f"{layer.component}_{layer.index}")


    def forward_inference(self, **kwargs):
        
        src_emb = kwargs.get("src_emb")
        
        if self.type == "encoder":
            return self.forward_inference_encoder_only(src_emb)
        
        tgt_emb = kwargs.get("tgt_emb")
        
        if self.type == "decoder":
            return self.forward_inference_decoder_only(tgt_emb)    
            
        return self.forward_inference_encoder_decoder(src_emb, tgt_emb)
        
        
    def forward_inference_encoder_decoder(self, src_emb, tgt_emb):
        """
        Greedy inference for the full encoder-decoder architecture.

        Parameters
        ----------
        curr_input : torch.Tensor
            Source sequence embeddings, shape (B, S_src, D).
        tgt_emb : torch.Tensor
            Decoder input embeddings, shape (B, S_tgt, D). Typically contains
            only the <bos> token at timestep zero but any prefix may be given.

        Returns
        -------
        torch.Tensor
            Predicted token indices after greedy decoding, shape (B, S_tgt).
        """

        # 1. Encode source sequence (only done once)
        curr_input = src_emb
        for encoder in self.encoders:
            curr_input = encoder.feed(curr_input)
        X_encoder = curr_input
        
        # 2. Start with just the BOS token from tgt_emb
        batch_size = src_emb.shape[0]
        bos_embedding = tgt_emb[:, 0:1, :]  # Just take the first token (BOS)
        
        # 3. Initialize sequence with BOS token
        generated_sequence = bos_embedding
        max_length = 128  # Or whatever maximum length you want
        
        # 4. Generate tokens autoregressively
        for i in range(max_length - 1):
            # Run through decoders
            curr_input = generated_sequence
            for decoder in self.decoders:
                curr_input = decoder.feed(curr_input, X_encoder)
            
            # Get logits for the next token position
            logits = self.decoders[-1].linear_feed(curr_input)  # (B, current_len, V)
            next_token_logits = logits[:, -1:, :]  # (B, 1, V)
            
            # Select next token (greedy)
            probs = softmax(next_token_logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1)  # (B, 1)
            
            # Convert token ID back to embedding
            next_token_emb = self.tokenizer.embedding(next_token_id)  # (B, 1, D)
            next_token_emb = self.tokenizer.positional_encoding(next_token_emb)
            
            # Append to generated sequence
            generated_sequence = torch.cat([generated_sequence, next_token_emb], dim=1)
            
            # Optional: Check for EOS tokens and stop early
            # if all sequences have generated EOS, break
    
        # 5. Convert token IDs to text
        # Extract token IDs from the generated sequence
        logits = self.decoders[-1].linear_feed(curr_input)
        probs = softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        
        # Convert to text
        texts = self.ids_to_text(pred_ids)
        return texts
        

    def ids_to_text(self, pred_ids: torch.Tensor) -> list[str]:
        """
        Convert a (B, S) tensor of token indices into decoded strings.
        """
        tok = self.tokenizer.tokenizer          # AutoTokenizer
        sequences = []
        for row in pred_ids:                    # iterate over batch dimension
            ids = row.tolist()
            # Optional: truncate at first PAD or EOS
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id)]
            sequences.append(
                tok.decode(ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=True)
            )
        return sequences
        

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
            print("__________________________________________")
            

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


