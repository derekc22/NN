import torch
from src.network import Network
from src.rnn_cell import RNNCell
from utils.functions import activate
import os


class RNN(Network):
    
    def __init__(self, pretrained, device, training, **kwargs):

        super().__init__(model_type="rnn", training=training, **kwargs)

        self.device = torch.device(device)
        self.save_fpath = kwargs.get("save_fpath")
        
        specifications = kwargs.get("specifications")
        self.stateful = specifications.get("stateful")
        self.autoregressive = specifications.get("autoregressive")
        self.teacher_forcing = specifications.get("teacher_forcing") # If no teacher_forcing factor is set (ie like at inference), the model should be 100% auto-regressive
        
        if self.stateful:
            self.state_initialized = False

        if not pretrained:
            self.layers = self.build_layers(kwargs.get("architecture"))
        else:
            self.layers = self.load_layers(kwargs.get("params"))

        self.whh_activation = self.layers[0].whh_activation
        self.why_activation = self.layers[-1].why_activation
        
        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.set_optimizer()





    def load_layers(self, params):
        
        layers = [ 
            
            RNNCell( 
            pretrained=True, 
            device=self.device,
            type="hidden",
            pretrained_wxh=wxh,
            pretrained_whh=whh,
            pretrained_bh=bh,
            hidden_activation=hidden_activation,
            index=index ) for (wxh, whh, bh, hidden_activation, index) in list(params.values())[:-1] ] + [ 
                
            RNNCell(
            pretrained=True, 
            device=self.device,
            type="output",
            pretrained_wxh=wxh,
            pretrained_whh=whh,
            pretrained_bh=bh,
            hidden_activation=hidden_activation,
            pretrained_why=why,
            pretrained_by=by,
            output_activation=output_activation,
            index=index ) for (wxh, whh, bh, hidden_activation, why, by, output_activation, index) in [list(params.values())[-1]] ]
    
        return layers



    def build_layers(self, architecture):
        hidden_state_neuron_counts = architecture.get("hidden_state_neuron_counts")
        hidden_activation = architecture.get("hidden_activation")
        output_activation = architecture.get("output_activation")
        output_feature_count = architecture.get("output_feature_count")
        num_layers = len(hidden_state_neuron_counts)
        input_feature_count = architecture.get("input_feature_count")

        layers = [ 
            
            RNNCell( 
            pretrained=False, 
            device=self.device,
            type="hidden",
            wxh_input_count=input_feature_count if i == 0 else hidden_state_neuron_counts[i-1],
            wxh_neuron_count=hidden_state_neuron_counts[i],
            whh_input_count=hidden_state_neuron_counts[i],
            whh_neuron_count=hidden_state_neuron_counts[i],
            hidden_activation=hidden_activation,
            index=i+1 ) for i in range(num_layers-1) ] + [ 
                    
            RNNCell(
            pretrained=False, 
            device=self.device,
            type="output",
            wxh_input_count=input_feature_count if num_layers == 1 else hidden_state_neuron_counts[-2],
            wxh_neuron_count=hidden_state_neuron_counts[-1],
            whh_input_count=hidden_state_neuron_counts[-1],
            whh_neuron_count=hidden_state_neuron_counts[-1],
            hidden_activation=hidden_activation,
            why_input_count=hidden_state_neuron_counts[-1],
            why_neuron_count=output_feature_count,
            output_activation=output_activation,
            index=num_layers ) 
        ]
        
        return layers
    

    def save_parameters(self, qualifier=""):
        save_fpath = f"{self.save_fpath}/{qualifier}"
        os.makedirs(save_fpath, exist_ok=True)
        for layer in self.layers:
            #layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
            layer.index = str(layer.index).zfill(2)
            torch.save(layer.wxh, f"{save_fpath}/layer_{layer.index}_wxh.pth")
            torch.save(layer.whh, f"{save_fpath}/layer_{layer.index}_whh_{layer.whh_activation}.pth")
            torch.save(layer.bh, f"{save_fpath}/layer_{layer.index}_bh.pth")
            if layer.type == "output":
                torch.save(layer.why, f"{save_fpath}/layer_{layer.index}_why_{layer.why_activation}.pth")
                torch.save(layer.by, f"{save_fpath}/layer_{layer.index}_by.pth")


 



    def reset_hidden_state(self, batch_size):
        return [torch.zeros(
            size=(batch_size, layer.wxh_neuron_count), 
            dtype=torch.float32, 
            device=self.device) for layer in self.layers]



    def forward_non_autoregressive(self, X, training):
        """
        No autoregressive handling - working
        """
    
        T = X.shape[1] 
        batch_size = X.shape[0]

        if self.stateful and not self.state_initialized: 
            for layer in self.layers: 
                layer.generate_state(batch_size)
            self.state_initialized = True

        ht1_l = [layer.ht1 for layer in self.layers] if self.stateful else self.reset_hidden_state(batch_size)
        whh_l = [layer.whh for layer in self.layers]
        wxh_l = [layer.wxh for layer in self.layers]
        bh_l  = [layer.bh  for layer in self.layers]
        why = self.layers[-1].why
        by = self.layers[-1].by


        output_feature_count = by.shape[-1]
        Y = torch.zeros(batch_size, T, output_feature_count, device=self.device)


        for t in range(T):

            x = X[:, t, :]

            ht_l = [None] * self.num_layers
            ht_l[0] = activate(
                ht1_l[0] @ whh_l[0] + bh_l[0] + 
                x @ wxh_l[0], self.whh_activation)

            for i in range(1, self.num_layers):
                ht_l[i] = activate(
                    ht1_l[i] @ whh_l[i] + bh_l[i] +
                    ht_l[i-1] @ wxh_l[i], self.whh_activation)

            Y[:, t, :] = activate( 
                ht_l[-1] @ why + by, self.why_activation)
            
            ht1_l = ht_l
        
        # time.sleep(1)
        if self.stateful:
            for (layer, ht1_i) in zip(self.layers, ht1_l):
                layer.ht1 = ht1_i.detach()

        # print(ht_l[0])
        return Y
 

    def forward_autoregressive(self, X, training):
        
        T = X.shape[1] 
        batch_size = X.shape[0]

        if self.stateful and not self.state_initialized: 
            for layer in self.layers: 
                layer.generate_state(batch_size)
            self.state_initialized = True

        ht1_l = [layer.ht1 for layer in self.layers] if self.stateful else self.reset_hidden_state(batch_size)
        whh_l = [layer.whh for layer in self.layers]
        wxh_l = [layer.wxh for layer in self.layers]
        bh_l  = [layer.bh  for layer in self.layers]
        why = self.layers[-1].why
        by = self.layers[-1].by

        output_feature_count = by.shape[0]
        Y = torch.zeros(batch_size, T, output_feature_count, device=self.device)

        x = X[:, 0, :]
        # print(x.shape)
        # exit()
        for t in range(T):

            # ht_l = [None] * len(ht1_l)
            ht_l = [None] * self.num_layers
            ht_l[0] = activate(
                ht1_l[0] @ whh_l[0] + bh_l[0] + 
                x @ wxh_l[0], self.whh_activation)

            # for i in range(1, len(ht_l)):
            for i in range(1, self.num_layers):
                ht_l[i] = activate(
                    ht1_l[i] @ whh_l[i] + bh_l[i] +
                    ht_l[i-1] @ wxh_l[i], self.whh_activation)

            y = activate( 
                ht_l[-1] @ why + by, self.why_activation)
            Y[:, t, :] = y
            # print(y.shape)

            ht1_l = ht_l

            if training:
                # 0 means fully autoregressive, 1 means fully teacher-forced
                teacher_forcing_factor = (
                    self.epoch/self.epochs if (self.teacher_forcing == "progressive") # progressive teacher-forcing
                    else 1 - self.epoch/self.epochs if (self.teacher_forcing == "regressive") # regressive teacher-forcing
                    else float(self.teacher_forcing) if isinstance(self.teacher_forcing, (int, float)) 
                    else 0
                )
                x = X[:, t+1, :] if t < teacher_forcing_factor*(T-1) else y.detach() 
                """SHOULD THIS BE t OR t+1 (I believe it should be t+1)"""
            else: x = y.detach()

            # print(x.shape)
            # exit()

        if self.stateful:
            for (layer, ht1_i) in zip(self.layers, ht1_l):
                layer.ht1 = ht1_i.detach()

        return Y
    

    

    def forward(self, X, training):
        """ if this is uncommented, it will force the pure teacher forcing LSTM implementations to run inference auto regressively (which i believe is the correct way to do it)
        however, since pure auto regressive inference currently sucks (both for the auto regressive and teacher forced implementations), i will leave it commented out such that I can at least see good results at inference for the teacher forced implementations
        but yeah i think this would be the more correct way to do it (i.e. to always run inference auto regressively) """
        # if self.autoregressive or not training: 
        if self.autoregressive:
            return self.forward_autoregressive(X, training)
        return self.forward_non_autoregressive(X, training)





    # def backprop(self, loss):
    #     self.zerograd()

    #     loss.backward()

    #     if self.grad_clip_norm is not None or self.grad_clip_value is not None:
    #         self.clip_gradients()

    #     with torch.no_grad():

    #         if not self.optimizer:
    #             self.update()
    #         else:
    #             self.t += 1
    #             self.optimizer_update()
