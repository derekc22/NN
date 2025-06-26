import torch
from src.network import Network
from src.rnn_cell import RNNCell
from src.functions import activate
import os


class RNN(Network):
    
    def __init__(self, pretrained, device_type, training, **kwargs):

        super().__init__(model_type="rnn", training=training, kwargs=kwargs)

        self.device_type = torch.device(device_type)
        self.stateful = kwargs.get("stateful", False)
        self.save_fpath = kwargs.get("save_fpath")
        
        if self.stateful:
            self.stateful_initialized = False
        self.autoregressive = kwargs.get("autoregressive", False)
        self.teacher_forcing = kwargs.get("teacher_forcing", 0) # If no teacher_forcing factor is set (ie like at inference), the model should be 100% auto-regressive

        if not pretrained:
            architecture = kwargs.get("architecture")
            self.input_feature_count = kwargs.get("input_feature_count")
            self.layers = self.buildLayers(architecture=architecture)
        else:
            self.layers = self.loadLayers(model_params=kwargs.get("model_params"))

        self.whh_nonlinearity = self.layers[0].whh_nonlinearity
        self.why_nonlinearity = self.layers[-1].why_nonlinearity
        
        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.setOptimizer()





    def loadLayers(self, model_params):
        
        layers = [ 
            
        RNNCell( 
            pretrained=True, 
            device_type=self.device_type,
            type="hidden",
            pretrained_wxh=wxh,
            pretrained_whh=whh,
            pretrained_bh=bh,
            hidden_activation_function=hidden_activation,
            index=index ) for (wxh, whh, bh, hidden_activation, index) in list(model_params.values())[:-1] ] + [ 
                
        RNNCell(
            pretrained=True, 
            device_type=self.device_type,
            type="output",
            pretrained_wxh=wxh,
            pretrained_whh=whh,
            pretrained_bh=bh,
            hidden_activation_function=hidden_activation,
            pretrained_why=why,
            pretrained_by=by,
            output_activation_function=output_activation,
            index=index ) for (wxh, whh, bh, hidden_activation, why, by, output_activation, index) in [list(model_params.values())[-1]] ]
        
        return layers



    def buildLayers(self, architecture):
        hidden_state_neuron_counts = architecture.get("hidden_state_neuron_counts")
        hidden_activation_function = architecture.get("hidden_activation_function")
        output_activation_function = architecture.get("output_activation_function")
        output_feature_count = architecture.get("output_feature_count")
        num_layers = len(hidden_state_neuron_counts)

        layers = [ 
        
        RNNCell( 
            pretrained=False, 
            device_type=self.device_type,
            type="hidden",
            wxh_input_count=self.input_feature_count if i == 0 else hidden_state_neuron_counts[i-1],
            wxh_neuron_count=hidden_state_neuron_counts[i],
            whh_input_count=hidden_state_neuron_counts[i],
            whh_neuron_count=hidden_state_neuron_counts[i],
            hidden_activation_function=hidden_activation_function,
            index=i+1 ) for i in range(num_layers-1) ] + [ 
                
        RNNCell(
            pretrained=False, 
            device_type=self.device_type,
            type="output",
            wxh_input_count=self.input_feature_count if num_layers == 1 else hidden_state_neuron_counts[-2],
            wxh_neuron_count=hidden_state_neuron_counts[-1],
            whh_input_count=hidden_state_neuron_counts[-1],
            whh_neuron_count=hidden_state_neuron_counts[-1],
            hidden_activation_function=hidden_activation_function,
            why_input_count=hidden_state_neuron_counts[-1],
            why_neuron_count=output_feature_count,
            output_activation_function=output_activation_function,
            index=num_layers ) ]
        
        return layers
    

    def saveParameters(self):
        os.makedirs(f"{self.save_fpath}", exist_ok=True)
        for layer in self.layers:
            layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
            torch.save(layer.wxh, f"{self.save_fpath}/layer_{layer.index}_wxh.pth")
            torch.save(layer.whh, f"{self.save_fpath}/layer_{layer.index}_whh_{layer.whh_nonlinearity}.pth")
            torch.save(layer.bh, f"{self.save_fpath}/layer_{layer.index}_bh.pth")
            if layer.type == "output":
                torch.save(layer.why, f"{self.save_fpath}/layer_{layer.index}_why_{layer.why_nonlinearity}.pth")
                torch.save(layer.by, f"{self.save_fpath}/layer_{layer.index}_by.pth")


 



    def resetHiddenState(self, batch_size):
        return [torch.zeros(
            size=(batch_size, layer.wxh_neuron_count), 
            dtype=torch.float32, 
            device=self.device_type) for layer in self.layers]



    def forwardNonAutoRegressive(self, X, training):
        """
        No autoregressive handling - working
        """
    
        T = X.shape[1] 
        batch_size = X.shape[0]

        if self.stateful and not self.stateful_initialized: 
            for layer in self.layers: 
                layer.generateState(batch_size)
            self.stateful_initialized = True

        ht1_l = [layer.ht1 for layer in self.layers] if self.stateful else self.resetHiddenState(batch_size)
        whh_l = [layer.whh for layer in self.layers]
        wxh_l = [layer.wxh for layer in self.layers]
        bh_l  = [layer.bh  for layer in self.layers]
        why = self.layers[-1].why
        by = self.layers[-1].by


        output_feature_count = by.shape[-1]
        Y = torch.zeros(batch_size, T, output_feature_count, device=self.device_type)


        for t in range(T):

            x = X[:, t, :]

            ht_l = [None] * self.num_layers
            ht_l[0] = activate(
                ht1_l[0] @ whh_l[0] + bh_l[0] + 
                x @ wxh_l[0], self.whh_nonlinearity)

            for i in range(1, self.num_layers):
                ht_l[i] = activate(
                    ht1_l[i] @ whh_l[i] + bh_l[i] +
                    ht_l[i-1] @ wxh_l[i], self.whh_nonlinearity)

            Y[:, t, :] = activate( 
                ht_l[-1] @ why + by, self.why_nonlinearity)
            
            ht1_l = ht_l
        
        # time.sleep(1)
        if self.stateful:
            for (layer, ht1_i) in zip(self.layers, ht1_l):
                layer.ht1 = ht1_i.detach()

        # print(ht_l[0])
        return Y
 

    def forwardAutoRegressive(self, X, training):
        
        T = X.shape[1] 
        batch_size = X.shape[0]

        if self.stateful and not self.stateful_initialized: 
            for layer in self.layers: 
                layer.generateState(batch_size)
            self.stateful_initialized = True

        ht1_l = [layer.ht1 for layer in self.layers] if self.stateful else self.resetHiddenState(batch_size)
        whh_l = [layer.whh for layer in self.layers]
        wxh_l = [layer.wxh for layer in self.layers]
        bh_l  = [layer.bh  for layer in self.layers]
        why = self.layers[-1].why
        by = self.layers[-1].by

        output_feature_count = by.shape[0]
        Y = torch.zeros(batch_size, T, output_feature_count, device=self.device_type)

        x = X[:, 0, :]
        # print(x.shape)
        # exit()
        for t in range(T):

            # ht_l = [None] * len(ht1_l)
            ht_l = [None] * self.num_layers
            ht_l[0] = activate(
                ht1_l[0] @ whh_l[0] + bh_l[0] + 
                x @ wxh_l[0], self.whh_nonlinearity)

            # for i in range(1, len(ht_l)):
            for i in range(1, self.num_layers):
                ht_l[i] = activate(
                    ht1_l[i] @ whh_l[i] + bh_l[i] +
                    ht_l[i-1] @ wxh_l[i], self.whh_nonlinearity)

            y = activate( 
                ht_l[-1] @ why + by, self.why_nonlinearity)
            Y[:, t, :] = y
            # print(y.shape)

            ht1_l = ht_l

            if training:
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
            return self.forwardAutoRegressive(X, training)
        return self.forwardNonAutoRegressive(X, training)





    def backprop(self, loss):
        self.zerograd()

        loss.backward()

        if self.grad_clip_norm is not None or self.grad_clip_value is not None:
            self.clipGradients()

        with torch.no_grad():

            if not self.optimizer:
                self.update()
            else:
                self.t += 1
                self.optimizerUpdate()
