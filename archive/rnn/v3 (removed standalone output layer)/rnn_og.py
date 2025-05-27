import torch
from src.network_og import Network
from src.rnn_cell_og import RNNCell
from src.layer import Layer
import os


class RNN(Network):
    
    def __init__(self, pretrained, device_type, training, **kwargs):

        super().__init__(model_type="rnn", training=training, kwargs=kwargs)

        self.device_type = torch.device(device_type)
        self.stateful = kwargs.get("stateful", False)
        self.auto_regressive = kwargs.get("auto_regressive", False)
        self.teacher_forcing = kwargs.get("teacher_forcing")
        # self.batch_size = kwargs.get("batch_size")

        if not pretrained:
            architecture = kwargs.get("architecture")
            self.input_feature_count = kwargs.get("input_feature_count")
            self.layers = self.buildLayers(architecture=architecture)
            self.save_fpath = kwargs.get("save_fpath")
        else:
            self.layers = self.loadLayers(model_params=kwargs.get("model_params"))

        self.whh_nonlinearity = self.layers[0].whh_nonlinearity
        self.why_nonlinearity = self.layers[-1].why_nonlinearity
        

        if not self.layers:
            raise ValueError("Layers is uninitialized!")
        self.num_layers = len(self.layers)
        self.hidden_layers = self.layers[:-1]
        self.L = len(self.hidden_layers)

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
            hidden_activation_function=activation,
            stateful=self.stateful,
            # batch_size=self.batch_size,
            index=index ) for (wxh, whh, bh, activation, index) in list(model_params.values())[:-1] ] + [ 
                
        RNNCell(
            pretrained=True, 
            device_type=self.device_type,
            type="output",
            pretrained_wxh=wxh,
            pretrained_by=by,
            output_activation_function=activation,
            stateful=self.stateful,
            # batch_size=self.batch_size,
            index=index ) for (wxh, by, activation, index) in [list(model_params.values())[-1]] ]
        
        return layers



    def buildLayers(self, architecture):
        # print("INPUT FEATURE COUNT: ", input_feature_count)
        hidden_state_neuron_counts = architecture.get("hidden_state_neuron_counts")
        hidden_activation_function = architecture.get("hidden_activation_function")
        output_activation_function = architecture.get("output_activation_function")
        output_feature_count = architecture.get("output_feature_count")
        # time_steps = architecture.get("time_steps")
        
        hidden_depth = len(hidden_state_neuron_counts)

        layers = [ 
        
        RNNCell( 
            pretrained=False, 
            device_type=self.device_type,
            type="hidden",
            wxh_input_count=self.input_feature_count if i == 0 else hidden_state_neuron_counts[i-1] ,
            wxh_neuron_count=hidden_state_neuron_counts[i],
            whh_input_count=hidden_state_neuron_counts[i],
            whh_neuron_count=hidden_state_neuron_counts[i],
            hidden_activation_function=hidden_activation_function,
            stateful=self.stateful,
            # batch_size=self.batch_size,
            index=i+1 ) for i in range(hidden_depth) ] + [ 
                
        RNNCell(
            pretrained=False, 
            device_type=self.device_type,
            type="output",
            wxh_input_count=hidden_state_neuron_counts[-1],
            wxh_neuron_count=output_feature_count,
            output_activation_function=output_activation_function,
            stateful=self.stateful,
            # batch_size=self.batch_size,
            index=hidden_depth+1 )]

        return layers
    

    def saveParameters(self):
        os.makedirs(f"{self.save_fpath}", exist_ok=True)
        for layer in self.layers:
            layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
            if layer.type == "hidden":
                torch.save(layer.wxh, f"{self.save_fpath}/layer_{layer.index}_wxh.pth")
                torch.save(layer.whh, f"{self.save_fpath}/layer_{layer.index}_whh_{layer.whh_nonlinearity}.pth")
                torch.save(layer.bh, f"{self.save_fpath}/layer_{layer.index}_bh.pth")
            elif layer.type == "output":
                torch.save(layer.wxh, f"{self.save_fpath}/layer_{layer.index}_wxh_{layer.why_nonlinearity}.pth")
                torch.save(layer.by, f"{self.save_fpath}/layer_{layer.index}_by.pth")


 



    def zeroState(self, batch_size):
        return [torch.zeros(
            size=(batch_size, layer.wxh_neuron_count), 
            dtype=torch.float32, 
            device=self.device_type) for layer in self.hidden_layers]

    # def forward_stateless(self, X, training):

    #     ht1_l = self.zeroState()
        
    #     whh_l = [layer.whh for layer in self.hidden_layers]
    #     wxh_l = [layer.wxh for layer in self.layers]
    #     bh_l  = [layer.bh  for layer in self.hidden_layers]
    #     why = self.layers[-1].wxh
    #     by = self.layers[-1].by

    #     T = X.shape[1] 
    #     output_feature_count = by.shape[0]
    #     Y = torch.zeros(T, output_feature_count, device=X.device)

    #     for t in range(T):
    #         x = X[t]
    #         print(x.shape)
    #         exit()

    #         ht_l = [None] * len(ht1_l)
    #         ht_l[0] = Layer.staticActivate(
    #             torch.matmul(ht1_l[0], whh_l[0]) + bh_l[0] + 
    #             torch.matmul(x, wxh_l[0]), self.whh_nonlinearity)

    #         for i in range(1, len(ht_l)):
    #             ht_l[i] = Layer.staticActivate(
    #                 torch.matmul(ht1_l[i], whh_l[i]) + bh_l[i] +
    #                 torch.matmul(ht_l[i-1], wxh_l[i]), self.whh_nonlinearity)

    #         Y[t] = Layer.staticActivate( 
    #             torch.matmul(ht_l[-1], why) + by, self.why_nonlinearity)
            
    #         ht1_l = ht_l

    #     return Y



    # def forward_stateful(self, X, training):

    #     ht1_l = [layer.ht1 for layer in self.hidden_layers]
    #     whh_l = [layer.whh for layer in self.hidden_layers]
    #     wxh_l = [layer.wxh for layer in self.layers]
    #     bh_l  = [layer.bh  for layer in self.hidden_layers]
    #     why = self.layers[-1].wxh
    #     by = self.layers[-1].by

    #     T = X.shape[1] 
    #     output_feature_count = by.shape[0]
    #     Y = torch.zeros(T, output_feature_count, device=X.device)

    #     for t in range(T):
    #         x = X[t]

    #         ht_l = [None] * len(ht1_l)
    #         ht_l[0] = Layer.staticActivate(
    #             torch.matmul(ht1_l[0], whh_l[0]) + bh_l[0] + 
    #             torch.matmul(x, wxh_l[0]), self.whh_nonlinearity)

    #         for i in range(1, len(ht_l)):
    #             ht_l[i] = Layer.staticActivate(
    #                 torch.matmul(ht1_l[i], whh_l[i]) + bh_l[i] +
    #                 torch.matmul(ht_l[i-1], wxh_l[i]), self.whh_nonlinearity)

    #         Y[t] = Layer.staticActivate( 
    #             torch.matmul(ht_l[-1], why) + by, self.why_nonlinearity)
            
    #         ht1_l = ht_l

    #     for layer, ht1_i in zip(self.hidden_layers, ht1_l):
    #         layer.ht1 = ht1_i.detach()


    #     return Y


    # def forward(self, X, training):
    #     if self.stateful:
    #         return self.forward_stateful(X, training)
    #     return self.forward_stateless(X, training)



    def forwardNonAutoRegressive(self, X, training):
        """
        No autoregressive handling - working
        """
        
        T = X.shape[1] 
        batch_size = X.shape[0]

        if self.stateful: 
            for layer in self.hidden_layers: 
                layer.generate_state(batch_size)

        ht1_l = [layer.ht1 for layer in self.hidden_layers] if self.stateful else self.zeroState(batch_size)
        whh_l = [layer.whh for layer in self.hidden_layers]
        wxh_l = [layer.wxh for layer in self.layers]
        bh_l  = [layer.bh  for layer in self.hidden_layers]
        why = self.layers[-1].wxh
        by = self.layers[-1].by


        output_feature_count = by.shape[0]
        Y = torch.zeros(batch_size, T, output_feature_count, device=X.device)

        for t in range(T):
            x = X[:, t, :]

            # ht_l = [None] * len(ht1_l)
            ht_l = [None] * self.L
            ht_l[0] = Layer.staticActivate(
                torch.matmul(ht1_l[0], whh_l[0]) + bh_l[0] + 
                torch.matmul(x, wxh_l[0]), self.whh_nonlinearity)

            # for i in range(1, len(ht_l)):
            for i in range(1, self.L):
                ht_l[i] = Layer.staticActivate(
                    torch.matmul(ht1_l[i], whh_l[i]) + bh_l[i] +
                    torch.matmul(ht_l[i-1], wxh_l[i]), self.whh_nonlinearity)

            Y[:, t, :] = Layer.staticActivate( 
                torch.matmul(ht_l[-1], why) + by, self.why_nonlinearity)
            
            ht1_l = ht_l

        if self.stateful:
            for (layer, ht1_i) in zip(self.hidden_layers, ht1_l):
                layer.ht1 = ht1_i.detach()
        
        # print(ht_l[0])
        return Y
 

    def forwardAutoRegressive(self, X, training):
        
        T = X.shape[1] 
        batch_size = X.shape[0]

        if self.stateful: 
            for layer in self.hidden_layers: 
                layer.generate_state(batch_size)

        ht1_l = [layer.ht1 for layer in self.hidden_layers] if self.stateful else self.zeroState(batch_size)
        whh_l = [layer.whh for layer in self.hidden_layers]
        wxh_l = [layer.wxh for layer in self.layers]
        bh_l  = [layer.bh  for layer in self.hidden_layers]
        why = self.layers[-1].wxh
        by = self.layers[-1].by

        output_feature_count = by.shape[0]
        Y = torch.zeros(batch_size, T, output_feature_count, device=X.device)

        x = X[:, 0, :]
        # print(x.shape)
        # exit()
        for t in range(T):

            # ht_l = [None] * len(ht1_l)
            ht_l = [None] * self.L
            ht_l[0] = Layer.staticActivate(
                torch.matmul(ht1_l[0], whh_l[0]) + bh_l[0] + 
                torch.matmul(x, wxh_l[0]), self.whh_nonlinearity)

            # for i in range(1, len(ht_l)):
            for i in range(1, self.L):
                ht_l[i] = Layer.staticActivate(
                    torch.matmul(ht1_l[i], whh_l[i]) + bh_l[i] +
                    torch.matmul(ht_l[i-1], wxh_l[i]), self.whh_nonlinearity)

            y = Layer.staticActivate( 
                torch.matmul(ht_l[-1], why) + by, self.why_nonlinearity)
            Y[:, t, :] = y
            # print(y.shape)

            ht1_l = ht_l

            teacher_forcing_factor = (
                self.epoch/self.epochs if (self.teacher_forcing == "progressive") # progressive teacher-forcing
                else 1 - self.epoch/self.epochs if (self.teacher_forcing == "regressive") # regressive teacher-forcing
                else float(self.teacher_forcing) if isinstance(self.teacher_forcing, (int, float)) 
                else 0
            )


            x = X[:, t, :] if (training and t < teacher_forcing_factor*T) else y.detach()

            # print(x.shape)
            # exit()

        if self.stateful:
            for (layer, ht1_i) in zip(self.hidden_layers, ht1_l):
                layer.ht1 = ht1_i.detach()

        return Y


    def forward(self, X, training):
        # if this is uncommented, it will force the pure teacher forcing RNN implementations to run inference auto regressively (which i believe is the correct way to do it)
        # however, since pure auto regressive inference currently sucks (both for the auto regressive and teacher forced implementations), i will leave it commented out such that I can at least see good results at inference for the teacher forced implementations
        # but yeah i think this would be the more correct way to do it (i.e. to always run inference auto regressively)
        # if self.auto_regressive or not training: 
        if self.auto_regressive:
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
