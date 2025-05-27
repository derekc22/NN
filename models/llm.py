import torch
from src.network import Network
from rnn_cell import RNNCell
from src.layer import Layer
import os


class RNN(Network):
    
    def __init__(self, pretrained, device_type, training, **kwargs):

        super().__init__(model_type="rnn", training=training, kwargs=kwargs)

        self.device_type = torch.device(device_type)
        # self.t_steps = kwargs.get("time_steps")

        if not pretrained:
            architecture = kwargs.get("architecture")
            # self.why_neuron_count = architecture.get("output_feature_count")
            # self.whh_nonlinearity = architecture.get("hidden_activation_function")
            # self.why_nonlinearity = architecture.get("output_activation_function")

            self.layers = self.buildLayers(architecture=architecture, input_feature_count=kwargs.get("input_feature_count"))
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
            hidden_activation_function=activation,
            index=index ) for (wxh, whh, bh, activation, index) in list(model_params.values())[:-1] ] + [ 
                
        RNNCell(
            pretrained=True, 
            device_type=self.device_type,
            type="output",
            pretrained_wxh=wxh,
            pretrained_by=by,
            output_activation_function=activation,
            index=index ) for (wxh, by, activation, index) in [list(model_params.values())[-1]] ]
        
        return layers



    def buildLayers(self, architecture, input_feature_count):
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
            wxh_input_count=input_feature_count if i == 0 else hidden_state_neuron_counts[i-1] ,
            wxh_neuron_count=hidden_state_neuron_counts[i],
            whh_input_count=hidden_state_neuron_counts[i],
            whh_neuron_count=hidden_state_neuron_counts[i],
            hidden_activation_function=hidden_activation_function,
            index=i+1) for i in range(hidden_depth) ] + [ 
                
        RNNCell(
            pretrained=False, 
            device_type=self.device_type,
            type="output",
            wxh_input_count=hidden_state_neuron_counts[-1],
            wxh_neuron_count=output_feature_count,
            output_activation_function=output_activation_function,
            index=hidden_depth+1 )]

        return layers
    

    def saveParameters(self):
        os.makedirs('params/paramsRNN', exist_ok=True)
        for layer in self.layers:
            layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
            if layer.type == "hidden":
                torch.save(layer.wxh, f"./params/paramsRNN/layer_{layer.index}_wxh.pth")
                torch.save(layer.whh, f"./params/paramsRNN/layer_{layer.index}_whh_{layer.whh_nonlinearity}.pth")
                torch.save(layer.bh, f"./params/paramsRNN/layer_{layer.index}_bh.pth")
            elif layer.type == "output":
                torch.save(layer.wxh, f"./params/paramsRNN/layer_{layer.index}_wxh_{layer.why_nonlinearity}.pth")
                torch.save(layer.by, f"./params/paramsRNN/layer_{layer.index}_by.pth")


    #  def forward(self, X, training):

    #     ht_l = [layer.ht for layer in self.layers[:-1]]
    #     ht1_l = [layer.ht1 for layer in self.layers[:-1]]
    #     wxh_l = [layer.wxh for layer in self.layers]
    #     whh_l = [layer.whh for layer in self.layers[:-1]]
    #     bh_l = [layer.bh for layer in self.layers[:-1]]
    #     why = self.layers[-1].wxh
    #     by = self.layers[-1].by
        
    #     # t_steps = X.shape[0]
    #     # print(t_steps)
    #     output_feature_count = by.shape[0]

    #     def feed_in_l():
    #         for i in range(1, len(ht_l)):
    #             ht_l[i] = Layer.staticActivate( 
    #                 torch.matmul(ht1_l[i], whh_l[i]) + bh_l[i] + 
    #                 torch.matmul(ht_l[i-1], wxh_l[i]), self.whh_nonlinearity)
    #         yt = Layer.staticActivate(torch.matmul(ht_l[-1], why) + by, self.why_nonlinearity)
    #         return yt
        

    #     def feed(X):
    #         """feed in t"""
    #         nonlocal ht1_l
    #         Y = torch.zeros( size=(self.t_steps, output_feature_count) )
    #         for t in range(self.t_steps):
    #             x = X[t]
    #             ht_l[0] = Layer.staticActivate(
    #                 torch.matmul(ht1_l[0], whh_l[0]) + bh_l[0] + 
    #                 torch.matmul(x, wxh_l[0]), self.whh_nonlinearity)
    #             Y[t] = feed_in_l()
    #             # ht1_l = ht_l.copy()
    #             ht1_l[:] = ht_l

    #             for layer, ht1_i in zip(self.layers[:-1], ht1_l):
    #                 layer.ht1 = ht1_i

    #         return Y

    #     return feed(X)


    # def forward(self, X, training):

    #     ht1_l = [layer.ht1 for layer in self.layers[:-1]]        
    #     ht_l = [None]*len(self.layers[:-1])

    #     wxh_l = [layer.wxh for layer in self.layers]
    #     whh_l = [layer.whh for layer in self.layers[:-1]]
    #     bh_l  = [layer.bh  for layer in self.layers[:-1]]
    #     why = self.layers[-1].wxh
    #     by = self.layers[-1].by

    #     output_feature_count = by.shape[0]

    #     def feed_in_l():
    #         for i in range(1, len(ht_l)):
    #             ht_l[i] = Layer.staticActivate(
    #                 torch.matmul(ht1_l[i], whh_l[i]) + bh_l[i] + 
    #                 torch.matmul(ht_l[i-1], wxh_l[i]), self.whh_nonlinearity )
    #         yt = Layer.staticActivate(torch.matmul(ht_l[-1], why) + by, self.why_nonlinearity)
    #         return yt

    #     def feed(X):
    #         nonlocal ht1_l

    #         Y = torch.zeros(self.t_steps, output_feature_count, device=self.device)
    #         for t in range(self.t_steps):
    #             x = X[t]
    #             ht_l[0] = Layer.staticActivate(
    #                 torch.matmul(ht1_l[0], whh_l[0]) + bh_l[0] + 
    #                 torch.matmul(x, wxh_l[0]), self.whh_nonlinearity)
    #             Y[t] = feed_in_l()

    #             for layer, ht_i in zip(self.layers[:-1], ht_l):
    #                 layer.ht1 = ht_i.detach()
    #             ht1_l = [layer.ht1 for layer in self.layers[:-1]]

    #         return Y

    #     return feed(X)


    def forward(self, X, training):
        """
        X: Tensor of shape (T, input_dim)
        returns Y: Tensor of shape (T, output_feature_count)
        """
        # 1) Prepare per-layer parameters and activations
        #    (unchanged from your original code)
        ht1_l = [layer.ht1 for layer in self.layers[:-1]]                
        whh_l = [layer.whh for layer in self.layers[:-1]]
        wxh_l = [layer.wxh for layer in self.layers]
        bh_l  = [layer.bh  for layer in self.layers[:-1]]
        why   = self.layers[-1].wxh
        by    = self.layers[-1].by

        T = X.shape[0]  # actual sequence length
        output_feature_count = by.shape[0]
        Y = torch.zeros(T, output_feature_count, device=self.device)

        # 2) Loop over time-steps WITHOUT detaching hidden state
        for t in range(T):
            x = X[t]

            # First hidden layer
            ht_l = [None] * len(ht1_l)
            ht_l[0] = Layer.staticActivate(
                torch.matmul(ht1_l[0], whh_l[0]) + bh_l[0] + 
                torch.matmul(x, wxh_l[0]), self.whh_nonlinearity)

            # Remaining hidden layers
            for i in range(1, len(ht_l)):
                ht_l[i] = Layer.staticActivate(
                    torch.matmul(ht1_l[i], whh_l[i]) + bh_l[i] +
                    torch.matmul(ht_l[i-1], wxh_l[i]), self.whh_nonlinearity)

            # Output layer
            Y[t] = Layer.staticActivate( 
                torch.matmul(ht_l[-1], why) + by, self.why_nonlinearity)

            # Prepare for next time step—keep the full gradient graph alive
            ht1_l = ht_l

        # 3) After the entire sequence, detach only once
        #    so the next call to forward() starts fresh.
        for layer, h in zip(self.layers[:-1], ht1_l):
            layer.ht1 = h.detach()

        return Y




    def backprop(self, loss):
        self.zerograd()

        loss.backward()

        with torch.no_grad():

            if not self.optimizer:
                self.update()
            else:
                self.t += 1
                self.optimizerUpdate()
