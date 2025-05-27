import torch
from src.network import Network
from rnn_cell import RecurrentCell


class RNN(Network):
    
    def __init__(self, pretrained, device_type, training, **kwargs):

        super().__init__(model_type="rnn", training=training, kwargs=kwargs)

        self.device_type = device_type

        if not pretrained:
            architecture = kwargs.get("architecture")
            self.layers = self.buildLayers(architecture=architecture, input_feature_count=kwargs.get("input_feature_count"))
        # else:
            # self.layers = self.loadLayers(rnn_model_params=kwargs.get("rnn_model_params"))

        # self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.setOptimizer()

        if not self.layers:
            raise ValueError("Layers is uninitialized!")




    # def loadLayers(self, model_params):
    #     layers = [RecurrentCell(
    #     pretrained=True, 
    #     device_type=self.device_type, 
    #     pretrained_weights=weights, 
    #     pretrained_biases=biases, 
    #     nonlinearity=nonlinearity, 
    #     index=index) for (weights, biases, nonlinearity, index) in model_params.values()]
        
    #     return layers


    def buildLayers(self, architecture, input_feature_count):
        # print("INPUT FEATURE COUNT: ", input_feature_count)
        hidden_state_neuron_counts = architecture.get("hidden_state_neuron_counts")
        hidden_activation_function = architecture.get("hidden_activation_function")
        output_activation_function = architecture.get("output_activation_function")
        output_feature_count = architecture.get("output_feature_count")
        time_steps = architecture.get("time_steps")
        

        layers = [RecurrentCell(
        pretrained=False, 
        device_type=self.device_type,
        time_steps=time_steps,
        input_count=input_feature_count,
        hidden_state_neuron_counts=hidden_state_neuron_counts,
        output_feature_count=output_feature_count,
        hidden_activation_function=hidden_activation_function,
        output_activation_function=output_activation_function,
        )]
        # index=1)]


        return layers
    

    def saveParameters(self):
        layer = self.layers[0]
        layer.index = "01" #"0" + str(layers.index) if layers.index < 10 else layers.index
        
        torch.save(layer.why, f"./params/paramsRNN/layer_why_{layer.why_nonlinearity}.pth")
        torch.save(layer.by, "./params/paramsRNN/layer_by.pth")
        
        for i, (wxh_l, whh_li, bh_li) in enumerate(zip(layer.wxh_l, layer.whh_l, layer.bh_l)):
            torch.save(wxh_l, f"./params/paramsRNN/layer_{i}_wxh.pth")
            torch.save(whh_li, f"./params/paramsRNN/layer_{i}_whh_{layer.whh_nonlinearity}.pth")
            torch.save(bh_li, f"./params/paramsRNN/layer_{i}_bh.pth")




    def forward(self, curr_input, training):
        layer = self.layers[0]
        out = layer.feed(curr_input)
        # if training and self.dropout_rate:
        #     curr_input = self.dropout(curr_input)
        return out




    def backprop(self, loss):
        self.zerograd()

        loss.backward()

        with torch.no_grad():

            if not self.optimizer:
                self.update()
            # else:
            #     self.t += 1
            #     self.optimizerUpdate()
