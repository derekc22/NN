import torch
from src.network import Network
from src.dense import DenseLayer
import os


class MLP(Network):

    def __init__(self, pretrained, training, device_type, **kwargs):

        super().__init__(model_type="mlp", training=training, **kwargs)

        self.device_type = torch.device(device_type)
        self.save_fpath = kwargs.get("save_fpath")

        if not pretrained:
            architecture = kwargs.get("architecture")
            self.input_feature_count = kwargs.get("input_feature_count")
            self.check_config(architecture=architecture)
            self.layers = self.build_layers(architecture=architecture) #or mlp_architecture.get("input_data_dim"))
        else:
            self.layers = self.load_layers(model_params=kwargs.get("model_params"))

        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.set_optimizer()




    def load_layers(self, model_params):
        layers = [
            DenseLayer(
            pretrained=True, 
            device_type=self.device_type, 
            pretrained_weights=weights, 
            pretrained_biases=biases, 
            nonlinearity=nonlinearity, 
            index=index) for (weights, biases, nonlinearity, index) in model_params.values()
        ]

        return layers


    def build_layers(self, architecture):

        neuron_counts = architecture.get("neuron_counts")
        activation_fns = architecture.get("activation_fns")
        neuron_counts = [self.input_feature_count] + neuron_counts
        num_layers = len(neuron_counts)-1

        layers = [
            DenseLayer(
            pretrained=False, 
            device_type=self.device_type, 
            input_count=neuron_counts[i],
            neuron_count=neuron_counts[i+1], 
            nonlinearity=activation_fns[i], 
            index=i+1) for i in range(num_layers)
        ]

        return layers


    def save_parameters(self):
        os.makedirs(f"{self.save_fpath}", exist_ok=True)
        for layer in self.layers:
            #layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
            layer.index = str(layer.index).zfill(2)
            torch.save(layer.weights, f"{self.save_fpath}/layer_{layer.index}_weights_{layer.nonlinearity}.pth")
            torch.save(layer.biases, f"{self.save_fpath}/layer_{layer.index}_biases_{layer.nonlinearity}.pth")




    def forward(self, curr_input, training):
        for layer in self.layers:

            curr_input = layer.feed(curr_input)

            if training and self.dropout_rate and layer != self.layers[-1]:
                curr_input = self.dropout(curr_input)

        return curr_input.squeeze()




    def dropout(self, curr_input):
        # print("FC DROPOUT")

        drop_count = int(self.dropout_rate * curr_input.numel())
        dropout_row_indices = torch.randint(low=0, high=curr_input.size(dim=0), size=(drop_count,))
        dropout_col_indices = torch.randint(low=0, high=curr_input.size(dim=1), size=(drop_count,))

        curr_input[dropout_row_indices, dropout_col_indices] = 0

        return curr_input




    # def backprop(self, loss):

    #     self.zerograd()

    #     loss.backward()

    #     with torch.no_grad():

    #         if not self.optimizer:
    #             self.update()
    #         else:
    #             self.t += 1
    #             self.optimizer_update()