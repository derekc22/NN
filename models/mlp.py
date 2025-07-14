import torch
from src.network import Network
from src.dense import DenseLayer
import os


class MLP(Network):

    def __init__(self, pretrained, training, device, **kwargs):

        super().__init__(model_type="mlp", training=training, **kwargs)

        self.device = torch.device(device)
        self.save_fpath = kwargs.get("save_fpath")

        if not pretrained:
            self.layers = self.build_layers(kwargs.get("architecture"))
        else:
            self.layers = self.load_layers(kwargs.get("params"))

        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.set_optimizer()




    def load_layers(self, params):
        layers = [
            DenseLayer(
            pretrained=True, 
            device=self.device, 
            pretrained_weights=weights, 
            pretrained_biases=biases, 
            activation=activation, 
            index=index) for (weights, biases, activation, index) in params.values()
        ]

        return layers


    def build_layers(self, architecture):

        neuron_counts = architecture.get("neuron_counts")
        activations = architecture.get("activations")
        input_feature_count = architecture.get("input_feature_count")

        neuron_counts = [input_feature_count] + neuron_counts
        num_layers = len(neuron_counts)-1

        layers = [
            DenseLayer(
            pretrained=False, 
            device=self.device, 
            input_count=neuron_counts[i],
            neuron_count=neuron_counts[i+1], 
            activation=activations[i], 
            index=i+1) for i in range(num_layers)
        ]

        return layers


    def save_parameters(self, qualifier=""):
        save_fpath = f"{self.save_fpath}/{qualifier}"
        os.makedirs(save_fpath, exist_ok=True)
        for layer in self.layers:
            layer.index = str(layer.index).zfill(2)
            torch.save(layer.weights, f"{save_fpath}/layer_{layer.index}_weights_{layer.activation}.pth")
            torch.save(layer.biases, f"{save_fpath}/layer_{layer.index}_biases_{layer.activation}.pth")




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