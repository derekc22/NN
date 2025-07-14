import os
from src.network import Network
from models.mlp import MLP
from src.layer import CNNLayer
from typing_extensions import override
import torch


class CNN(Network):

    def __init__(self, pretrained, training, device, **kwargs):

        super().__init__(model_type="cnn", training=training, **kwargs)

        self.device = torch.device(device)
        self.save_fpath = kwargs.get("save_fpath")

        specifications = kwargs.get("specifications")

        if not pretrained:
            self.layers = self.build_layers(kwargs.get("architecture"))
        else:
            self.layers = self.load_layers(kwargs.get("params"))
        self.MLP = self.init_mlp(pretrained, training, **kwargs)

        if not (self.layers and self.MLP.layers):
            raise ValueError("CNN or MLP layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.set_optimizer()




    def load_layers(self, params):

        layers = [CNNLayer(
            pretrained=True, 
            device=self.device, 
            type=layer_type, 
            pretrained_kernels=kernels, 
            pretrained_biases=biases, 
            activation=activation,
            kernel_stride=stride, 
            index=index) for (layer_type, kernels, biases, activation, stride, index) in params.values()
        ]

        return layers


    def init_mlp(self, pretrained, training, **kwargs):
        if not pretrained:
            return MLP (
                pretrained=False, 
                training=training,
                device=self.device,
                architecture=self.calc_mlp_input_shape(
                    kwargs.get("architecture"), 
                    kwargs.get("mlp_architecture")),
                hyperparameters=kwargs.get("mlp_hyperparameters"), 
                save_fpath=kwargs.get("mlp_save_fpath"),
                specifications=kwargs.get("specifications")
                )
        return MLP(
                pretrained=True, 
                training=training, 
                device=self.device,
                params=kwargs.get("mlp_params"), 
                hyperparameters=kwargs.get("mlp_hyperparameters"), 
                save_fpath=kwargs.get("mlp_save_fpath"),
                specifications=kwargs.get("specifications")
            )
        
        


    def build_layers(self, architecture):
        layer_types = architecture.get("type")
        filter_counts = architecture.get("filter_counts")
        kernel_shapes = architecture.get("kernel_shapes")
        kernel_strides = architecture.get("kernel_strides")
        activation_fns = architecture.get("activation_fns")
        num_layers = len(layer_types)

        layers = [
            CNNLayer(
            pretrained=False, 
            device=self.device, 
            type=layer_types[i], 
            filter_count=filter_counts[i],
            kernel_shape=kernel_shapes[i], 
            kernel_stride=kernel_strides[i], 
            activation=activation_fns[i], 
            index=i+1) for i in range(num_layers)
        ]

        return layers



    def save_parameters(self, qualifier=""):
        save_fpath = f"{self.save_fpath}/{qualifier}"
        os.makedirs(save_fpath, exist_ok=True)
        for layer in self.layers:
            layer.index = str(layer.index).zfill(2)
            torch.save(layer.kernels, f"{save_fpath}/layer_{layer.index}_kernels_{layer.activation}_{layer.type}_{layer.kernel_stride}.pth")
            torch.save(layer.biases, f"{save_fpath}/layer_{layer.index}_biases_{layer.activation}_{layer.type}_{layer.kernel_stride}.pth")

        self.MLP.save_parameters()





            

    def calc_mlp_input_shape(self, architecture, mlp_architecture):
        
        print("calculating MLP input shape...")
        input_data_dim = (1, ) + tuple(architecture.get("input_data_dim"))
        dummy_data = torch.empty(size=input_data_dim, device=self.device)
        dummy_MLP_input = self.forward(dummy_data, training=True, dummy=True)
        dummy_mlp_input_feature_count = dummy_MLP_input.size(dim=1)
        
        mlp_architecture["input_feature_count"] = dummy_mlp_input_feature_count
        
        return mlp_architecture






    def forward(self, curr_input, training, dummy=False): 

        for layer in self.layers:
            if layer.type == "convolutional":

                curr_input = layer.convolve(curr_input)

                # if self.dropout_rate:
                #     curr_input = self.spatial_dropout(curr_input)

                # if training and not dummy and self.dropout_rate: #layer != self.layers[-1]:
                #     curr_input = layer.convolve(curr_input, dropout_rate=self.dropout_rate)
                # else:
                #     curr_input = layer.convolve(curr_input)

            else:
                curr_input = layer.maxpool(curr_input)


        curr_input_batch_size = curr_input.size(dim=0)
        # flattened_feature_map = curr_input.view(curr_input_batch_size, -1).to(torch.float32).T
        flattened_feature_map = curr_input.view(curr_input_batch_size, -1).to(torch.float32)
        # print(flattened_feature_map.shape)


        if dummy:
            return flattened_feature_map

        return self.MLP.forward(flattened_feature_map, training=training)




    # def spatial_dropout(self, curr_input):

    #   drop_count = int(self.dropout_rate * curr_input.size(dim=1))

    #   dropout_indices = torch.randint(low=0, high=curr_input.size(dim=1), size=(drop_count, ))

    #   curr_input[:, dropout_indices, :, :] = 0

    #   # print(f"curr_input_dim = {curr_input.size()}. dim 1 = {curr_input.size(dim=1)}")
    #   # print(dropout_indices)
    #   # print(curr_input)

    #   return curr_input






    @override
    def backprop(self, loss):

        self.zerograd()
        self.MLP.zerograd()


        loss.backward()

        with torch.no_grad():

            if not self.optimizer:
                self.update()
            else:
                self.t += 1
                self.optimizer_update()


            if not self.MLP.optimizer:
                self.MLP.update()
            else:
                self.MLP.t += 1
                self.MLP.optimizer_update()
