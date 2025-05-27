import torch
import numpy as np
import torch.nn as nn


torch.manual_seed(42)

class RNNCell():

    def __init__(self, pretrained, device_type, **kwargs):

        # super().__init__(pretrained, device_type, **kwargs)

        self.index = int(kwargs.get("index"))
        self.device_type = device_type
        self.whh_nonlinearity = kwargs.get("hidden_activation_function")
        self.why_nonlinearity = kwargs.get("output_activation_function") 
        self.type = kwargs.get("type")
        self.stateful = kwargs.get("stateful")
        # self.batch_dim = kwargs.get("batch_dim")
        
        if not pretrained:

            wxh_input_count = kwargs.get("wxh_input_count")
            wxh_neuron_count = kwargs.get("wxh_neuron_count")
            whh_input_count = kwargs.get("whh_input_count")
            whh_neuron_count = kwargs.get("whh_neuron_count")

            """
                Note: whh_input_count = whh_neuron_count
                That is, the hidden state transformation matrices (whh) are square for all recurrent cells
                Thus, the hidden state does not change dimensions as the RNN steps in T
                That is, the dimensions of the hidden state are fixed
                This is an intentional design choice as it is unconventional to have the hidden state dimensions change shape
                What DOES change is the dimension of X (input) to O (output) as the RNN steps in L
                Again, this is an intentional design choice
            """

            # for the output cell, 'wxh' refers to the 'why' weight matrix
            # the attribute name 'wxh' is used for both hidden cells and output cells for consistency sake 
            # this avoids having to litter the code with extra if statements when attempting to access the weights matrix of an output cell (if self.type == "output" -> self.why)
            stddev_wxh = np.sqrt(2 / (wxh_input_count + wxh_neuron_count))
            self.wxh = torch.normal(
                0, stddev_wxh, 
                size=(wxh_input_count, wxh_neuron_count), 
                dtype=torch.float32, 
                device=self.device_type)  # Xavier Initialization

            if self.type == "hidden":
                stddev_whh = np.sqrt(2 / (whh_input_count + whh_neuron_count))
                self.whh = torch.normal(
                    0, stddev_whh,
                    size=(whh_input_count, whh_neuron_count), 
                    dtype=torch.float32, 
                    device=self.device_type)  # Xavier Initialization

                self.bh = torch.zeros(
                    whh_neuron_count, 
                    dtype=torch.float32, 
                    device=self.device_type)

            elif self.type == "output":
                self.by = torch.zeros(wxh_neuron_count, dtype=torch.float32, device=self.device_type)
            

        else:
            self.wxh = kwargs.get("pretrained_wxh").to(device=self.device_type)
            wxh_neuron_count = self.wxh.shape[-1]
            if self.type == "hidden":
                self.whh = kwargs.get("pretrained_whh").to(device=self.device_type)
                self.bh = kwargs.get("pretrained_bh").to(device=self.device_type)
            elif self.type == "output":
                self.by = kwargs.get("pretrained_by").to(device=self.device_type)
            

        # state is now directly handled by the RNN class (not the RNNCell class). disregard this. simply for historical documentation purposes
        # if self.type == "hidden":
        #     self.ht1 = torch.zeros(
        #         wxh_neuron_count, 
        #         dtype=torch.float32, 
        #         device=self.device_type) 
        #     # self.ht = 0

        self.wxh_neuron_count = wxh_neuron_count


        self.wxh.requires_grad_()
        if self.type == "hidden":
            self.whh.requires_grad_()
            self.bh.requires_grad_()
        elif self.type == "output":
            self.by.requires_grad_()





    def generateState(self, batch_dim):
        if self.stateful and self.type == "hidden":
            self.ht1 = torch.zeros(
            batch_dim, self.wxh_neuron_count, 
            dtype=torch.float32, 
            device=self.device_type)



    def __repr__(self):
        pass
        # return (f"__________________________________________\n"
        #         f"MLP Layer {self.index}\nWeights:\n{self.weights}\nBiases:\n{self.biases}\nActivation:\n{self.nonlinearity}\n"
        #         f"__________________________________________")


