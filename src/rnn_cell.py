import torch
import numpy as np


torch.manual_seed(42)

class RNNCell():

    def __init__(self, pretrained, device_type, **kwargs):

        # super().__init__(pretrained, device_type, **kwargs)

        self.index = int(kwargs.get("index"))
        self.device_type = device_type
        self.whh_nonlinearity = kwargs.get("hidden_nonlinearity")
        self.why_nonlinearity = kwargs.get("output_nonlinearity") 
        self.type = kwargs.get("type")
        # self.batch_size = kwargs.get("batch_size")
        
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


            stddev_wxh = np.sqrt(2 / (wxh_input_count + wxh_neuron_count))
            self.wxh = torch.normal(
                0, stddev_wxh, 
                size=(wxh_input_count, wxh_neuron_count), 
                dtype=torch.float32, 
                device=self.device_type)  # Xavier Initialization

            stddev_whh = np.sqrt(2 / (whh_input_count + whh_neuron_count))
            self.whh = torch.normal(
                0, stddev_whh,
                size=(whh_input_count, whh_neuron_count), 
                dtype=torch.float32, 
                device=self.device_type)  # Xavier Initialization

            self.bh = torch.zeros(
                1, whh_neuron_count, 
                dtype=torch.float32, 
                device=self.device_type)

            if self.type == "output":
                why_input_count = kwargs.get("why_input_count")
                why_neuron_count = kwargs.get("why_neuron_count")
                stddev_why = np.sqrt(2 / (why_input_count + why_neuron_count))
                self.why = torch.normal(
                    0, stddev_why, 
                    size=(why_input_count, why_neuron_count), 
                    dtype=torch.float32, 
                    device=self.device_type)  # Xavier Initialization
            
                self.by = torch.zeros(1, why_neuron_count, dtype=torch.float32, device=self.device_type)
            
        else:
            self.wxh = kwargs.get("pretrained_wxh").to(device=self.device_type)
            wxh_neuron_count = self.wxh.shape[-1]
            self.whh = kwargs.get("pretrained_whh").to(device=self.device_type)
            self.bh = kwargs.get("pretrained_bh").to(device=self.device_type)
            if self.type == "output":
                self.why = kwargs.get("pretrained_why").to(device=self.device_type)
                self.by = kwargs.get("pretrained_by").to(device=self.device_type)
            


        self.wxh_neuron_count = wxh_neuron_count


        self.wxh.requires_grad_()
        self.whh.requires_grad_()
        self.bh.requires_grad_()
        if self.type == "output":
            self.why.requires_grad_()
            self.by.requires_grad_()




    def generate_state(self, batch_size):
        self.ht1 = torch.zeros(
        batch_size, self.wxh_neuron_count, 
        dtype=torch.float32, 
        device=self.device_type)




    # def __repr__(self):
    #     return (f"__________________________________________\n"
    #             f"RNN Cell {self.index}\nwxh:\n{self.wxh.shape}\nwhh:{self.whh.shape}\\bh:\n{self.bh.shape}\nwhh Activation:\n{self.whh_nonlinearity}\n"
    #             f"__________________________________________")


