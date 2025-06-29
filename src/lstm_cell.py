import torch
import numpy as np


# torch.manual_seed(42)

class LSTMCell():

    def __init__(self, pretrained, device_type, **kwargs):

        self.index = int(kwargs.get("index"))
        self.device_type = device_type
        self.gate_nonlinearity = kwargs.get("gate_nonlinearity")
        self.why_nonlinearity = kwargs.get("output_nonlinearity")
        self.type = kwargs.get("type")
        # self.stateful = kwargs.get("stateful")
        
        if not pretrained:

            ht1_input_count = kwargs.get("ht1_input_count")
            xt_input_count = kwargs.get("xt_input_count")
            gate_neuron_count = kwargs.get("gate_neuron_count")

            stddev_wgate = np.sqrt(2 / (ht1_input_count + xt_input_count + gate_neuron_count))

            self.wf = torch.normal(
                0, stddev_wgate, 
                size=(ht1_input_count + xt_input_count , gate_neuron_count), 
                dtype=torch.float32, 
                device=self.device_type)  # Xavier Initialization
            
            self.wi = torch.normal(
                0, stddev_wgate, 
                size=(ht1_input_count + xt_input_count , gate_neuron_count), 
                dtype=torch.float32, 
                device=self.device_type)  # Xavier Initialization
            
            self.wc = torch.normal(
                0, stddev_wgate, 
                size=(ht1_input_count + xt_input_count , gate_neuron_count), 
                dtype=torch.float32, 
                device=self.device_type)  # Xavier Initialization
            
            self.wo = torch.normal(
                0, stddev_wgate, 
                size=(ht1_input_count + xt_input_count , gate_neuron_count), 
                dtype=torch.float32, 
                device=self.device_type)  # Xavier Initialization
            
            self.bf = torch.zeros(gate_neuron_count, dtype=torch.float32, device=self.device_type)
            self.bi = torch.zeros(gate_neuron_count, dtype=torch.float32, device=self.device_type)
            self.bc = torch.zeros(gate_neuron_count, dtype=torch.float32, device=self.device_type)
            self.bo = torch.zeros(gate_neuron_count, dtype=torch.float32, device=self.device_type)
       

            if self.type == "output":
                why_neuron_count = kwargs.get("why_neuron_count")
                stddev_why = np.sqrt(2 / ( gate_neuron_count + why_neuron_count ))
                
                self.why = torch.normal(
                    0, stddev_why,
                    size=(gate_neuron_count, why_neuron_count), 
                    dtype=torch.float32, 
                    device=self.device_type)  # Xavier Initialization

                self.by = torch.zeros(1, why_neuron_count, dtype=torch.float32, device=self.device_type)


        else:
            self.wf = kwargs.get("pretrained_wf").to(device=self.device_type)
            gate_neuron_count = self.wf.shape[-1]
            self.wi = kwargs.get("pretrained_wi").to(device=self.device_type)
            self.wc = kwargs.get("pretrained_wc").to(device=self.device_type)
            self.wo = kwargs.get("pretrained_wo").to(device=self.device_type)
            
            self.bf = kwargs.get("pretrained_bf").to(device=self.device_type)
            self.bi = kwargs.get("pretrained_bi").to(device=self.device_type)
            self.bc = kwargs.get("pretrained_bc").to(device=self.device_type)
            self.bo = kwargs.get("pretrained_bo").to(device=self.device_type)

            if self.type == "output":
                self.why = kwargs.get("pretrained_why").to(device=self.device_type)
                self.by = kwargs.get("pretrained_by").to(device=self.device_type)

        self.gate_neuron_count = gate_neuron_count

        self.wf.requires_grad_()
        self.wi.requires_grad_()
        self.wc.requires_grad_()
        self.wo.requires_grad_()

        self.bf.requires_grad_()
        self.bi.requires_grad_()
        self.bc.requires_grad_()
        self.bo.requires_grad_()

        if self.type == "output":
            self.why.requires_grad_()
            self.by.requires_grad_()





    def generate_state(self, batch_size):
        self.ht1 = torch.zeros(
        batch_size, self.gate_neuron_count, 
        dtype=torch.float32, 
        device=self.device_type)
        
        self.Ct1 = torch.zeros(
        batch_size, self.gate_neuron_count, 
        dtype=torch.float32, 
        device=self.device_type)






    def __repr__(self):
        pass
        # return (f"__________________________________________\n"
        #         f"MLP Layer {self.index}\nWeights:\n{self.weights}\nBiases:\n{self.biases}\nActivation:\n{self.nonlinearity}\n"
        #         f"__________________________________________")




