import torch
import numpy as np
import torch.nn as nn
from typing import override
from src.dense import DenseLayer

# torch.manual_seed(42)

class RecurrentCell(DenseLayer):

    def __init__(self, pretrained, device_type, **kwargs):

        # super().__init__(pretrained, device_type, **kwargs)

        # self.index = int(kwargs.get("index"))

        self.device_type = device_type


        if not pretrained:
            input_count = kwargs.get("input_count")
            # self.batch_size = kwargs.get("batch_size")
            hidden_state_neuron_counts = kwargs.get("hidden_state_neuron_counts")
            # print(hidden_state_neuron_counts)
            # exit()
            wxh_neuron_counts = hidden_state_neuron_counts #kwargs.get("wxh_neuron_counts")
            whh_neuron_counts = hidden_state_neuron_counts #kwargs.get("whh_neuron_counts")

            self.whh_nonlinearity = kwargs.get("hidden_activation_function")
            self.why_nonlinearity = kwargs.get("output_activation_function")
            self.why_neuron_count = kwargs.get("output_feature_count")

            self.t_steps = kwargs.get("time_steps")
            self.network_depth = len(hidden_state_neuron_counts)

            # stddev = np.sqrt(2 / input_count)
            stddev = np.sqrt(2 / 3)
            
            # print((input_count, wxh_neuron_counts[0]))
            # exit()
            self.wxh_l0 = torch.normal(0, stddev, size=(input_count, wxh_neuron_counts[0]), dtype=torch.float32, device=self.device_type) # He Initialization
            # self.whh_l0 = torch.normal(0, stddev, size=(wxh_neuron_counts[0], whh_neuron_counts[0]), dtype=torch.float32, device=self.device_type) # He Initialization
            # self.bh_l0 = torch.zeros(whh_neuron_counts[0], dtype=torch.float32, device=self.device_type)
            # self.ht1_0 = torch.zeros(size=(batch_size, wxh_neuron_counts[0])) #None #self.ht_0.clone()
            

            self.ht1_l = [torch.zeros(
                wxh_neuron_count, 
                dtype=torch.float32, 
                device=self.device_type) for wxh_neuron_count in wxh_neuron_counts
            ]

            self.ht_l=[None]*self.network_depth
            
            """good"""
            self.wxh_l = [self.wxh_l0] + [torch.normal(
                0, stddev,
                size=(wxh_neuron_counts[i-1], wxh_neuron_counts[i]), 
                dtype=torch.float32, 
                device=self.device_type) for i in range(1, self.network_depth)
            ]

            """good"""
            self.whh_l = [torch.normal(
                0, stddev,
                size=(wxh_neuron_counts[i], whh_neuron_counts[i]), 
                dtype=torch.float32, 
                device=self.device_type) for i in range(self.network_depth)
            ]

            """good"""
            self.bh_l = [torch.zeros(
                whh_neuron_counts[i], 
                dtype=torch.float32, 
                device=self.device_type) for i in range(self.network_depth)
            ]
            
            # print((whh_neuron_counts[-1], self.why_neuron_count))
            # exit()
            self.why = torch.normal(0, stddev, size=(whh_neuron_counts[-1], self.why_neuron_count), dtype=torch.float32, device=self.device_type) # He Initialization
            self.by = torch.zeros(self.why_neuron_count, dtype=torch.float32, device=self.device_type)
            
        
        else:
            pass


        self.why.requires_grad_()
        self.by.requires_grad_()
        for wxh_l, whh_li, bh_li in zip(self.wxh_l, self.whh_l, self.bh_l):
            wxh_l.requires_grad_()
            whh_li.requires_grad_()
            bh_li.requires_grad_()



    def __repr__(self):
        pass
        # return (f"__________________________________________\n"
        #         f"MLP Layer {self.index}\nWeights:\n{self.weights}\nBiases:\n{self.biases}\nActivation:\n{self.nonlinearity}\n"
        #         f"__________________________________________")



    
    def feed_in_l(self):

        for i in range(1, len(self.ht_l)):
            
            self.ht_l[i] = self.activate( 
                torch.matmul(self.ht1_l[i], self.whh_l[i]) + self.bh_l[i] + 
                torch.matmul(self.ht_l[i-1], self.wxh_l[i]), self.whh_nonlinearity)
            
        yt = self.activate( torch.matmul(self.ht_l[-1], self.why) + self.by, self.why_nonlinearity)
        return yt
    

    @override
    def feed(self, X):
        """feed in t"""

        # t_steps = time_steps#X.shape[0]
        Y = torch.zeros( size=(self.t_steps, self.why_neuron_count) )
        for t in range(self.t_steps):

            x = X[t]
            # print("a: ", (torch.matmul(self.ht1_l[0], self.whh_l[0]) + self.bh_l[0]).shape)
            # print("b: ", torch.matmul(x, self.wxh_l[0]).shape)

            self.ht_l[0] = self.activate( 
                torch.matmul(self.ht1_l[0], self.whh_l[0]) + self.bh_l[0] + 
                torch.matmul(x, self.wxh_l[0]), self.whh_nonlinearity)

            Y[t] = self.feed_in_l()

            self.ht1_l = self.ht_l.copy()

            # for h in self.ht_l:
            #     print(h.shape)

        return Y



