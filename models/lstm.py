import torch
from src.network import Network
from src.lstm_cell import LSTMCell
from utils.functions import activate
import os


class LSTM(Network):
    
    def __init__(self, pretrained, device_type, training, **kwargs):

        super().__init__(model_type="lstm", training=training, kwargs=kwargs)

        self.device_type = torch.device(device_type)
        self.stateful = kwargs.get("stateful", False)
        self.save_fpath = kwargs.get("save_fpath")
        
        if self.stateful:
            self.state_initialized = False
        self.autoregressive = kwargs.get("autoregressive", False)
        self.autoregressive = kwargs.get("autoregressive", False)
        self.teacher_forcing = kwargs.get("teacher_forcing")

        if not pretrained:
            architecture = kwargs.get("architecture")
            self.input_feature_count = kwargs.get("input_feature_count")
            self.layers = self.buildLayers(architecture=architecture)
        else:
            self.layers = self.loadLayers(model_params=kwargs.get("model_params"))

        self.gate_nonlinearity = self.layers[0].gate_nonlinearity
        self.why_nonlinearity = self.layers[-1].why_nonlinearity
        

        if not self.layers:
            raise ValueError("Layers are uninitialized!")
        self.num_layers = len(self.layers)

        if training and self.optimizer:
            self.setOptimizer()





    def loadLayers(self, model_params):
        
        layers = [ 

            LSTMCell( 
            pretrained=True, 
            device_type=self.device_type,
            type="hidden",
            pretrained_wf=wf,
            pretrained_wi=wi,
            pretrained_wc=wc,
            pretrained_wo=wo,
            pretrained_bf=bf,
            pretrained_bi=bi,
            pretrained_bc=bc,
            pretrained_bo=bo,
            gate_nonlinearity=gate_activation_fn,
            index=index ) for (wf, wi, wc, wo, bf, bi, bc, bo, gate_activation_fn, index) in list(model_params.values())[:-1] ] + [ 
                
            LSTMCell(
            pretrained=True, 
            device_type=self.device_type,
            type="output",
            pretrained_wf=wf,
            pretrained_wi=wi,
            pretrained_wc=wc,
            pretrained_wo=wo,
            pretrained_bf=bf,
            pretrained_bi=bi,
            pretrained_bc=bc,
            pretrained_bo=bo,
            gate_nonlinearity=gate_activation_fn,
            pretrained_why=why,
            pretrained_by=by,
            output_nonlinearity=output_activation_fn,
            index=index ) for (wf, wi, wc, wo, bf, bi, bc, bo, gate_activation_fn, why, by, output_activation_fn, index) in [list(model_params.values())[-1]] ]
        
        return layers



    def buildLayers(self, architecture):

        gate_neuron_counts = architecture.get("gate_neuron_counts")
        gate_activation_fn = architecture.get("gate_activation_fn")
        output_activation_fn = architecture.get("output_activation_fn")
        output_feature_count = architecture.get("output_feature_count")
        num_layers = len(gate_neuron_counts)
        

        layers = [
            
            LSTMCell(
            pretrained=False,
            device_type=self.device_type,
            type="hidden",
            xt_input_count=self.input_feature_count if i == 0 else gate_neuron_counts[i-1],
            ht1_input_count=gate_neuron_counts[i],
            gate_neuron_count=gate_neuron_counts[i],
            gate_activation_fn=gate_activation_fn,
            index=i+1 ) for i in range(num_layers-1) ] + [

            LSTMCell(
            pretrained=False,
            device_type=self.device_type,
            type="output",
            xt_input_count=self.input_feature_count if num_layers == 1 else gate_neuron_counts[-2],
            ht1_input_count=gate_neuron_counts[-1],
            gate_neuron_count=gate_neuron_counts[-1],
            gate_nonlinearity=gate_activation_fn,
            why_neuron_count=output_feature_count,
            output_nonlinearity=output_activation_fn,
            index=num_layers ) 
        ] 
   
        return layers
    

    def saveParameters(self):
        os.makedirs(f"{self.save_fpath}", exist_ok=True)
        for layer in self.layers:
            layer.index = "0" + str(layer.index) if layer.index < 10 else layer.index
            torch.save(layer.wf, f"{self.save_fpath}/layer_{layer.index}_wf_{layer.gate_nonlinearity}.pth")
            torch.save(layer.wi, f"{self.save_fpath}/layer_{layer.index}_wi_{layer.gate_nonlinearity}.pth")
            torch.save(layer.wc, f"{self.save_fpath}/layer_{layer.index}_wc_{layer.gate_nonlinearity}.pth")
            torch.save(layer.wo, f"{self.save_fpath}/layer_{layer.index}_wo_{layer.gate_nonlinearity}.pth")

            torch.save(layer.bf, f"{self.save_fpath}/layer_{layer.index}_bf.pth")
            torch.save(layer.bi, f"{self.save_fpath}/layer_{layer.index}_bi.pth")
            torch.save(layer.bc, f"{self.save_fpath}/layer_{layer.index}_bc.pth")
            torch.save(layer.bo, f"{self.save_fpath}/layer_{layer.index}_bo.pth")

            if layer.type == "output":
                torch.save(layer.why, f"{self.save_fpath}/layer_{layer.index}_why_{layer.why_nonlinearity}.pth")
                torch.save(layer.by, f"{self.save_fpath}/layer_{layer.index}_by.pth")



 



    def resetHiddenState(self, batch_size):
        return [torch.zeros(
            size=(batch_size, layer.gate_neuron_count), 
            dtype=torch.float32, 
            device=self.device_type) for layer in self.layers]

    def resetCellState(self, batch_size):
        return [torch.zeros(
            size=(batch_size, layer.gate_neuron_count), 
            dtype=torch.float32, 
            device=self.device_type) for layer in self.layers]


    def calculateState(self, x, ht1_i, Ct1_i, wf_i, bf_i, wi_i, bi_i, wc_i, bc_i, wo_i, bo_i ):
        ht1_xt_i = torch.cat([ht1_i, x], dim=1)
        ft = activate(ht1_xt_i @ wf_i + bf_i, self.gate_nonlinearity)
        it = activate(ht1_xt_i @ wi_i + bi_i, self.gate_nonlinearity)
        Ct_tilde = torch.tanh(ht1_xt_i @ wc_i + bc_i)
        ot = activate(ht1_xt_i @ wo_i + bo_i, self.gate_nonlinearity)
        Ct = ft * Ct1_i + it * Ct_tilde
        ht_i = ot * torch.tanh(Ct)

        return ht_i
    


    def forwardNonAutoRegressive(self, X, training):
        """
        No autoregressive handling - working
        """
        
        T = X.shape[1] 
        batch_size = X.shape[0]

        if self.stateful and not self.state_initialized: 
            for layer in self.layers: 
                layer.generateState(batch_size)
            self.state_initialized = True

        ht1_l = [layer.ht1 for layer in self.layers] if self.stateful else self.resetHiddenState(batch_size)
        Ct1_l = [layer.Ct1 for layer in self.layers] if self.stateful else self.resetCellState(batch_size)
        
        wf_l = [layer.wf for layer in self.layers]
        bf_l = [layer.bf for layer in self.layers]
        wi_l = [layer.wi for layer in self.layers]
        bi_l = [layer.bi for layer in self.layers]
        wc_l = [layer.wc for layer in self.layers]
        bc_l = [layer.bc for layer in self.layers]
        wo_l = [layer.wo for layer in self.layers]
        bo_l = [layer.bo for layer in self.layers]

        why = self.layers[-1].why
        by = self.layers[-1].by

        output_feature_count = by.shape[0]
        Y = torch.zeros(batch_size, T, output_feature_count, device=self.device_type)

        for t in range(T):
            x = X[:, t, :]

            ht_l = [None] * self.num_layers

            ht_l[0] = self.calculateState(
                    x, 
                    ht1_l[0], Ct1_l[0], 
                    wf_l[0], bf_l[0], 
                    wi_l[0], bi_l[0], 
                    wc_l[0], bc_l[0], 
                    wo_l[0], bo_l[0] 
                )

            for i in range(1, self.num_layers):
                # print("num_layers:", self.num_layers)
                ht_l[i] = self.calculateState(
                    ht_l[i-1], 
                    ht1_l[i], Ct1_l[i], 
                    wf_l[i], bf_l[i], 
                    wi_l[i], bi_l[i], 
                    wc_l[i], bc_l[i], 
                    wo_l[i], bo_l[i] 
                )
            
            # print("done")
            Y[:, t, :] = activate( 
                ht_l[-1] @ why + by, self.why_nonlinearity)
            
            ht1_l = ht_l

        if self.stateful:
            for (layer, ht1_i) in zip(self.layers, ht1_l):
                layer.ht1 = ht1_i.detach()

        return Y
 


    def forwardAutoRegressive(self, X, training):
        """
        Autoregressive handling - working
        """
        
        T = X.shape[1] 
        batch_size = X.shape[0]

        if self.stateful and not self.state_initialized: 
            for layer in self.layers: 
                layer.generateState(batch_size)
            self.state_initialized = True

        ht1_l = [layer.ht1 for layer in self.layers] if self.stateful else self.resetHiddenState(batch_size)
        Ct1_l = [layer.Ct1 for layer in self.layers] if self.stateful else self.resetCellState(batch_size)
        
        wf_l = [layer.wf for layer in self.layers]
        bf_l = [layer.bf for layer in self.layers]
        wi_l = [layer.wi for layer in self.layers]
        bi_l = [layer.bi for layer in self.layers]
        wc_l = [layer.wc for layer in self.layers]
        bc_l = [layer.bc for layer in self.layers]
        wo_l = [layer.wo for layer in self.layers]
        bo_l = [layer.bo for layer in self.layers]

        why = self.layers[-1].why
        by = self.layers[-1].by

        output_feature_count = by.shape[0]
        Y = torch.zeros(batch_size, T, output_feature_count, device=self.device_type)

        x = X[:, 0, :]
        for t in range(T):

            ht_l = [None] * self.num_layers

            ht_l[0] = self.calculateState(
                    x, 
                    ht1_l[0], Ct1_l[0], 
                    wf_l[0], bf_l[0], 
                    wi_l[0], bi_l[0], 
                    wc_l[0], bc_l[0], 
                    wo_l[0], bo_l[0] 
                )

            for i in range(1, self.num_layers):

                ht_l[i] = self.calculateState(
                    ht_l[i-1], 
                    ht1_l[i], Ct1_l[i], 
                    wf_l[i], bf_l[i], 
                    wi_l[i], bi_l[i], 
                    wc_l[i], bc_l[i], 
                    wo_l[i], bo_l[i] 
                )
            
            y = activate( 
                ht_l[-1] @ why + by, self.why_nonlinearity)
            Y[:, t, :] = y
            
            ht1_l = ht_l

            if training:
                teacher_forcing_factor = (
                    self.epoch/self.epochs if (self.teacher_forcing == "progressive") # progressive teacher-forcing
                    else 1 - self.epoch/self.epochs if (self.teacher_forcing == "regressive") # regressive teacher-forcing
                    else float(self.teacher_forcing) if isinstance(self.teacher_forcing, (int, float)) 
                    else 0
                )
                x = X[:, t+1, :] if t < teacher_forcing_factor*(T-1) else y.detach() 
               
            else: x = y.detach()

            # print(x.shape)
            # exit()

        if self.stateful:
            for (layer, ht1_i) in zip(self.layers, ht1_l):
                layer.ht1 = ht1_i.detach()

        return Y
    



    def forward(self, X, training):
        """ if this is uncommented, it will force the pure teacher forcing LSTM implementations to run inference auto regressively (which i believe is the correct way to do it)
        however, since pure auto regressive inference currently sucks (both for the auto regressive and teacher forced implementations), i will leave it commented out such that I can at least see good results at inference for the teacher forced implementations
        but yeah i think this would be the more correct way to do it (i.e. to always run inference auto regressively) """
        # if self.autoregressive or not training: 
        if self.autoregressive:
            return self.forwardAutoRegressive(X, training)
        return self.forwardNonAutoRegressive(X, training)





    # def backprop(self, loss):
    #     self.zerograd()

    #     loss.backward()

    #     if self.grad_clip_norm is not None or self.grad_clip_value is not None:
    #         self.clipGradients()

    #     with torch.no_grad():

    #         if not self.optimizer:
    #             self.update()
    #         else:
    #             self.t += 1
    #             self.optimizerUpdate()
