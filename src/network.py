import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.functional as F
# from src.encoder import Encoder
# from src.decoder import Decoder


class Network:

    def __init__(self, model_type, training, **kwargs):

        self.model_type = model_type

        if training:
            hyperparameters = kwargs.get("hyperparameters")
            self.learn_rate = hyperparameters.get("learn_rate")
            self.batch_size = hyperparameters.get("batch_size")
            self.loss_func = hyperparameters.get("loss_func")
            self.reduction = hyperparameters.get("reduction")
            self.optimizer = hyperparameters.get("optimizer")
            self.lambda_L2 = hyperparameters.get("lambda_L2")  # regularization strength which controls how much the L2 penalty influences the loss. Larger λ values increase the regularization effect.
            self.dropout_rate = hyperparameters.get("dropout_rate")
            self.grad_clip_norm  = hyperparameters.get('grad_clip_norm')
            self.grad_clip_value = hyperparameters.get('grad_clip_value')
            self.epoch = None # [FIXED???] temp hack (should ideally be under the 'if training' block) for the rnn implementation. update later.
            self.epochs = None # [FIXED???] temp hack (should ideally be under the 'if training' block) for the rnn implementation. update later





    def collect_parameters(self):
        """
        Walk self.layers and return a flat list of all weight Tensors whose
        .grad you want to clip. Adapt these attribute names to match your layers.
        """
        params = []
        for layer in self.layers:
            if self.model_type == "cnn" and layer.type == "convolutional":
                params.extend([layer.kernels, layer.biases])
            elif self.model_type == "mlp":
                params.extend([layer.weights, layer.biases])
            elif self.model_type == "rnn":
                params.append(layer.wxh)
                params.extend([layer.whh, layer.bh])
                if layer.type == "output":
                    params.extend([layer.why, layer.by])
        return params


    def clip_gradients(self):
        """
        Apply PyTorch's native clipping to the gradients in-place.
        Call immediately after loss.backward().
        """
        params = self.collect_parameters()
        # print(len(params))

        # grads = [p.grad for p in params if p.grad is not None]
        # total_grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
        # print("Before clip, grad norm:", total_grad_norm.item())
        
        if self.grad_clip_norm is not None:
            clip_grad_norm_(params, self.grad_clip_norm)
        if self.grad_clip_value is not None:
            clip_grad_value_(params, self.grad_clip_value)

        # grads = [p.grad for p in params if p.grad is not None]
        # total_grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
        # print("After clip, grad norm:", total_grad_norm.item())





    def set_optimizer(self):

        # match self.optimizer:

        if self.optimizer == "adam":
            self.t = 0
            self.weight_moment_list = [[0, 0]]*self.num_layers
            self.bias_moment_list = [[0, 0]]*self.num_layers
            
            if self.model_type == "rnn":
                self.wxh_moment_list = [[0, 0]]*self.num_layers
                # although only the output layer has why and by, 
                # we must still multiply this list by self.num_layers since all of these moment lists are accessed via layer index
                self.why_moment_list = [[0, 0]]*self.num_layers 
                self.by_moment_list = [[0, 0]]*self.num_layers

            elif self.model_type == "lstm":
                self.wf_moment_list = [[0, 0]]*self.num_layers
                self.wi_moment_list = [[0, 0]]*self.num_layers
                self.wc_moment_list = [[0, 0]]*self.num_layers
                self.wo_moment_list = [[0, 0]]*self.num_layers

                self.bf_moment_list = [[0, 0]]*self.num_layers
                self.bi_moment_list = [[0, 0]]*self.num_layers
                self.bc_moment_list = [[0, 0]]*self.num_layers
                self.bo_moment_list = [[0, 0]]*self.num_layers

                self.why_moment_list = [[0, 0]]*self.num_layers
                self.by_moment_list = [[0, 0]]*self.num_layers
                
            elif self.model_type == "transformer":
                
                self.WQ_moment_list = [[0, 0]]*self.num_layers
                self.WK_moment_list = [[0, 0]]*self.num_layers
                self.WV_moment_list = [[0, 0]]*self.num_layers
                self.WO_moment_list = [[0, 0]]*self.num_layers
                
                self.WQ_masked_moment_list = [[0, 0]]*self.num_layers
                self.WK_masked_moment_list = [[0, 0]]*self.num_layers
                self.WV_masked_moment_list = [[0, 0]]*self.num_layers
                self.WO_masked_moment_list = [[0, 0]]*self.num_layers

                # self.ffw1_moment_list = [[0, 0]]*self.num_layers
                # self.ffb1_moment_list = [[0, 0]]*self.num_layers
                # self.ffw2_moment_list = [[0, 0]]*self.num_layers
                # self.ffb2_moment_list = [[0, 0]]*self.num_layers
                
                self.linear_moment_list = [[0, 0]]*self.num_layers





    def inference(self, data):
        with torch.no_grad(): # new
            return self.forward(data, training=False)





    ## def train(self, data, target, epochs, save_params=True):
    # def train(self, **kwargs):
        
    #     data, target, epochs, save_params = (
    #         kwargs.get("data"),  kwargs.get("target"), 
    #         kwargs.get('epochs'),  kwargs.get("save_params", True)
    #     )

    #     epoch_plt = []
    #     loss_plt = []
    #     self.epochs = epochs

    #     if not self.batch_size: self.batch_size = data.shape[0]

    #     for epoch in range(epochs):
            
    #         self.epoch = epoch+1

    #         data_batch, target_batch = self.batch(data, target)
    #         pred_batch = self.forward(data_batch, training=True, **kwargs)
    #         # print(f"pred: {pred_batch.shape}")

    #         loss = getattr(self, self.loss_func)(pred_batch, target_batch)

    #         if self.lambda_L2:
    #             loss += self.l2_regularization()
    #         self.backprop(loss)


    #         epoch_plt.append(epoch)
    #         loss_plt.append(loss.item())
    #         print(f"epoch = {epoch+1}, loss = {loss}")
    #         print(f"__________________________________________")
            

    #     if save_params:
    #         self.save_parameters()
            
    #     return epoch_plt, loss_plt
    

    def train(self, data, target, epochs, save_params=True):

        epoch_plt = []
        loss_plt = []
        self.epochs = epochs

        if not self.batch_size: self.batch_size = data.shape[0]

        for epoch in range(epochs):
            
            self.epoch = epoch+1

            data_batch, target_batch = self.batch(data, target)
            pred_batch = self.forward(data_batch, training=True)

            loss = getattr(self, self.loss_func)(pred_batch, target_batch)

            if self.lambda_L2:
                loss += self.l2_regularization()
            self.backprop(loss)


            epoch_plt.append(epoch)
            loss_plt.append(loss.item())
            print(f"epoch = {epoch+1}, loss = {loss}")
            print(f"__________________________________________")
            

        if save_params:
            self.save_parameters()
            
        return epoch_plt, loss_plt





    def reduce(self, x):
        if self.reduction == "mean":
            return x.mean()
        if self.reduction == "sum":
            return x.sum()





    # def batch(self, data, target):
    #     batch_indices = torch.randperm(n=data.size(dim=0))[:self.batch_size]  # stochastic

    #     data_batch = data[batch_indices]

    #     target_batch = target.T[batch_indices].T

    #     return data_batch, target_batch


    def batch(self, data, target):
        # print( "yoooo", self.batch_size)
        if self.batch_size:
            # print("batched")
            # batch_indices = torch.randperm(n=data.size(dim=0))[:self.batch_size]  # stochastic
            batch_indices = torch.randperm(n=data.shape[0])[:self.batch_size]  # stochastic

            data_batch = data[batch_indices]
            target_batch = target[batch_indices]
            return data_batch, target_batch

        
        return data, target


    # def CELoss(self, logits, labels):
    #     # print(logits.shape)
    #     # print(labels.shape)
    #     # exit()
    #     loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    #     loss = loss_fn(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))
    #     return loss

    def CELoss(self, logits, targets):
        """
        logits: Tensor of shape (B, S, V) — raw scores from model
        targets: Tensor of shape (B, S) — token IDs
        ignore_index: token ID to ignore in loss computation (e.g., pad_token_id)
        """
        ignore_index=self.tokenizer.pad_token_id
        
        # Numerical stability: subtract max logit from each row
        max_logits = logits.max(dim=1, keepdim=True).values  # (B*S, 1)
        logits_stable = logits - max_logits                  # (B*S, V)

        # Exponentiate logits
        exp_logits = torch.exp(logits_stable)                # (B*S, V)

        # Compute partition function (sum over vocab)
        sum_exp = exp_logits.sum(dim=1, keepdim=True)        # (B*S, 1)

        # Log-Softmax: log(exp(logit) / sum_exp)
        log_probs = logits_stable - torch.log(sum_exp)       # (B*S, V)

        # Select the log-prob for the true class
        idx = torch.arange(logits.size(0))
        selected_log_probs = log_probs[idx, targets]         # (B*S,)

        # Mask out ignored indices
        if ignore_index is not None:
            mask = targets != ignore_index
            selected_log_probs = selected_log_probs[mask]

        # Negative log likelihood
        loss = -selected_log_probs.mean()
            
        ce_loss_reduced = self.reduce(loss)
        return ce_loss_reduced



    def CCELoss(self, pred_batch, target_batch):

        epsilon = 1e-8
        pred_batch = torch.clamp(pred_batch, epsilon, 1 - epsilon)
        errs = torch.mul(target_batch, torch.log(pred_batch))

        cce_loss = -torch.sum(errs, dim=0)  # CCE (Categorical Cross Entropy) Loss
        cce_loss_reduced = self.reduce(cce_loss)

        return cce_loss_reduced




    def CCELossNN(self, data, target):
        criterion = nn.CrossEntropyLoss()

        pred = self.forward(data).reshape(1, -1)
        target = torch.tensor([torch.nonzero(target == 1.0)[:, 0].item()])
        return criterion(pred, target)



    def BCEWithLogitsLoss(self, pred_batch, target_batch):

        max_val = torch.clamp(pred_batch, min=0)
        stable_log_exp = max_val + torch.log(1 + torch.exp(-torch.abs(pred_batch)))

        errs = stable_log_exp - torch.mul(target_batch, pred_batch)

        bce_with_logits_loss = torch.sum(errs, dim=0)  # BCE with logits loss (DO NOT USE SIGMOID ACTIVATION)
        bce_with_logits_loss_reduced = self.reduce(bce_with_logits_loss)

        return bce_with_logits_loss_reduced







    def BCELoss(self, pred_batch, target_batch):

        epsilon = 1e-8
        pred_batch = torch.clamp(pred_batch, epsilon, 1 - epsilon)
        errs = target_batch * torch.log(pred_batch) + (1 - target_batch) * torch.log(1 - pred_batch)

        bce_loss = -(1/self.batch_size)*torch.sum(errs, dim=0)  # BCE (Binary Cross Entropy) Loss
        bce_loss_reduced = self.reduce(bce_loss)

        return bce_loss_reduced



    def FocalBCELoss(self, pred_batch, target_batch):

        """Hyperparameter"""
        alpha = 0.7  # positive class weighting
        gamma = 2  #
        epsilon = 1e-8

        pred_batch = torch.clamp(pred_batch, epsilon, 1 - epsilon)
        errs = alpha * torch.pow((1-pred_batch), gamma) * (target_batch * torch.log(pred_batch) + (1-target_batch) * torch.log(1-pred_batch))

        focal_bce_loss = -(1/self.batch_size)*torch.sum(errs, dim=0)  # Focal BCE (Binary Cross Entropy) Loss
        focal_bce_loss_reduced = self.reduce(focal_bce_loss)

        return focal_bce_loss_reduced



    def MSELoss(self, pred_batch, target_batch):
        print(pred_batch.shape)
        print(target_batch.shape)
        errs = (pred_batch - target_batch)**2
        # mse_loss = (1/self.batch_size)*torch.sum(errs, dim=0) if self.batch_size else (1/pred_batch.shape[0])*torch.sum(errs, dim=0) # MSE (Mean Squared Error) Loss
        mse_loss = (1/self.batch_size)*torch.sum(errs, dim=0) # MSE (Mean Squared Error) Loss
        mse_loss_reduced = self.reduce(mse_loss)

        return mse_loss_reduced
    
    def SSELoss(self, pred_batch, target_batch):
        errs = (pred_batch - target_batch)**2
        sse_loss = torch.sum(errs, dim=0)  # SSE (Sum Squared Error) Loss
        sse_loss_reduced = self.reduce(sse_loss)

        return sse_loss_reduced



    def update(self):

        for layer in self.layers:

            if self.model_type == "cnn" and layer.type == "convolutional":
                layer.kernels -= self.learn_rate * layer.kernels.grad
                layer.biases -= self.learn_rate * layer.biases.grad

            elif self.model_type == "mlp":
                layer.weights -= self.learn_rate * layer.weights.grad
                layer.biases -= self.learn_rate * layer.biases.grad

            elif self.model_type == "rnn":
                layer.wxh -= self.learn_rate * layer.wxh.grad
                layer.whh -= self.learn_rate * layer.whh.grad
                layer.bh -= self.learn_rate * layer.bh.grad

                if layer.type == "output":
                    layer.why -= self.learn_rate * layer.why.grad
                    layer.by -= self.learn_rate * layer.by.grad

            elif self.model_type == "lstm":
                layer.wf -= self.learn_rate * layer.wf.grad
                layer.wi -= self.learn_rate * layer.wi.grad
                layer.wc -= self.learn_rate * layer.wc.grad
                layer.wo -= self.learn_rate * layer.wo.grad

                layer.bf -= self.learn_rate * layer.bf.grad
                layer.bi -= self.learn_rate * layer.bi.grad
                layer.bc -= self.learn_rate * layer.bc.grad
                layer.bo -= self.learn_rate * layer.bo.grad

                if layer.type == "output":
                    layer.why -= self.learn_rate * layer.why.grad
                    layer.by -= self.learn_rate * layer.by.grad
    
            elif self.model_type == "transformer":
                
                layer.WQ -= self.learn_rate * layer.WQ.grad
                layer.WK -= self.learn_rate * layer.WK.grad
                layer.WV -= self.learn_rate * layer.WV.grad
                layer.WO -= self.learn_rate * layer.WO.grad

                # for ff_layer in layer.ff_layers: 
                #     ff_layer.weights -= self.learn_rate * ff_layer.weights.grad           
                #     ff_layer.biases -= self.learn_rate * ff_layer.biases.grad           
                
                if layer.component == "decoder":
                    layer.WQ_masked -= self.learn_rate * layer.WQ_masked.grad
                    layer.WK_masked -= self.learn_rate * layer.WK_masked.grad
                    layer.WV_masked -= self.learn_rate * layer.WV_masked.grad
                    layer.WO_masked -= self.learn_rate * layer.WO_masked.grad
                    if layer.type == "output":
                        layer.linear -= self.learn_rate * layer.linear.grad
    
    
    def adam(self, layer_index, gt, param_type, *args):

        moment_list = (
            self.weight_moment_list if param_type == "weight" 
            else self.bias_moment_list if param_type == "bias" 
            else getattr(self, f"{param_type}_moment_list")
        )

        mt_1, vt_1 = moment_list[layer_index]

        """Hyperparameter"""
        beta1 = 0.9  # first moment estimate decay rate (smaller = more aggressive)
        beta2 = 0.999  # second moment estimate decay rate (smaller = more aggressive)
        epsilon = 1e-8

        mt = beta1*mt_1 + (1-beta1)*gt
        vt = beta2*vt_1 + (1-beta2)*gt**2
        mt_hat = mt/(1-beta1**self.t)
        vt_hat = vt/(1-beta2**self.t)

        moment_list[layer_index] = [mt, vt]

        adam_grad = (self.learn_rate*mt_hat)/(torch.sqrt(vt_hat) + epsilon)

        return adam_grad




    def optimizer_update(self):

        optimizer_func = getattr(self, self.optimizer)

        for layer in self.layers:

            if self.model_type == "cnn" and layer.type == "convolutional":
                # layer_index = self.layers.index(layer)
                layer_index = layer.index-1 # why am i not just doing this???
                layer.kernels -= optimizer_func(layer_index=layer_index, gt=layer.kernels.grad, param_type="weight")
                layer.biases -= optimizer_func(layer_index=layer_index, gt=layer.biases.grad, param_type="bias")
                # print(layer.kernels)

            elif self.model_type == "mlp":
                # layer_index = self.layers.index(layer)
                layer_index = layer.index-1 # why am i not just doing this???
                layer.weights -= optimizer_func(layer_index=layer_index, gt=layer.weights.grad, param_type="weight")
                layer.biases -= optimizer_func(layer_index=layer_index, gt=layer.biases.grad, param_type="bias")

            elif self.model_type == "rnn":
                # layer_index = self.layers.index(layer)
                layer_index = layer.index-1 # why am i not just doing this???
                layer.wxh -= optimizer_func(layer_index=layer_index, gt=layer.wxh.grad, param_type="wxh")
                layer.whh -= optimizer_func(layer_index=layer_index, gt=layer.whh.grad, param_type="weight")
                layer.bh -= optimizer_func(layer_index=layer_index, gt=layer.bh.grad, param_type="bias")
            
                if layer.type == "output":
                    layer.why -= optimizer_func(layer_index=layer_index, gt=layer.why.grad, param_type="why")
                    layer.by -= optimizer_func(layer_index=layer_index, gt=layer.by.grad, param_type="by")
            
            elif self.model_type == "lstm":
                # layer_index = self.layers.index(layer)
                layer_index = layer.index-1 # why am i not just doing this???
                layer.wf -= optimizer_func(layer_index=layer_index, gt=layer.wf.grad, param_type="wf")
                layer.wi -= optimizer_func(layer_index=layer_index, gt=layer.wi.grad, param_type="wi")
                layer.wc -= optimizer_func(layer_index=layer_index, gt=layer.wc.grad, param_type="wc")
                layer.wo -= optimizer_func(layer_index=layer_index, gt=layer.wo.grad, param_type="wo")

                layer.bf -= optimizer_func(layer_index=layer_index, gt=layer.bf.grad, param_type="bf")
                layer.bi -= optimizer_func(layer_index=layer_index, gt=layer.bi.grad, param_type="bi")
                layer.bc -= optimizer_func(layer_index=layer_index, gt=layer.bc.grad, param_type="bc")
                layer.bo -= optimizer_func(layer_index=layer_index, gt=layer.bo.grad, param_type="bo")

                if layer.type == "output":
                    layer.why -= optimizer_func(layer_index=layer_index, gt=layer.why.grad, param_type="why")
                    layer.by -= optimizer_func(layer_index=layer_index, gt=layer.by.grad, param_type="by")
            
            elif self.model_type == "transformer":
                # layer_index = self.layers.index(layer)
                layer_index = layer.index-1 # why am i not just doing this???
                layer.WQ -= optimizer_func(layer_index=layer_index, gt=layer.WQ.grad, param_type="WQ")
                layer.WK -= optimizer_func(layer_index=layer_index, gt=layer.WK.grad, param_type="WK")
                layer.WV -= optimizer_func(layer_index=layer_index, gt=layer.WV.grad, param_type="WV")
                layer.WO -= optimizer_func(layer_index=layer_index, gt=layer.WO.grad, param_type="WO")
                
                if layer.component == "decoder":
                    layer.WQ_masked -= optimizer_func(layer_index=layer_index, gt=layer.WQ_masked.grad, param_type="WQ_masked")
                    layer.WK_masked -= optimizer_func(layer_index=layer_index, gt=layer.WK_masked.grad, param_type="WK_masked")
                    layer.WV_masked -= optimizer_func(layer_index=layer_index, gt=layer.WV_masked.grad, param_type="WV_masked")
                    layer.WO_masked -= optimizer_func(layer_index=layer_index, gt=layer.WO_masked.grad, param_type="WO_masked")

                    if layer.type == "output":
                        layer.linear -= optimizer_func(layer_index=layer_index, gt=layer.linear.grad, param_type="linear")






    def l2_regularization(self):

        weight_sum = 0

        for layer in self.layers:
            if self.model_type == "cnn" and layer.type == "convolutional":
                weight_sum += torch.sum(layer.kernels ** 2)
            elif self.model_type == "mlp":
                weight_sum += torch.sum(layer.weights ** 2)
            elif self.model_type == "rnn":
                weight_sum += ( torch.sum(layer.wxh ** 2) + torch.sum(layer.whh ** 2) )
                if layer.type == "output":
                    weight_sum += torch.sum(layer.why ** 2)
                weight_sum = torch.mean(weight_sum)


        regularization = self.lambda_L2*weight_sum
        # print(regularization)

        return regularization







    def check_config(self, architecture):

        config_lengths = [len(v) for k, v in architecture.items()]
        all_same_length = all(config_length == config_lengths[0] for config_length in config_lengths)

        if not all_same_length:
            raise IndexError(f"{self.model_type} Configuration Error. Recheck sizes of configuration objects: {config_lengths}")






    def print_layers(self):
        for layer in self.layers:
            print(layer)

        if self.model_type == "cnn":
            for layer in self.MLP.layers:
                print(layer)



    def zerograd(self):

        for layer in self.layers:

            if self.model_type == "cnn" and layer.type == "convolutional":
                layer.kernels.grad = None
                layer.biases.grad = None

            elif self.model_type == "mlp":
                layer.weights.grad = None
                layer.biases.grad = None

            elif self.model_type == "rnn":
                layer.wxh.grad = None
                layer.whh.grad = None
                layer.bh.grad = None
                if layer.type == "output":
                    layer.why.grad = None
                    layer.by.grad = None

            elif self.model_type == "lstm":
                layer.wf.grad = None
                layer.wi.grad = None
                layer.wc.grad = None
                layer.wo.grad = None
                layer.bf.grad = None
                layer.bi.grad = None
                layer.bc.grad = None
                layer.bo.grad = None
                if layer.type == "output":
                    layer.why.grad = None
                    layer.by.grad = None
                    
            elif self.model_type == "transformer":
                
                layer.WQ.grad = None
                layer.WK.grad = None
                layer.WV.grad = None
                layer.WO.grad = None

                # for ff_layer in layer.ff_layers:
                #     ff_layer.weights.grad = None         
                #     ff_layer.biases.grad = None         
                
                if layer.component == "decoder":
                    layer.WQ_masked.grad = None
                    layer.WK_masked.grad = None
                    layer.WV_masked.grad = None
                    layer.WO_masked.grad = None
                    if layer.type == "output":
                        layer.linear.grad = None
                    



    def backprop(self, loss):
        self.zerograd()

        loss.backward()

        if self.grad_clip_norm is not None or self.grad_clip_value is not None:
            self.clip_gradients()

        with torch.no_grad():

            if not self.optimizer:
                self.update()
            else:
                self.t += 1
                self.optimizer_update()