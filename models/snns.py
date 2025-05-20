#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the Spiking Neural Network (SNN) baseline is defined using the
surrogate gradient method.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from einops import rearrange, repeat
# import os
# import sys
# fscil_directory = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(fscil_directory,"../../Dilated"))
from DCLS.construct.modules import Dcls1d


class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x

class SpikeFunctionSuperSpike(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_out = grad_x / (1.0 + 10.0*torch.abs(x))
        return grad_out

class SpikeFunctionSLAYER(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        alpha=5
        c=0.4
        grad_out = grad_x * c * alpha / (2 * torch.exp(x.abs() * alpha))
        return grad_out

# def mem_reset(mem, thresh):
#     """Generates detached reset signal if mem > threshold.
#     Returns reset."""
#     mem_shift = mem - thresh
#     reset = SpikeFunctionBoxcar.apply(mem_shift).clone().detach()

#     return reset

def gating_function(gate, alpha_exp, input_size, gate_size, gate_bias=False, extra=None):
    gamma = None
    if 'mamba' in gate:
        if gate == 'mamba':
            Wgate = nn.Linear(input_size, gate_size, bias=gate_bias)
        elif gate == 'mamba_dtscalar':
            Wgate = nn.Linear(input_size, 1, bias=gate_bias)        
        if alpha_exp:
            gate = lambda x,alpha : torch.exp(-F.softplus(Wgate(x))*alpha) 
        else:
            gate = lambda x,alpha : torch.exp(-F.softplus(Wgate(x)))*alpha
    else:
        Wgate = nn.Linear(input_size, gate_size, bias=gate_bias)
        if gate == "gla":
            input_gate = lambda x : torch.pow(torch.sigmoid(Wgate(x)), 1/16) #extra['tau']
        elif gate == 'mLSTM':
            input_gate = lambda x : torch.sigmoid(Wgate(x))
        elif gate == 'hgrn':
            gamma = nn.Parameter(torch.rand(gate_size))
            nn.init.uniform_(gamma, a=0.0, b=1.0)
            def input_gate(x):
                gamma_eff = torch.clamp(gamma, 0, 1)
                return gamma_eff + (1-gamma_eff) * torch.sigmoid(Wgate(x))    
        elif gate == 'rwkv':
            input_gate = lambda x : torch.exp(-torch.exp(Wgate(x)))     
        if alpha_exp:
            gate = lambda x,alpha : input_gate(x)*torch.exp(-alpha) 
        else:
            gate = lambda x,alpha : input_gate(x)*alpha           
    return gate, Wgate, gamma 

def pad_to(tensor, target_size, dim):
    """
    Pads `tensor` with zeros on the right *along* dimension `dim`
    so that its size[dim] == target_size.
    Works for tensors of *any* rank.
    """
    # total numbers in the pad tuple = 2 * tensor.ndim
    pad = [0] * (2 * tensor.ndim)
    # we want to pad the “right” side of dimension `dim`
    # in F.pad, pad = (pad_last_dim_left, pad_last_dim_right,
    #                 pad_2ndlast_left,   pad_2ndlast_right, …)
    # index of the “right” pad for dim k is:
    #   2*(tensor.ndim - k - 1) + 1
    pad_index = 2 * (tensor.ndim - dim - 1) + 1
    pad[pad_index] = target_size - tensor.size(dim)
    return F.pad(tensor, pad)

class SNN(nn.Module):
    """
    A multi-layered Spiking Neural Network (SNN).

    It accepts input tensors formatted as (batch, time, feat). In the case of
    4d inputs like (batch, time, feat, channel) the input is flattened as
    (batch, time, feat*channel).

    The function returns the outputs of the last spiking or readout layer
    with shape (batch, time, feats) or (batch, feats) respectively, as well
    as the firing rates of all hidden neurons with shape (num_layers*feats).

    Arguments
    ---------
    input_shape : tuple
        Shape of an input example.
    layer_sizes : int list
        List of number of neurons in all hidden layers
    neuron_type : str
        Type of neuron model, either 'LIF', 'adLIF', 'RLIF' or 'RadLIF'.
    threshold : float
        Fixed threshold value for the membrane potential.
    dropout : float
        Dropout rate (must be between 0 and 1).
    normalization : str
        Type of normalization (batchnorm, layernorm). Every string different
        from batchnorm and layernorm will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    use_readout_layer : bool
        If True, the final layer is a non-spiking, non-recurrent LIF and outputs
        a cumulative sum of the membrane potential over time. The outputs have
        shape (batch, labels) with no time dimension. If False, the final layer
        is the same as the hidden layers and outputs spike trains with shape
        (batch, time, labels).
    """

    def __init__(
        self,
        input_shape,
        layer_sizes,
        neuron_type="LIF",
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        use_readout_layer=True,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.reshape = True if len(input_shape) > 3 else False
        self.input_size = float(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_outputs = layer_sizes[-1]
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.use_readout_layer = use_readout_layer
        self.is_snn = True
        self.jaxreadout = extra_features.get('jaxreadout', False)
        self.delayreadout = True if "delay" in neuron_type.lower() else False

        self.extra_features = extra_features

        # if neuron_type not in ["LIF", "CSiLIF", "SiLIF", "adLIF", "CadLIF", "CadLIFAblation", "RAFAblation", "RingInitLIFcomplex", "BRF", "ResonateFire", "RSEadLIF", "LIFfeature", "adLIFnoClamp","LIFfeatureDim", "adLIFclamp", "RLIF", "RadLIF", "LIFcomplex", "LIFcomplexBroad", "LIFrealcomplex","ReLULIFcomplex", "RLIFcomplex","RLIFcomplex1MinAlphaNoB","RLIFcomplex1MinAlpha", "LIFcomplex_gatedB", "LIFcomplex_gatedDt", "LIFcomplexDiscr", "DelaySiLIF", "DelayReadout"]:
        #     raise ValueError(f"Invalid neuron type {neuron_type}")

        # Init trainable parameters
        self.snn = self._init_layers()

    def _init_layers(self):

        snn = nn.ModuleList([])
        input_size = self.input_size
        snn_class = self.neuron_type + "Layer"
        
        

        if self.use_readout_layer:
            num_hidden_layers = self.num_layers - 1
        else:
            num_hidden_layers = self.num_layers

        #Hidden layers
        if self.extra_features.get('use_input_layer', False):
            input_class = self.extra_features.get('input_layer_type', False) + "Layer"
            extra_in = self.extra_features
            extra_in["dt_min"] = extra_in.get('dt_min_in', extra_in.get("dt_min"))
            extra_in["dt_max"] = extra_in.get('dt_max_in', extra_in.get("dt_max"))
            extra_in["alpha"] = extra_in.get('alpha_in', None)
            extra_in["alpha_range"] = extra_in.get('alpha_range_in', None)
            extra_in["alpha_range_min"] = extra_in.get('alpha_range_min_in', None)
            extra_in["alpha_range_max"] = extra_in.get('alpha_range_max_in', None)
            extra_in["alpha_img_pi_rat"] = extra_in.get('alpha_img_pi_rat_in', None)
            extra_in["alpha_rand"] = extra_in.get('alpha_rand_in', None)

            
            snn.append(
                globals()[input_class](
                    input_size=input_size,
                    hidden_size=self.layer_sizes[0],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                    bidirectional=self.bidirectional,
                    extra_features = extra_in
                )
            )
            input_size = self.layer_sizes[0] * (1 + self.bidirectional)
            layer_range = range(1,num_hidden_layers)
        else:
            layer_range = range(0,num_hidden_layers)
        
        for i in layer_range:
            snn.append(
                globals()[snn_class](
                    input_size=input_size,
                    hidden_size=self.layer_sizes[i],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                    bidirectional=self.bidirectional,
                    extra_features = self.extra_features
                )
            )
            input_size = self.layer_sizes[i] * (1 + self.bidirectional)

        # Readout layer
        if self.use_readout_layer:
            if self.neuron_type == 'RSEadLIF':
                snn.append(
                    SEReadoutLayer(
                        input_size=input_size,
                        hidden_size=self.layer_sizes[-1],
                        batch_size=self.batch_size,
                        dropout=self.dropout,
                        normalization=self.normalization,
                        use_bias=self.use_bias,
                    )
                )
            elif self.jaxreadout:
                snn.append(
                    JaxReadoutLayer(
                        input_size=input_size,
                        hidden_size=self.layer_sizes[-1],
                        batch_size=self.batch_size,
                        dropout=self.dropout,
                        normalization=self.normalization,
                        use_bias=self.use_bias,
                        extra_features=self.extra_features
                    )
                )
            elif self.delayreadout:
                snn.append(
                    DelayReadoutLayer(
                        input_size=input_size,
                        hidden_size=self.layer_sizes[-1],
                        batch_size=self.batch_size,
                        dropout=self.dropout,
                        normalization=self.normalization,
                        use_bias=self.use_bias,
                        extra_features=self.extra_features
                    )
                )                
            else:
                snn.append(
                    ReadoutLayer(
                        input_size=input_size,
                        hidden_size=self.layer_sizes[-1],
                        batch_size=self.batch_size,
                        dropout=self.dropout,
                        normalization=self.normalization,
                        use_bias=self.use_bias,
                        extra_features=self.extra_features
                    )
                )

        return snn

    def forward(self, x):

        # Reshape input tensors to (batch, time, feats) for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            else:
                raise NotImplementedError

        # Process all layers
        all_spikes = [x.detach()]


        if self.extra_features['residual']:
            res = 0
            for i, snn_lay in enumerate(self.snn):
                if not (self.use_readout_layer and i == self.num_layers - 1):
                    x = snn_lay(x) + res
                    res = x
                    all_spikes.append(x.detach())  
                else:
                    x = snn_lay(x)
        else:
            for i, snn_lay in enumerate(self.snn):
                x = snn_lay(x)
                if not (self.use_readout_layer and i == self.num_layers - 1):
                    all_spikes.append(x.detach())

        # Compute mean firing rate of each spiking neuron
        with torch.no_grad():
            try:
                firing_rates = torch.cat(all_spikes, dim=2)
                firing_rates[firing_rates>0.9] = 1 # To account for dropout puting spikes at 
                firing_rates = firing_rates.mean(dim=(0, 1))
                sop = 0.0
                for i, spikes in enumerate(all_spikes):
                    sop += torch.sum(spikes!=0, dim=(1,2)).mean(dtype=float).item()*self.layer_sizes[i]
            except:
                pad_dim = 1
                max_len = max(t.size(pad_dim) for t in all_spikes)
                padded = [pad_to(t, max_len, pad_dim) for t in all_spikes]

                firing_rates = torch.cat(padded, dim=2).mean(dim=(0, 1))
                sop = 0.0
                for i, spikes in enumerate(all_spikes):
                    sop += torch.sum(spikes!=0, dim=(1,2)).mean(dtype=float).item()*self.layer_sizes[i]

                for snn_lay, x in zip(self.snn, all_spikes):
                    delays = torch.clamp(30 - self.W.P.squeeze().int(), max=30)
                    kernel_sizes = torch.zeros_like(delays)
                    for i in range(delays.shape[1]):
                        z = torch.zeros_like(x[0])
                        z[0,0] = 1
                        z = z.permute(1,0)
                        z = F.pad(z, (snn_lay.left_padding, snn_lay.right_padding), 'constant', 0)  # we use padding for the delays kernel
                        Wz = self.W(z)
                        kernel_sizes[:,i] = torch.sum(Wz[0,:,:]!=0, dim=1)
                    
                    



        

        return x, firing_rates, sop

    # Only for Delay modules
    def decrease_sig(self, epoch):

        # Decreasing to 0.23 instead of 0.5

        alpha = 0
        sig = self.snn[-1].W.SIG[0,0,0,0].detach().cpu().item()
        if epoch <25 and sig > 0.23:
            alpha = (0.23/self.snn[-1].sig_init)**(1/(25))

            for layer in self.snn:
                layer.W.SIG *= alpha
                # No need to clamp after modifying sigma
                #block[0][0].clamp_parameters()


class CSiLIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        alpha_img =  math.pi * torch.ones(self.hidden_size) 
            
        self.register("log_dt", log_dt, lr=0.001)
        self.register("log_log_alpha", log_log_alpha, lr=0.001)
        self.register("alpha_img", alpha_img, lr=0.001)

        self.b = nn.Parameter(torch.rand(self.hidden_size))
        self.reset_factor = 0.5


        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        alpha_cont = -torch.exp(self.log_log_alpha)+1j*self.alpha_img
        alpha_cont *=torch.exp(self.log_dt)
        alpha = torch.exp(alpha_cont)         
            
        for t in range(Wx.shape[1]):

            reset = st

            # Compute membrane potential (LIF)
            ut = alpha * (ut - self.reset_factor*reset) + self.b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)
        

        return torch.stack(s, dim=1)

class LIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False
    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):
            
            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            ut = alpha * (ut - reset) + (1 - alpha) * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class LIFfeatureLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features="_"
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.extra_features = extra_features
        device = torch.device("cuda")
        if "1-200_1-5"  in extra_features:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 200)]
        else:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        dt = 1
        if "Dt1ms" in extra_features:
            dt = 0.001
        elif "Dt1" in extra_features:
            dt = 1
        if "dtParam" in extra_features:            
            self.register("dt", torch.ones(1)*dt, lr=0.01)
        else:
            self.dt = dt
        

        dt_min = 0.01
        dt_max = 0.4

        if "dtLog" in extra_features:
            log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
            self.register("log_dt", log_dt, lr=0.01)

        if  "logAlpha" in extra_features:
            self.log_alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.log_alpha,torch.log(torch.tensor(self.alpha_lim[0])), torch.log(torch.tensor(self.alpha_lim[1])))

        elif "cont" in extra_features:
            if "A0_5" in extra_features:
                log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
                self.register("log_log_alpha", log_log_alpha, lr=0.01)
            elif "A0_5Const" in extra_features:
                self.log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
            else:
                self.log_log_alpha = nn.Parameter(torch.Tensor(self.hidden_size))
                nn.init.uniform_(self.log_log_alpha, torch.log(-torch.log(torch.tensor(self.alpha_lim[1]))/self.dt), torch.log(-torch.log(torch.tensor(self.alpha_lim[0]))/self.dt))
        
                        

        else:
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            self.log_log_alpha = torch.ones(self.hidden_size).to(device)

        
        
        if "imag" in extra_features:
            alpha_img =  math.pi * torch.ones(self.hidden_size).to(device) # torch.arange(self.hidden_size)
            self.register("alpha_img", alpha_img, lr=0.01)
 

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []
        if "imag"  in self.extra_features:
            eigenval = -torch.exp(self.log_log_alpha)+1j*self.alpha_img
        else:
            eigenval = -torch.exp(self.log_log_alpha) 

        if "dtLog"  in self.extra_features:
            self.dt = torch.exp(self.log_dt)
            
        
        if "logAlpha" in self.extra_features :
            alpha = torch.exp(self.log_alpha)
        elif "cont" in self.extra_features:
            alpha = torch.exp(self.dt*eigenval)
        else:
            alpha = self.alpha
        if "NoClamp" not in self.extra_features:
            alpha = torch.clamp(alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        if "B" in  self.extra_features:
            b = self.b
        else:
            b = (1 - alpha)

        # Loop over time axis
        for t in range(Wx.shape[1]):
            
            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            ut = alpha * (ut - reset) + b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            if "imag"  in self.extra_features:
                st = self.spike_fct(2*ut.real - self.threshold)
            else:
                st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class LIFfeatureDimLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features="_"
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        
        self.dim = 1
        if "dim2"  in extra_features:
            self.dim = 2
        self.b = nn.Parameter(torch.rand(self.hidden_size, self.dim)*0.5)

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.extra_features = extra_features
        device = torch.device("cuda")
        if "1-200_1-5"  in extra_features:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 200)]
        else:
            self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        dt = 1
        if "Dt1ms" in extra_features:
            dt = 0.001
        elif "Dt1" in extra_features:
            dt = 1
        if "dtParam" in extra_features:            
            self.register("dt", torch.ones(1)*dt, lr=0.01)
        else:
            self.dt = dt
        
        dt_min = 0.01
        dt_max = 0.4

        if "dtLog" in extra_features:
            log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
            self.register("log_dt", log_dt, lr=0.01)

        if  "logAlpha" in extra_features:
            self.log_alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.log_alpha,torch.log(torch.tensor(self.alpha_lim[0])), torch.log(torch.tensor(self.alpha_lim[1])))

        elif "cont" in extra_features:
            if "A0_5" in extra_features:
                log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
                self.register("log_log_alpha", log_log_alpha, lr=0.01)
            elif "A0_5Const" in extra_features:
                self.log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size)).to(device)
            else:
                self.log_log_alpha = nn.Parameter(torch.Tensor(self.hidden_size, self.dim))
                nn.init.uniform_(self.log_log_alpha, torch.log(-torch.log(torch.tensor(self.alpha_lim[1]))/self.dt), torch.log(-torch.log(torch.tensor(self.alpha_lim[0]))/self.dt))
        
                        

        else:
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            self.log_log_alpha = torch.ones(self.hidden_size).to(device)

        
        
        if "imag" in extra_features:
            alpha_img =  math.pi * torch.ones(self.hidden_size).to(device) # torch.arange(self.hidden_size)
            self.register("alpha_img", alpha_img, lr=0.01)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], self.dim).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []
        if "imag"  in self.extra_features:
            eigenval = -torch.exp(self.log_log_alpha)+1j*self.alpha_img
        else:
            eigenval = -torch.exp(self.log_log_alpha) 

        if "dtLog"  in self.extra_features:
            self.dt = torch.exp(self.log_dt)
            
        
        if "logAlpha" in self.extra_features :
            alpha = torch.exp(self.log_alpha)
        elif "cont" in self.extra_features:
            alpha = torch.exp(self.dt*eigenval)
        else:
            alpha = self.alpha

        if "B" in  self.extra_features:
            b = self.b
        else:
            b = (1 - alpha)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -  st.unsqueeze(-1).expand(-1,-1, self.dim)) + self.b * Wx[:, t, :].unsqueeze(-1).expand(-1,-1, self.dim)

            # Compute spikes with surrogate gradient
            if "imag"  in self.extra_features:
                st = self.spike_fct(2*ut.real - self.threshold)
            else:
                st = self.spike_fct(0.5*torch.sum(ut, dim=-1).real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)



class adLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._adlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _adlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else:
                reset = st

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * reset
            ut = alpha * (ut - reset) + (1 - alpha)* (Wx[:, t, :] - wt)

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class CadLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [0.36, 0.96]
        self.beta_lim = [0.96, 0.99]
        self.a_lim = [0.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        init.xavier_uniform_(self.W.weight)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features.get('rst_detach', False):
            self.rst_detach = True
        else:
            self.rst_detach = False

        self.gate = False
        if extra_features.get('gating', False):
            self.alpha_gate_fn, self.W_alpha_gate, self.alpha_gamma = gating_function(extra_features['gating'], False, self.hidden_size, self.hidden_size)
            self.beta_gate_fn, self.W_beta_gate, self.beta_gamma = gating_function(extra_features['gating'], False, self.hidden_size, self.hidden_size)            
            self.gate = True

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._cadlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _cadlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        if self.gate:
            # Loop over time axis
            gated_beta = self.beta_gate_fn(Wx, beta)
            gated_alpha = self.alpha_gate_fn(Wx, alpha)
            for t in range(Wx.shape[1]):

                # if self.rst_detach:
                #     reset = st.clone().detach()
                # else:
                #     reset = st

                # Compute potential (adLIF)
                wt = gated_beta[:,t,:] * wt + a * ut + b * st
                ut =  gated_alpha[:,t,:]* (ut - st) + (1 - gated_alpha[:,t,:])* (Wx[:, t, :] - wt)

                # Compute spikes with surrogate gradient
                st = self.spike_fct(ut - self.threshold)
                s.append(st)
        else:
            # Loop over time axis
            for t in range(Wx.shape[1]):

                # if self.rst_detach:
                #     reset = st.clone().detach()
                # else:
                #     reset = st

                # Compute potential (adLIF)
                wt = beta * wt + a * ut + b * st
                ut = alpha * (ut - st) + (1 - alpha)* (Wx[:, t, :] - wt)

                # Compute spikes with surrogate gradient
                st = self.spike_fct(ut - self.threshold)
                s.append(st)

        return torch.stack(s, dim=1)

class RSEadLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.dt = 1.0
        self.tau_u_lim = [5, 25]
        self.tau_w_lim = [60, 300]
        self.a_lim = [0.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.q = 120

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))
        self.theta = nn.Parameter(torch.Tensor(self.hidden_size))
        
        nn.init.uniform_(self.theta)
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
        nn.init.orthogonal_(self.V)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._seadlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def SLAYER(self, x, alpha=5, c=0.4):
        return c * alpha / (2 * torch.exp(x.abs() * alpha))

    def _seadlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        utm1 = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        ut = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        tau_u = self.tau_u_lim[0] + self.theta * (self.tau_u_lim[1]- self.tau_u_lim[0])
        tau_w = self.tau_w_lim[0] + self.theta * (self.tau_w_lim[1]- self.tau_w_lim[0])
        alpha = torch.exp(-self.dt / tau_u)
        beta = torch.exp(-self.dt / tau_w)
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute potential (adLIF)
            
            ut = alpha * utm1 + (1 - alpha)* (Wx[:, t, :] + F.linear(st, self.V, None) - wt)
            u_thr = ut - self.threshold
            st = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * self.SLAYER(u_thr).detach()
            ut = ut * (1 - st.detach())
            
            wt = beta * wt + (1 - beta)* (a * ut + b * st) * self.q


            s.append(st)

        return torch.stack(s, dim=1)

class CadLIFAblationLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [0.36, 0.96]
        self.beta_lim = [0.96, 0.99]
        self.a_lim = [0.0, 1.0]
        self.b_lim = [0.0, 2.0]        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        init.xavier_uniform_(self.W.weight)

        self.recurrent = extra_features.get('recurrent', False)
        if self.recurrent:
            self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.continuous = extra_features['continuous']
        self.reparam = extra_features['reparam']
        self.taylor = extra_features['taylor']
        self.s4_init = extra_features['s4_init']
        self.dt_train = extra_features['dt_train']
        self.dt_uniform = extra_features['dt_uniform']
        self.shared_alpha = extra_features['shared_alpha']       
        self.shared_a = extra_features['shared_a']   
        self.inp_b = extra_features['inp_b']  



        if self.reparam and self.dt_train: 
            self.dt = nn.Parameter(torch.full((self.hidden_size,), math.log(0.004)))
            # nn.init.uniform_(self.dt, math.log(0.001), math.log(0.5))
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, math.log(10.0), math.log(250.0))
            nn.init.uniform_(self.beta, math.log(2.5), math.log(10.0))  
        elif self.reparam: 
            self.dt = torch.tensor([math.log(0.004)], requires_grad=False)
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, math.log(10.0), math.log(250.0))
            nn.init.uniform_(self.beta, math.log(2.5), math.log(10.0))   
        elif self.dt_train:
            self.dt = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.dt, 0.001, 0.4)
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, 10.0, 250.0)
            nn.init.uniform_(self.beta, 2.5, 10.0)  
        elif self.continuous and self.dt_uniform:
            self.dt = torch.rand(self.hidden_size)*(0.4 - 0.001) + 0.001
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, 10.0, 250.0)
            nn.init.uniform_(self.beta, 2.5, 10.0)              
        elif self.continuous:
            self.dt = torch.tensor([math.log(0.004)], requires_grad=False)
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, 10.0, 250.0)
            nn.init.uniform_(self.beta, 2.5, 10.0)           
        else:
            self.dt = 0
            self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
            self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])

        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        if self.inp_b:
            self.inp_b = nn.Parameter(torch.rand(self.hidden_size))
        else:
            self.inp_b = None


        self.no_clamp = extra_features['no_clamp']

        self.threshold = 1.0

        if extra_features['half_reset']:
            self.reset_factor = 0.5
        else:
            self.reset_factor = 1.0

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._rf_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _rf_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)

        s = []       

        if self.reparam or self.s4_init:
            alpha = torch.exp(-torch.exp(self.alpha)*torch.exp(self.dt.to(device)))
            beta = torch.exp(-torch.exp(self.beta)*torch.exp(self.dt.to(device)))
        elif self.continuous or self.dt_train:
            dt = torch.clamp(self.dt.to(device), min = 0.0004, max=1.0)
            alpha = torch.exp(-self.alpha*dt)
            beta = torch.exp(-self.beta*dt)
        else:
            alpha = self.alpha
            beta = self.beta

        if not self.no_clamp:
            alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
            beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])


        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])
        B_u = (1- alpha)
        R_u = (1- alpha)

        if self.shared_alpha:
            beta = alpha
        if self.shared_a:
            b = a
            R_u = a
        if self.inp_b!=None:
            B_u = self.inp_b


        # Loop over time axis
        for t in range(Wx.shape[1]):


            wt = beta * wt + a * ut + b * st #+ B_w*Wx[:, t, :] 
            ut = alpha * (ut - st) + B_u* Wx[:, t, :] - R_u * wt

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class SiLIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.a_lim = [0.0, 1.0]
        self.b_lim = [0.0, 2.0]        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        init.xavier_uniform_(self.W.weight)

        self.recurrent = extra_features.get('recurrent', False)
        if self.recurrent:
            self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dt = nn.Parameter(torch.full((self.hidden_size,), math.log(0.004)))
        # nn.init.uniform_(self.dt, math.log(0.001), math.log(0.5))
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, math.log(10.0), math.log(250.0))
        nn.init.uniform_(self.beta, math.log(2.5), math.log(10.0))  


        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        self.threshold = 1.0
        self.reset_factor = 1.0

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._rf_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _rf_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)

        s = []       

        alpha = torch.exp(-torch.exp(self.alpha)*torch.exp(self.dt.to(device)))
        beta = torch.exp(-torch.exp(self.beta)*torch.exp(self.dt.to(device)))

        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])
        B_u = (1- alpha)
        R_u = (1- alpha)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            wt = beta * wt + a * ut + b * st #+ B_w*Wx[:, t, :] 
            ut = alpha * (ut - st) + B_u* Wx[:, t, :] - R_u * wt

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class DelaySiLIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.a_lim = [0.0, 1.0]
        self.b_lim = [0.0, 2.0]        
        self.spike_fct = SpikeFunctionBoxcar.apply

        max_delay = 300//10
        max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number

        self.W = Dcls1d(self.input_size, self.hidden_size, kernel_count=1, groups = 1, 
                                dilated_kernel_size = max_delay, bias=False, version='gauss')
        
        torch.nn.init.kaiming_uniform_(self.W.weight, nonlinearity='relu')

        torch.nn.init.uniform_(self.W.P, a = -max_delay//2, b = max_delay//2)
        self.W.clamp_parameters()

        self.sig_init = max_delay // 2
        torch.nn.init.constant_(self.W.SIG,  self.sig_init)
        self.W.SIG.requires_grad = False

        self.final_epoch = 100//4

        self.left_padding = max_delay-1
        self.right_padding = (max_delay-1) // 2

        # Trainable parameters
        # self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        # init.xavier_uniform_(self.W.weight)

        self.recurrent = extra_features.get('recurrent', False)
        if self.recurrent:
            self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dt = nn.Parameter(torch.full((self.hidden_size,), math.log(0.004)))
        # nn.init.uniform_(self.dt, math.log(0.001), math.log(0.5))
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, math.log(10.0), math.log(250.0))
        nn.init.uniform_(self.beta, math.log(2.5), math.log(10.0))  


        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        self.threshold = 1.0
        self.reset_factor = 1.0

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        x = x.permute(0,2,1)
        x = F.pad(x, (self.left_padding, self.right_padding), 'constant', 0)  # we use padding for the delays kernel

        # we use convolution of delay kernels
        Wx = self.W(x)

        # We permute again: (batch, neurons, time) => (batch, times, neurons) in order to be processed by batchnorm or Lif
        Wx = Wx.permute(0,2,1)


        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._rf_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _rf_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)

        s = []       

        alpha = torch.exp(-torch.exp(self.alpha)*torch.exp(self.dt.to(device)))
        beta = torch.exp(-torch.exp(self.beta)*torch.exp(self.dt.to(device)))

        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])
        B_u = (1- alpha)
        R_u = (1- alpha)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            wt = beta * wt + a * ut + b * st #+ B_w*Wx[:, t, :] 
            ut = alpha * (ut - st) + B_u* Wx[:, t, :] - R_u * wt

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)



class RAFAblationLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        self.recurrent = extra_features.get('recurrent', False)
        if self.recurrent:
            self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.continuous = extra_features['continuous']
        self.reparam = extra_features['reparam']
        self.taylor = extra_features['taylor']
        self.s4_init = extra_features['s4_init']
        self.dt_train = extra_features['dt_train']
        self.dt_uniform = extra_features['dt_uniform']
        
        if self.s4_init and self.dt_train:
            alpha_real = torch.log(0.5 * torch.ones(self.hidden_size))
            dt_min = extra_features["dt_min"]
            dt_max = extra_features["dt_max"]
            dt = torch.rand(self.hidden_size)*(
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
            alpha_im =  math.pi * torch.ones(self.hidden_size)
            self.register("alpha_real", alpha_real, lr=0.001)
            self.register("dt", dt, lr=0.001)
            self.register("alpha_im", alpha_im, lr=0.001)
        elif self.s4_init and self.dt_uniform:
            alpha_real = torch.log(0.5 * torch.ones(self.hidden_size))
            dt_min = extra_features["dt_min"]
            dt_max = extra_features["dt_max"]
            self.dt = torch.rand(self.hidden_size)*(
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
            alpha_im =  math.pi * torch.ones(self.hidden_size)
            self.register("alpha_real", alpha_real, lr=0.001)
            self.register("alpha_im", alpha_im, lr=0.001) 
        elif self.s4_init:
            alpha_real = torch.log(0.5 * torch.ones(self.hidden_size))
            self.dt = torch.tensor([math.log(0.004)], requires_grad=False)
            alpha_im =  math.pi * torch.ones(self.hidden_size)
            self.register("alpha_real", alpha_real, lr=0.001)
            self.register("alpha_im", alpha_im, lr=0.001) 
        elif self.reparam and self.dt_train: 
            self.dt = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.dt, math.log(0.001), math.log(0.5))
            self.alpha_real = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha_real, 0.0, math.log(10.0))
            self.alpha_im = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha_im, 5.0, 10.0)
        elif self.reparam: 
            self.dt = torch.tensor([math.log(0.004)], requires_grad=False)
            self.alpha_real = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha_real, 0.0, math.log(10.0))
            self.alpha_im = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha_im, 5.0, 10.0)
        elif self.dt_train:
            self.dt = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.dt, 0.001, 0.4)
            self.alpha_real = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha_real, 1.0, 10.0)
            self.alpha_im = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha_im, 5.0, 10.0)
        elif self.continuous:
            self.dt = torch.tensor([math.log(0.004)], requires_grad=False)
            self.alpha_real = nn.Parameter(torch.Tensor(self.hidden_size))
            self.alpha_im = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha_real, 1.0, 10.0)
            nn.init.uniform_(self.alpha_im, 5.0, 10.0)
        elif self.taylor:
            self.dt = 0.004
            self.alpha_real = nn.Parameter(torch.Tensor(self.hidden_size))
            self.alpha_im = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha_real, 1.0, 10.0)
            nn.init.uniform_(self.alpha_im, 5.0, 10.0)            
        else:
            self.dt = 0.004
            self.alpha_real = nn.Parameter(torch.Tensor(self.hidden_size))
            self.alpha_im = nn.Parameter(torch.Tensor(self.hidden_size))
            nn.init.uniform_(self.alpha_real, 0.7, 0.96)
            nn.init.uniform_(self.alpha_im, 0.02, 0.04)

        if extra_features['extra_b']:
            self.b = nn.Parameter(torch.rand(self.hidden_size))
        else:
            self.b = 1.0

        self.threshold = 1.0

        if extra_features['half_reset']:
            self.reset_factor = 0.5
        else:
            self.reset_factor = 1.0

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._rf_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _rf_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)

        s = []       

        if self.recurrent:
            V = self.V.weight.clone().fill_diagonal_(0)

        if self.reparam or self.s4_init:
            alpha = torch.exp((-torch.exp(self.alpha_real)+1j*self.alpha_im)*torch.exp(self.dt.to(device)))
        elif self.continuous or self.dt_train:
            dt = torch.clamp(self.dt.to(device), min = 0.0004, max=1.0)
            alpha_real = torch.clamp(self.alpha_real, min = 0.1)
            alpha = torch.exp((-alpha_real+1j*self.alpha_im)*dt)
        elif self.taylor:
            alpha_real = torch.clamp(self.alpha_real, min = 0.1)
            alpha = 1 + (-alpha_real+1j*self.alpha_im)*self.dt
        else:
            alpha_real = torch.clamp(self.alpha_real, min = 0.3, max=0.99)
            alpha = alpha_real + 1j*self.alpha_im

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.recurrent:
                I = Wx[:, t, :] + torch.matmul(st, V)
            else:
                I = Wx[:, t, :]
            # Compute membrane potential (LIF)

            ut = alpha*(ut - self.reset_factor*st) + self.b * I

            # Compute spikes with surrogate gradient
            st = self.spike_fct((1/self.reset_factor)*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class ResonateFireLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        self.recurrent = extra_features.get('recurrent', False)
        if self.recurrent:
            self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dt = 0.004

        self.alpha_real = nn.Parameter(torch.Tensor(self.hidden_size))
        self.alpha_im = nn.Parameter(torch.Tensor(self.hidden_size))
        self.threshold = 1.0

        nn.init.uniform_(self.alpha_real, -10.0, -1.0)
        nn.init.uniform_(self.alpha_im, 5.0, 10.0)

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._rf_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _rf_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)

        s = []

        alpha_real = torch.clamp(self.alpha_real, max = -0.1)

        if self.recurrent:
            V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.recurrent:
                I = Wx[:, t, :] + torch.matmul(st, V)
            else:
                I = Wx[:, t, :]
            # Compute membrane potential (LIF)
            ut = ut + self.dt*((alpha_real + 1j*self.alpha_im)*ut +  I) - st

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class BRFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        self.recurrent = extra_features.get('recurrent', False)
        if self.recurrent:
            self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dt = 0.004

        self.gamma = 0.9
        self.alpha_real_off = nn.Parameter(torch.Tensor(self.hidden_size))
        self.alpha_im = nn.Parameter(torch.Tensor(self.hidden_size))
        self.threshold = 1.0

        nn.init.uniform_(self.alpha_real_off, 2.0, 3.0)
        nn.init.uniform_(self.alpha_im, 5.0, 10.0)

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._rf_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _rf_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        qt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)

        s = []

        p_w = (-1 + torch.sqrt(1-torch.square(self.dt*self.alpha_im)))/self.dt

        if self.recurrent:
            V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            b = p_w - self.alpha_real_off - qt
            if self.recurrent:
                I = Wx[:, t, :] + torch.matmul(st, V)
            else:
                I = Wx[:, t, :]
            ut = ut + self.dt*((b + 1j*self.alpha_im)*ut + I)

            theta = self.threshold + qt

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut.real - theta)
            s.append(st)

            qt = self.gamma*qt + st

        return torch.stack(s, dim=1)

class adLIFclampLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        self.alpha.data.clamp_(self.alpha_lim[0], self.alpha_lim[1])
        self.beta.data.clamp_(self.beta_lim[0], self.beta_lim[1])
        self.a.data.clamp_(self.a_lim[0], self.a_lim[1])
        self.b.data.clamp_(self.b_lim[0], self.b_lim[1])

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._adlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _adlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else:
                reset = st

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * reset
            ut = alpha * (ut - reset) + (1 - alpha)* (Wx[:, t, :] - wt)

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)
    


class adLIFnoClampLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._adlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _adlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = self.alpha
        beta = self.beta 
        a = self.a 
        b = self.b 

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else:
                reset = st

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * reset
            ut = alpha * (ut - reset) + (1 - alpha)* (Wx[:, t, :] - wt)

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)
    

class LIFcomplexLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        if extra_features['superspike']:
            self.spike_fct = SpikeFunctionSuperSpike.apply
        elif extra_features['slayer']:
            self.spike_fct = SpikeFunctionSLAYER.apply
        else:
            self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        if extra_features['weight_norm']:
            self.W = nn.utils.weight_norm(self.W)

        self.LRU_re_init = extra_features.get('LRU_re_init', False)
        self.LRU_img_init = extra_features.get('LRU_img_init', False)
        self.LRU_no_dt = extra_features.get('LRU_no_dt', False)
        self.r_max = extra_features.get('LRU_r_max', 1.0)
        self.r_min = extra_features.get('LRU_r_min', 0)
        self.max_phase = extra_features.get('LRU_max_phase', 6.28)

        alpha = extra_features.get('alpha', 0.5)
        alpha_range = extra_features.get('alpha_range', False)
        alpha_range_max = extra_features.get('alpha_range_max', False)
        alpha_range_min = extra_features.get('alpha_range_min', False)
        alpha_rand = extra_features.get('alpha_rand', False)

        if extra_features['xavier_init']:
            init.xavier_uniform_(self.W.weight)

        if not self.LRU_re_init :
            log_log_alpha = torch.log(alpha * torch.ones(self.hidden_size))
            
        else:
            u1 = torch.rand(self.hidden_size)
            log_log_alpha = torch.log(-alpha*torch.log(u1*(self.r_max**2-self.r_min**2)+self.r_min**2))
        if alpha_range:
            if alpha_rand == "Rand":
                log_log_alpha = torch.rand(self.hidden_size)*(math.log(alpha_range_max) - math.log(alpha_range_min)) + math.log(alpha_range_min)
            elif alpha_rand == "RandN":
                log_log_alpha = torch.randn(self.hidden_size)*(math.log(alpha_range_max) - math.log(alpha_range_min)) + math.log(alpha_range_min)

        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        dt_max = extra_features["dt_max"]
        alpha_img_pi_rat = extra_features.get('alpha_img_pi_rat', 1) 
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        if not self.LRU_img_init :
            alpha_img =  math.pi * alpha_img_pi_rat * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)
        else: 
            u2 = torch.rand(self.hidden_size)
            alpha_img = torch.log(self.max_phase*u2)
            
        
        self.register("log_dt", log_dt, lr=0.001)
        self.register("log_log_alpha", log_log_alpha, lr=0.001)
        self.register("alpha_img", alpha_img, lr=0.001)

        self.LRU_b = extra_features.get('LRU_b', False)
        if not extra_features['no_b']:
            self.b = 1 #placeholder
            if not self.LRU_b:
                self.b = nn.Parameter(torch.rand(self.hidden_size))
            else:
                self.b_re = nn.Parameter(torch.randn(self.hidden_size)/torch.sqrt(2.0 *torch.tensor( self.hidden_size, dtype=torch.float32)))
                self.b_img = nn.Parameter(torch.randn(self.hidden_size)/torch.sqrt(2.0 *torch.tensor( self.hidden_size, dtype=torch.float32)))
        else:
            self.b = None

        self.clamp_alpha = extra_features['clamp_alpha']
        self.alpha_min = extra_features['alpha_min']
        self.alpha_max = extra_features['alpha_max']
        if self.alpha_min >= self.alpha_max:
            self.alpha_min = self.alpha_max - 0.1

        if extra_features['no_reset']:
            self.reset_factor = 0
        else:
            if extra_features['complex_reset']:
                reset_factor = torch.tensor([0.5 - 0.5j], dtype=torch.cfloat)
                self.register_buffer('reset_factor', reset_factor)
            elif extra_features['half_reset']:
                self.reset_factor = 0.5
            else:
                self.reset_factor = 1.0

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.zero_init = extra_features['zero_init']

        self.LRU_norm =  extra_features.get('LRU_norm', False)
        if self.LRU_norm:
            alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
            gamma_log = torch.log(torch.sqrt(1-torch.abs(alpha)**2))
            self.register("gamma_log", gamma_log, lr=0.001)

        self.gate = False
        if extra_features['gating']:
            self.gate_fn, self.W_gate, self.gamma = gating_function(extra_features['gating'], True, self.hidden_size, self.hidden_size)
            self.gate = True

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        if self.zero_init:
            ut = torch.zeros(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
            st = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)
        else:   
            ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
            st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        if self.LRU_img_init :
            alpha_img = torch.exp(self.alpha_img)
        else: 
            alpha_img = self.alpha_img

        alpha_cont = -torch.exp(self.log_log_alpha)+1j*alpha_img

        if self.gate:
            alpha_for_gate = torch.exp(self.log_log_alpha)-1j*alpha_img
        if not self.LRU_no_dt:
            alpha_cont *=torch.exp(self.log_dt)
        alpha = torch.exp(alpha_cont)
        
        if self.clamp_alpha:
            real_part = alpha.real
            imag_part = alpha.imag

            # Clamp the real and imaginary parts
            clamped_real = torch.clamp(real_part, min=self.alpha_min, max=self.alpha_max)
            clamped_imag = torch.clamp(imag_part, min=0.0)

            # Recombine the clamped real and imaginary parts
            alpha = clamped_real + 1j * clamped_imag            
        
        if self.b!=None:
            b = self.b 
            if self.LRU_b:
                b = self.b_re + 1j * self.b_img

            if self.LRU_norm:
                b = b * torch.exp(self.gamma_log)
            

            # Loop over time axis
            if self.gate:
                gated_alpha = self.gate_fn(Wx, alpha_for_gate)
                for t in range(Wx.shape[1]):

                    if self.rst_detach:
                        reset = st.clone().detach()
                    else: 
                        reset = st

                    # Compute membrane potential (LIF)
                    ut =  gated_alpha[:,t,:]* (ut - self.reset_factor*reset) + b * Wx[:, t, :]

                    # Compute spikes with surrogate gradient
                    st = self.spike_fct(2*ut.real - self.threshold)
                    s.append(st)                
            else:
                for t in range(Wx.shape[1]):

                    if self.rst_detach:
                        reset = st.clone().detach()
                    else: 
                        reset = st

                    # Compute membrane potential (LIF)
                    ut = alpha * (ut - self.reset_factor*reset) + b * Wx[:, t, :]

                    # Compute spikes with surrogate gradient
                    st = self.spike_fct(2*ut.real - self.threshold)
                    s.append(st)
        else:
            # Loop over time axis
            for t in range(Wx.shape[1]):

                if self.rst_detach:
                    reset = st.clone().detach()
                else: 
                    reset = st

                # Compute membrane potential (LIF)
                ut = alpha * (ut - self.reset_factor*reset) + (1-alpha.real) * Wx[:, t, :]

                # Compute spikes with surrogate gradient
                st = self.spike_fct(2*ut.real - self.threshold)
                s.append(st)            

        return torch.stack(s, dim=1)

class RingInitLIFcomplexLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        if extra_features['superspike']:
            self.spike_fct = SpikeFunctionSuperSpike.apply
        elif extra_features['slayer']:
            self.spike_fct = SpikeFunctionSLAYER.apply
        else:
            self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        if extra_features['weight_norm']:
            self.W = nn.utils.weight_norm(self.W)


        r_max = extra_features['r_max']
        r_min = extra_features['r_min']

        u1 = torch.rand(self.hidden_size) 
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5*torch.log(u1*(r_max**2-r_min**2) + r_min**2)) 
        theta_log = torch.log(extra_features['max_phase']*u2)

        self.gamma_norm = extra_features['gamma_norm']
        if self.gamma_norm:
            alpha = torch.exp((-torch.exp(nu_log)+1j*theta_log))
            gamma_log = torch.log(torch.sqrt(1-torch.abs(alpha)**2))
            self.register("gamma_log", gamma_log, lr=0.001)

        self.register("nu_log", nu_log, lr=0.001)
        self.register("theta_log", theta_log, lr=0.001)

        if not extra_features['no_b']:
            self.b = nn.Parameter(torch.rand(self.hidden_size))
        else:
            self.b = None

        self.clamp_alpha = extra_features['clamp_alpha']
        self.alpha_min = extra_features['alpha_min']
        self.alpha_max = extra_features['alpha_max']
        if self.alpha_min >= self.alpha_max:
            self.alpha_min = self.alpha_max - 0.1

        if extra_features['no_reset']:
            self.reset_factor = 0
        else:
            if extra_features['complex_reset']:
                reset_factor = torch.tensor([0.5 - 0.5j], dtype=torch.cfloat)
                self.register_buffer('reset_factor', reset_factor)
            elif extra_features['half_reset']:
                self.reset_factor = 0.5
            else:
                self.reset_factor = 1.0

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.zero_init = extra_features['zero_init']



    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        if self.zero_init:
            ut = torch.zeros(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
            st = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)
        else:   
            ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
            st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.nu_log)+1j*self.theta_log))
        
        if self.clamp_alpha:
            real_part = alpha.real
            imag_part = alpha.imag

            # Clamp the real and imaginary parts
            clamped_real = torch.clamp(real_part, min=self.alpha_min, max=self.alpha_max)
            clamped_imag = torch.clamp(imag_part, min=0.0)

            # Recombine the clamped real and imaginary parts
            alpha = clamped_real + 1j * clamped_imag            
        
        # Loop over time axis
        if self.gamma_norm:
            gamma = torch.exp(self.gamma_log)
            for t in range(Wx.shape[1]):

                if self.rst_detach:
                    reset = st.clone().detach()
                else: 
                    reset = st

                # Compute membrane potential (LIF)
                ut = alpha * (ut - self.reset_factor*reset) + gamma*self.b * Wx[:, t, :]

                # Compute spikes with surrogate gradient
                st = self.spike_fct(2*ut.real - self.threshold)
                s.append(st)
        else:
            for t in range(Wx.shape[1]):

                if self.rst_detach:
                    reset = st.clone().detach()
                else: 
                    reset = st

                # Compute membrane potential (LIF)
                ut = alpha * (ut - self.reset_factor*reset) + self.b * Wx[:, t, :]

                # Compute spikes with surrogate gradient
                st = self.spike_fct(2*ut.real - self.threshold)
                s.append(st)
        
        return torch.stack(s, dim=1)

class LIFcomplexBroadLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.expansion = extra_features['exp_factor']
        self.state_size = self.expansion*self.hidden_size
        
        if extra_features['superspike']:
            self.spike_fct = SpikeFunctionSuperSpike.apply
        elif extra_features['slayer']:
            self.spike_fct = SpikeFunctionSLAYER.apply
        elif extra_features['relu_spike']:
            self.spike_fct = F.relu
        else:
            self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        if extra_features['xavier_init']:
            init.xavier_uniform_(self.W.weight)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size, self.expansion))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        log_dt = torch.rand(self.hidden_size, self.expansion)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        # alpha_img =  math.pi * torch.ones(self.hidden_size, self.expansion) # torch.arange(self.hidden_size)
        alpha_img = math.pi * repeat(torch.arange(self.expansion), 'n -> h n', h=self.hidden_size)
        
        self.register("log_log_alpha", log_log_alpha, lr=0.001)
        self.register("log_dt", log_dt, lr=0.001)
        self.register("alpha_img", alpha_img, lr=0.001)


        if not extra_features['no_b']:
            self.b = nn.Parameter(torch.rand(self.hidden_size, self.expansion))
        else:
            self.b = None

        if extra_features['c_sum']:
            C = torch.randn(self.hidden_size, self.expansion, dtype=torch.cfloat)
            self.c = nn.Parameter(torch.view_as_real(C))
        else:
            self.c = None        


        self.clamp_alpha = extra_features['clamp_alpha']
        self.alpha_min = extra_features['alpha_min']
        self.alpha_max = extra_features['alpha_max']
        if self.alpha_min >= self.alpha_max:
            self.alpha_min = self.alpha_max - 0.1

        if extra_features['no_reset']:
            self.reset_factor = 0
        else:
            if extra_features['complex_reset']:
                reset_factor = torch.tensor([0.5 - 0.5j], dtype=torch.cfloat)
                self.register_buffer('reset_factor', reset_factor)
            elif extra_features['half_reset']:
                self.reset_factor = 0.5
            else:
                self.reset_factor = 1.0

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)



    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], self.expansion, dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        
        if self.clamp_alpha:
            real_part = alpha.real
            imag_part = alpha.imag

            # Clamp the real and imaginary parts
            clamped_real = torch.clamp(real_part, min=self.alpha_min, max=self.alpha_max)
            clamped_imag = torch.clamp(imag_part, min=0.0)

            # Recombine the clamped real and imaginary parts
            alpha = clamped_real + 1j * clamped_imag            
        
        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            ut = alpha * (ut - self.reset_factor*reset.unsqueeze(-1).expand(-1,-1, self.expansion)) + self.b * Wx[:, t, :].unsqueeze(-1).expand(-1,-1, self.expansion)

            # Compute spikes with surrogate gradient
            if self.c!=None:
                C = torch.view_as_complex(self.c)
                self.spike_fct(torch.einsum('hn, bhn -> bh', C, ut).real- self.threshold)
            else:
                st = self.spike_fct(2*torch.sum(ut.real, dim=-1) - self.threshold)
            s.append(st)          

        return torch.stack(s, dim=1)

class LIFrealcomplexLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        if extra_features['half_reset']:
            self.reset_factor = 0.5
        else:
            self.reset_factor = 1.0

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.clamp_alpha = extra_features['clamp_alpha']
        self.alpha_min = extra_features['alpha_min']
        self.alpha_max = extra_features['alpha_max']
        if self.alpha_min >= self.alpha_max:
            self.alpha_min = self.alpha_max - 0.1

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        dt = torch.exp(self.log_dt)
        real_part = -torch.exp(self.log_log_alpha) * dt
        imaginary_part = self.alpha_img * dt

        # Compute the separate components
        exp_real = torch.exp(real_part)
        cos_imag = torch.cos(imaginary_part)
        sin_imag = torch.sin(imaginary_part)

        if self.clamp_alpha:
            alpha_real = torch.clamp(exp_real * cos_imag, min=self.alpha_min, max=self.alpha_max)
            alpha_imag = torch.clamp(exp_real * sin_imag, min=0.0)
        else:
            alpha_real = exp_real * cos_imag
            alpha_imag = exp_real * sin_imag
        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            wt = alpha_real * wt + alpha_imag * (ut + self.reset_factor*reset)
            ut = alpha_real * (ut - self.reset_factor*reset) - alpha_imag*wt + self.b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class ReLULIFcomplexLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        if extra_features['half_reset']:
            self.reset_factor = 0.5
        else:
            self.reset_factor = 1.0

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.shifted_relu = extra_features['shifted_relu']

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * ut + self.b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            if self.shifted_relu:
                st = F.relu(2*ut.real - self.threshold)
            else:
                st = F.relu(2*ut.real)
            s.append(st)

        return torch.stack(s, dim=1)

class RLIFcomplexLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = extra_features["dt_min"]
        dt_max = extra_features["dt_max"]
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        if extra_features['half_reset']:
            self.reset_factor = 0.5
        else:
            self.reset_factor = 1.0

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False


        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        V = self.V.weight.clone().fill_diagonal_(0)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (LIF)
            ut = alpha * (ut - self.reset_factor*reset) + self.b * (Wx[:, t, :] + torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class RLIFcomplex1MinAlphaLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = 0.01
        dt_max = 0.4
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        V = self.V.weight.clone().fill_diagonal_(0)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -st) + self.b * (Wx[:, t, :]) + (1-alpha)*(torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)
    
class RLIFcomplex1MinAlphaNoBLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = 0.01
        dt_max = 0.4
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        V = self.V.weight.clone().fill_diagonal_(0)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -st) + (1-alpha) * (Wx[:, t, :] + torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class LIFcomplexDiscrLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.D = nn.Parameter(torch.randn(self.hidden_size))
        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = 0.01
        dt_max = 0.4
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)

        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.hidden_size, 2*self.hidden_size, kernel_size=1),
            nn.GLU(dim=-2),
        )

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])
        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*torch.exp(self.log_dt))
        b_disc = self.b * (alpha-1.0)/(-torch.exp(self.log_log_alpha)+1j*self.alpha_img)
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -st) + b_disc * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class LIFcomplex_gatedBLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        dt_min = 0.01 #0.001 #0.01
        dt_max = 0.4 #0.1 #0.4
        #log_dt = torch.rand(self.hidden_size)*(
        ##    math.log(dt_max) - math.log(dt_min)
        #) + math.log(dt_min)

        dt_init = "random"
        dt_scale = 1.0
        dt_rank = "auto"
        self.dt_rank = math.ceil(self.hidden_size / 1) if dt_rank == "auto" else dt_rank

        #self.dt_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.x_proj = nn.Linear(
            self.hidden_size,1, bias=False
        )
        dt_min = 0.01
        dt_max = 0.4
        log_dt = torch.rand(self.hidden_size)*(
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.register("log_dt", log_dt, lr=0.01)
        

        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)



        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        #self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)
        self.sigm = nn.Sigmoid()
        self.normB = nn.BatchNorm1d(1, momentum=0.05)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []


        b = self.x_proj(rearrange(Wx, "b l d -> (b l) d"))  # (bl dt_rank)
        b = rearrange(b, "(b l) d -> b d l", l=Wx.shape[1])
        '''
        min_d = b.min(dim=2, keepdim=True)[0]
        max_d = b.max(dim=2, keepdim=True)[0]
        range_d = max_d - min_d
        epsilon = 1e-8
        range_d = range_d + epsilon
        b = (b - min_d) / range_d
        '''

        #b = self.normB(b)
        #b = self.sigm(b)
        dt = torch.exp(self.log_dt)
        b = torch.transpose((dt * torch.transpose(b, 1,2)) , 1,2)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])

        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img)*dt)
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut -st) + b[:,:,t] * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class LIFcomplex_gatedDtLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        log_log_alpha = torch.log(0.5 * torch.ones(self.hidden_size))
        #self.log_log_alpha_lim = [math.log(1 / 200), math.log(1 / 5)]
        self.dt_min = 0.01 #0.001 #0.01
        self.dt_max = 0.4 #0.1 #0.4
        #log_dt = torch.rand(self.hidden_size)*(
        ##    math.log(dt_max) - math.log(dt_min)
        #) + math.log(dt_min)

        dt_init = "random"
        dt_scale = 1.0
        dt_rank = "auto"
        self.dt_rank = math.ceil(self.hidden_size / 1) if dt_rank == "auto" else dt_rank

        #self.dt_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.x_proj = nn.Linear(
            self.hidden_size, self.dt_rank, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.hidden_size, bias=True)

        dt_init_std = (self.dt_rank)**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        
        dt = torch.exp(
            torch.rand(self.hidden_size) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=1e-4)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        #nn.init.uniform_(log_log_alpha, self.log_log_alpha_lim[0], self.log_log_alpha_lim[1])
        alpha_img =  math.pi * torch.ones(self.hidden_size) # torch.arange(self.hidden_size)



        self.register("log_log_alpha", log_log_alpha, lr=0.01)
        #self.register("log_dt", log_dt, lr=0.01)
        self.register("alpha_img", alpha_img, lr=0.01)

        self.b = nn.Parameter(torch.rand(self.hidden_size))

        # Initialize normalinzation
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        #Wx = self.output_linear(Wx.reshape(Wx.shape[0], Wx.shape[2], Wx.shape[1])).reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2], dtype=torch.cfloat).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []


        dt1 = self.x_proj(rearrange(Wx, "b l d -> (b l) d"))  # (bl dt_rank)
        bias = repeat(
            self.dt_proj.bias,
            "n -> n d",
            d=Wx.shape[0]*Wx.shape[1],
        )
        dt = F.softplus( self.dt_proj.weight @ dt1.t() + bias)
        dt = rearrange(dt, "d (b l) -> b d l", l=Wx.shape[1])
        dt = torch.clamp(dt, min = self.dt_min, max = self.dt_max)

        

        #dt = torch.sigmoid(dt)

        # Scale and shift to get values between 0.001 and 0.4
        #dt = 0.001 + dt * (0.4 - 0.001)

        # Bound values of the neuron parameters to plausible ranges
        #log_log__alpha = torch.clamp(self.log_log_alpha, min=self.log_log_alpha_lim[0], max=self.log_log_alpha_lim[1])

        alpha = torch.exp((-torch.exp(self.log_log_alpha)+1j*self.alpha_img).unsqueeze(0).unsqueeze(2).repeat(Wx.shape[0], 1, Wx.shape[1])*dt) # B H L 
        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha[:,:,t] * (ut -st) + self.b * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(2*ut.real - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class RLIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.orthogonal_(self.V.weight)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._rlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _rlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Set diagonal elements of recurrent matrix to zero
        V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):
            
            if self.rst_detach:
                reset = st.clone().detach()
            else: 
                reset = st

            # Compute membrane potential (RLIF)
            ut = alpha * (ut - reset) + (1 - alpha) * (Wx[:, t, :] + torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class RadLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RadLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    threshold : float
        Value of spiking threshold (fixed)
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
        nn.init.orthogonal_(self.V.weight)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        if extra_features['rst_detach']:
            self.rst_detach = True
        else:
            self.rst_detach = False

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._radlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _radlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Set diagonal elements of recurrent matrix to zero
        V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            if self.rst_detach:
                reset = st.clone().detach()
            else:
                reset = st

            # Compute potential (RadLIF)
            wt = beta * wt + a * ut + b * reset
            ut = alpha * (ut - reset) + (1 - alpha) * (
                Wx[:, t, :] + torch.matmul(st, V) - wt
            )

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class ReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        if extra_features.get('layernorm_readout', False):
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True
        else:
            self.normalize = False
            if normalization == "batchnorm":
                self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
                self.normalize = True
            elif normalization == "layernorm":
                self.norm = nn.LayerNorm(self.hidden_size)
                self.normalize = True


        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.time_offset = extra_features.get('time_offset', 0)

    def forward(self, x):

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)

        return out

    def _readout_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Loop over time axis
        for t in range(self.time_offset, Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)

        return out


class DelayReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]

        # Trainable parameters
        max_delay = 300//10
        max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number

        self.W = Dcls1d(self.input_size, self.hidden_size, kernel_count=1, groups = 1, 
                                dilated_kernel_size = max_delay, bias=False, version='gauss')
        
        torch.nn.init.kaiming_uniform_(self.W.weight, nonlinearity='relu')

        torch.nn.init.uniform_(self.W.P, a = -max_delay//2, b = max_delay//2)
        self.W.clamp_parameters()

        self.sig_init = max_delay // 2
        torch.nn.init.constant_(self.W.SIG,  self.sig_init)
        self.W.SIG.requires_grad = False

        self.final_epoch = 100//4

        self.left_padding = max_delay-1
        self.right_padding = (max_delay-1) // 2

        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        if extra_features.get('layernorm_readout', False):
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True
        else:
            self.normalize = False
            if normalization == "batchnorm":
                self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
                self.normalize = True
            elif normalization == "layernorm":
                self.norm = nn.LayerNorm(self.hidden_size)
                self.normalize = True


        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.time_offset = extra_features.get('time_offset', 0)

    def forward(self, x):

        x = x.permute(0,2,1)
        x = F.pad(x, (self.left_padding, self.right_padding), 'constant', 0)  # we use padding for the delays kernel

        # we use convolution of delay kernels
        Wx = self.W(x)

        # We permute again: (batch, neurons, time) => (time, batch, neurons) in order to be processed by batchnorm or Lif
        Wx = Wx.permute(0,2,1)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)

        return out

    def _readout_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Loop over time axis
        for t in range(self.time_offset, Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)

        return out


class JaxReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        extra_features=None
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias

        self.ro_int = extra_features['ro_int']

        self.alpha = np.exp(-0.001 / extra_features['tau_m'])

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)


        # Initialize normalization
        if extra_features['layernorm_readout']:
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True
        else:
            self.normalize = False
            if normalization == "batchnorm":
                self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
                self.normalize = True
            elif normalization == "layernorm":
                self.norm = nn.LayerNorm(self.hidden_size)
                self.normalize = True


        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

        self.time_offset = extra_features['time_offset']

    def forward(self, x):

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)

        return out

    def _readout_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = np.clip(self.alpha, 0.5,1.0)

        u = []

        for t in range(self.time_offset):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]

        ut = ut.detach()
        # Loop over time axis
        for t in range(self.time_offset, Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            u.append(ut)

        out = u[::-self.ro_int]
        return torch.stack(out, dim=1)

class SEReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    use_bias : bool
        If True, additional trainable bias is used with feedforward weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha = np.exp(-1.0 / 15.0)

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)

        return out

    def _readout_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = self.alpha

        # Loop over time axis
        for t in range(10, Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)

        return out