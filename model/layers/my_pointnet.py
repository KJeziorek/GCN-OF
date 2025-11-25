import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Sequential, Linear, BatchNorm1d

from model.layers.quantisation.observer import Observer, FakeQuantize, quantize_tensor, dequantize_tensor

import numpy as np

import torch
import torch.nn.functional as F


class MyPointNetConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = False,
        num_bits: int = 8,
        first_layer: bool = False,
        input_bits: int = 8
    ):
        
        super(MyPointNetConv, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.num_bits = num_bits
        self.first_layer = first_layer

        # Number of bits for quantization scales
        self.num_bits_obs = 32

        # Define layers
        self.linear = Linear(input_dim, output_dim, bias=bias)
        self.norm = BatchNorm1d(output_dim)

        self.use_relu = True
        self.global_nn = None
        self.add_self_loops = True
        self.use_observer_input: bool = True

        self.reset_parameters()

        # Modes for calibration and quantization
        self.register_buffer('calib_mode', torch.tensor(False, requires_grad=False))
        self.register_buffer('quantize_mode', torch.tensor(False, requires_grad=False))

        # Initialize quantization observers
        self.observer_input = Observer(num_bits=num_bits)
        self.observer_weight = Observer(num_bits=num_bits)
        self.observer_output = Observer(num_bits=num_bits)

        # Register buffers for quantization parameters
        self.register_buffer('m', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_in', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_w', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_out', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_m', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('num_bits_model', torch.tensor(num_bits, requires_grad=False))
        self.register_buffer('num_bits_scale', torch.tensor(self.num_bits_obs, requires_grad=False))

    def reset_parameters(self):
        '''
            Reset parameters of the model
        '''
        self.linear.reset_parameters()
        self.norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        
        '''
            Standard forward method of a PointNetConv layer
        '''

        out = self.message(x, pos, edge_index)
        
        # # Apply activation function
        # if self.use_relu:
        #     if not self.quantize_mode:
        #         out = F.relu(out)
        #     else:
        #         # In quantize mode, simulate quantized ReLU
        #         out[out < self.observer_output.zero_point] = self.observer_output.zero_point

        return out

    def message(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        '''
            Custom message function for PointNetConv.
            We select the message function based on the current mode (calibration, quantize, or float)
        '''
        if self.calib_mode.item() and not self.quantize_mode.item():
            return self.message_calib(x, pos, edge_index)
        elif self.quantize_mode.item():
            return self.message_quant(x, pos, edge_index)
        elif not self.calib_mode.item() and not self.quantize_mode.item():
            return self.message_float(x, pos, edge_index)
        else:
            raise ValueError('Invalid mode')

    def message_float(self, 
                      x: torch.Tensor, 
                      pos: torch.Tensor, 
                      edge_index: torch.Tensor) -> torch.Tensor:

        '''Calculate message for PointNetConv layer.'''
        pos_i = pos[edge_index[:, 0]]
        pos_j = pos[edge_index[:, 1]]
        x_j = x[edge_index[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)

        '''Propagate message through linear layer.'''
        msg = self.linear(msg)
        # msg = self.norm(msg)

        '''Update graph features.'''
        unique_positions, indices = torch.unique(edge_index[:,0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=msg.dtype, device=x.device)
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax", include_self=False)

        return pooled_features

    def message_calib(self, 
                      x: torch.Tensor, 
                      pos: torch.Tensor, 
                      edge_index: torch.Tensor) -> torch.Tensor:
        
        # gather messages
        pos_i = pos[edge_index[:, 0]]
        pos_j = pos[edge_index[:, 1]]
        x_j = x[edge_index[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)

        # fake-quantize inputs
        if self.use_observer_input:
            if self.training:
                self.observer_input.update(msg)
            msg = FakeQuantize.apply(msg, self.observer_input)

        # if training, run a dummy through BN to update its stats
        if self.training:
            dummy = self.linear(msg)
            _ = self.norm(dummy)

        # fuse BN into linear weights/bias
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        std = torch.sqrt(running_var + self.norm.eps)
        W_fused, b_fused = self.merge_norm(running_mean, std)

        # fake-quantize fused weights
        if self.training:
            self.observer_weight.update(W_fused)
        W_q = FakeQuantize.apply(W_fused, self.observer_weight)

        # apply quantized linear with fused bias
        msg = F.linear(msg, W_q, b_fused)

        '''Update output observer and calculate output.'''
        '''We calibrate based on the output of the Linear and also for diff POS for next layer'''
        if self.training:
            self.observer_output.update(msg)
            self.observer_output.update(pos_j-pos_i)
        msg = FakeQuantize.apply(msg, self.observer_output)

        '''Update graph features.'''
        unique_positions, indices = torch.unique(edge_index[:,0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=x.dtype, device=x.device)
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax", include_self=False)

        return pooled_features
    
    def message_quant(self, 
                      x: torch.Tensor, 
                      pos: torch.Tensor, 
                      edge_index: torch.Tensor) -> torch.Tensor:

        '''Quantize input features'''
        if self.first_layer:
            '''We need to quantize both features and POS for the first layer.'''
            pos_i = pos[edge_index[:, 0]]
            pos_j = pos[edge_index[:, 1]]
            x_j = x[edge_index[:, 1]]
            msg = torch.cat((x_j, pos_j - pos_i), dim=1)
            msg = self.observer_input.quantize_tensor(msg)
        else:
            '''For other layers, we only quantize POS, because features are already quantized.'''
            pos_i = pos[edge_index[:, 0]]
            pos_j = pos[edge_index[:, 1]]
            pos = self.observer_input.quantize_tensor(pos_j - pos_i)
            msg = torch.cat((x[edge_index[:, 1]], pos), dim=1)

        msg = msg - self.observer_input.zero_point
        msg = self.qlinear(msg)
        msg = (msg * self.m).round() + self.observer_output.zero_point
        msg = torch.clamp(msg, 0, 2**self.num_bits - 1)

        '''Update graph features.'''
        unique_positions, indices = torch.unique(edge_index[:,0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=x.dtype, device=x.device)
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax", include_self=False)

        return pooled_features

    def merge_norm(self,
                   mean: torch.Tensor,
                   std: torch.Tensor):
        '''
            Merge batch normalization parameters with linear weights.
        '''
        if self.norm.affine:
            gamma = self.norm.weight
            beta = self.norm.bias
        else:
            gamma = torch.ones_like(std)
            beta = torch.zeros_like(std)

        W = self.linear.weight

        if self.bias:
            b = self.linear.bias
        else:
            b = torch.zeros(self.output_dim, device=W.device)
        
        W_new = (gamma / std).unsqueeze(1) * W
        b_new = (gamma / std) * (b - mean) + beta
        
        return W_new, b_new
    
    def calibrate(self):
        self.calib_mode.fill_(True)

    def quantize(self,
               observer_input: Observer = None,
               observer_output: Observer = None):
        '''
            Quantize model - quantize weights/bias and calculate scales
        '''
        self.quantize_mode.fill_(True)

        if observer_input is not None:
            self.observer_input = observer_input
        if observer_output is not None:
            self.observer_output = observer_output

        # Quantize scales for input, weight, and output
        self.qscale_in.copy_( (2**self.num_bits_obs * self.observer_input.scale).round() )
        self.observer_input.scale = self.qscale_in / (2 ** self.num_bits_obs)

        self.qscale_w.copy_( (2**self.num_bits_obs * self.observer_weight.scale).round() )
        self.observer_weight.scale = self.qscale_w / (2 ** self.num_bits_obs)

        self.qscale_out.copy_( (2**self.num_bits_obs * self.observer_output.scale).round() )
        self.observer_output.scale = self.qscale_out / (2 ** self.num_bits_obs)

        # Compute scaling factor m
        qscale_m = (self.observer_weight.scale * self.observer_input.scale) / self.observer_output.scale
        self.qscale_m.copy_( (2**self.num_bits_obs * qscale_m).round() )
        self.m.copy_(self.qscale_m / (2 ** self.num_bits_obs))

        # Merge batch normalization parameters
        std = torch.sqrt(self.norm.running_var + self.norm.eps)
        weight, bias = self.merge_norm(self.norm.running_mean, std)

        with torch.no_grad():
            # Initialize quantized linear layer
            self.qlinear = Linear(self.input_dim, self.output_dim, bias=True).to(self.linear.weight.device)

            # Quantize weights
            quantized_weight = self.observer_weight.quantize_tensor(weight)
            quantized_weight = quantized_weight - self.observer_weight.zero_point
            self.qlinear.weight.copy_(quantized_weight)

            # Quantize biases
            quantized_bias = quantize_tensor(
                bias,
                scale=self.observer_weight.scale * self.observer_input.scale,
                zero_point=0,
                num_bits=32,
                signed=True,
            )
            self.qlinear.bias.copy_(quantized_bias)
    
    def get_parameters(self,
                       file_name: str = None):
        
        with open(file_name, 'w') as f:
            '''Save scales and zero points to file.'''
            f.write(f"Input scale ({int(self.num_bits_obs)} bit):\n {int(self.qscale_in)}\n")
            f.write(f"Input zero point:\n {int(self.observer_input.zero_point)}\n")
            f.write(f"Weight scale ({int(self.num_bits_obs)} bit):\n {int(self.qscale_w)}\n")
            f.write(f"Weight zero point:\n {int(self.observer_weight.zero_point)}\n")
            f.write(f"Output scale ({int(self.num_bits_obs)} bit):\n {int(self.qscale_out)}\n")
            f.write(f"Output zero point:\n {int(self.observer_output.zero_point)}\n")
            f.write(f"M Scales ({int(self.num_bits_obs)} bit):\n {int(self.qscale_m)}\n")

            '''Save weights and bias to file.'''
            bias = torch.flip(self.qlinear.bias, [0])
            bias = bias.detach().cpu().numpy().astype(np.int32).tolist()
            weight = torch.flip(self.qlinear.weight, [1])
            weight = weight.detach().cpu().numpy().astype(np.int32).tolist()
            
            f.write(f"Weight ({int(self.num_bits)} bit):\n")
            for idx, w in enumerate(weight):
                f.write(f"weights_conv[{idx}] = {str(w).replace('[', '{').replace(']', '}') + ';'}\n")

            f.write(f"\nBias ({int(self.num_bits)} bit):\n")
            f.write(f"bias_conv = {str(bias).replace('[', '{').replace(']', '}') + ';'}\n")

            '''Save LUT for POS quantization to file.'''
            input_range = list(range(int(self.observer_input.min), int(self.observer_input.max + 1)))
            output_range = self.observer_input.quantize_tensor(torch.tensor(input_range).to(self.linear.weight.device)) - self.observer_input.zero_point
            output_range = output_range.detach().cpu().numpy().astype(np.int32).tolist()

            f.write(f"Input range ({int(self.num_bits)} bit):\n {input_range}\n")
            f.write(f"Output range ({int(self.num_bits)} bit):\n {output_range}\n")
        
        with open(file_name.replace('.txt', '.mem'), 'w') as f:
            for idx, we in enumerate(weight):
                bin_vec = [np.binary_repr(w+self.observer_weight.zero_point.to(torch.int32).item(), width=self.num_bits+1)[1:] for w in we]
                Z1a2 = sum([w + self.observer_weight.zero_point.to(torch.int32).item() for w in we]) * \
                      self.observer_input.zero_point.to(torch.int32).item()
                NZ1Z2 = self.input_dim * \
                            self.observer_input.zero_point.to(torch.int32).item() * \
                                  self.observer_weight.zero_point.to(torch.int32).item()
                # Concat to bin_vec binary repr of bias
                bin_vec = bin_vec + [np.binary_repr(bias[len(bias)-idx-1] - Z1a2 + NZ1Z2, width=32)]
                dlugi_ciag_bitow = ''.join(bin_vec)
                # split to 72 bits chunks and write to file
                for i in range(0, len(dlugi_ciag_bitow), 72):
                    wartosc_hex = hex(int(dlugi_ciag_bitow[i:i+72], 2))
                    f.write(f"{str(wartosc_hex)[2:]}\n")


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.linear}, '
                f'global_nn={self.global_nn}), num_bits={self.num_bits}')