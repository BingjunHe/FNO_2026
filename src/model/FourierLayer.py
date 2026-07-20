import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FourierLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, modesSpace, modesTime, scaling=True):
        super().__init__()
        
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.modesSpace = modesSpace
        self.modesTime = modesTime
        
        if scaling:
            self.scale = 1 / (self.in_neurons * self.out_neurons)
        else:
            self.scale = 1
            
        #self.weights  = nn.Parameter(self.scale * torch.rand(in_neurons, out_neurons, self.modesSpace * 2, self.modesSpace * 2, self.modesTime, dtype=torch.cfloat))
        self.weight_channel = nn.Parameter(
            self.scale * torch.rand(in_neurons, out_neurons, dtype=torch.cfloat)
        )
        self.weight_x = nn.Parameter(
            self.scale * torch.rand(out_neurons, modesSpace * 2, dtype=torch.cfloat)
        )
        self.weight_y = nn.Parameter(
            self.scale * torch.rand(out_neurons, modesSpace * 2, dtype=torch.cfloat)
        )
        self.weight_z = nn.Parameter(
            self.scale * torch.rand(out_neurons, modesSpace * 2, dtype=torch.cfloat)
        )

    '''
    def compl_mul3d(self, input, weights, einsumBool=True): 
    # (batch, in_channel, x,y,t), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        if einsumBool: # time for 1 forward t = 0.0082
            return torch.einsum("bixyt,ioxyt->boxyt", input, weights)
        else: # time for 1 forward t = 0.058
            batch_size = input.shape[0]
            # out_neurons = self.weights.shape[1]
            x_size = input.shape[2]
            y_size = input.shape[3]
            t_size = input.shape[4]

            out = torch.zeros(batch_size, self.out_neurons, x_size, y_size, t_size)
            for i in range(t_size):
                for j in range(y_size):
                    for k in range(x_size):
                        out[..., k, j, i] = torch.matmul(input[..., k, j, i], self.weights[..., k, j, i])
            return out
    '''

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.fftn(x, dim=[-3, -2, -1])
        xShapeLast = x.shape[-1]
        del x
        x_ft = torch.fft.fftshift(x_ft, dim=(-3, -2, -1))

        out_ft = torch.zeros(batchsize, self.out_neurons, x_ft.size(-3), x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x_ft.device) # device=x.device
        midX, midY, midZ =  x_ft.size(-3) // 2, x_ft.size(-2) // 2, x_ft.size(-1) // 2
        
        #out_ft[..., midX - self.modesSpace:midX + self.modesSpace, midY - self.modesSpace:midY + self.modesSpace, :self.modesTime] = \
        #    self.compl_mul3d(x_ft[..., midX - self.modesSpace:midX + self.modesSpace, midY - self.modesSpace:midY + self.modesSpace, :self.modesTime], self.weights)
        x_slice = x_ft[..., midX - self.modesSpace:midX + self.modesSpace, 
                           midY - self.modesSpace:midY + self.modesSpace, 
                           midZ - self.modesTime:midZ + self.modesTime]
        out_slice = torch.einsum("bixyz, io -> boxyz", x_slice, self.weight_channel)

        #switch the order of the weights to match the einsum
        wx = self.weight_x.view(1, self.out_neurons, self.modesSpace * 2, 1, 1)
        wy = self.weight_y.view(1, self.out_neurons, 1, self.modesSpace * 2, 1)
        wz = self.weight_z.view(1, self.out_neurons, 1, 1, self.modesTime * 2)
        out_slice = out_slice * wx * wy * wz
        out_ft[..., midX - self.modesSpace:midX + self.modesSpace, 
                   midY - self.modesSpace:midY + self.modesSpace, 
                   midZ - self.modesTime:midZ + self.modesTime] = out_slice
        del x_ft, out_slice, x_slice

        #iFFT
        out_ft = torch.fft.fftshift(out_ft, dim=(-3, -2, -1))
        out_ft = torch.fft.ifftn(out_ft, dim=[-3, -2, -1]).real
        return out_ft