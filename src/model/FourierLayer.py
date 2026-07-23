import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def _complex_parameter(value):
    """Store a complex tensor as two real channels for AMP-compatible gradients."""
    return nn.Parameter(torch.view_as_real(value).contiguous())

def _as_complex(parameter):
    return torch.view_as_complex(parameter.contiguous())

class FourierLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, modesSpace, scaling=True):
        super().__init__()
        
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.modesSpace = modesSpace
        
        if scaling:
            self.scale = 1 / (self.in_neurons * self.out_neurons)
        else:
            self.scale = 1
            
        #self.weights  = nn.Parameter(self.scale * torch.rand(in_neurons, out_neurons, self.modesSpace * 2, self.modesSpace * 2, self.modesTime, dtype=torch.cfloat))
        self.weight_channel = _complex_parameter(
            self.scale * torch.rand(in_neurons, out_neurons, dtype=torch.cfloat)
        )
        self.weight_x = _complex_parameter(
            self.scale * torch.rand(out_neurons, modesSpace * 2, dtype=torch.cfloat)
        )
        self.weight_y = _complex_parameter(
            self.scale * torch.rand(out_neurons, modesSpace * 2, dtype=torch.cfloat)
        )
        self.weight_z = _complex_parameter(
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
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.fftn(x, dim=[-3, -2, -1])
            del x
            x_ft = torch.fft.fftshift(x_ft, dim=(-3, -2, -1))

            out_ft = torch.zeros(batchsize, self.out_neurons, x_ft.size(-3), x_ft.size(-2), x_ft.size(-1), dtype=x_ft.dtype, device=x_ft.device) # device=x.device
            midX, midY, midZ =  x_ft.size(-3) // 2, x_ft.size(-2) // 2, x_ft.size(-1) // 2
            
            #out_ft[..., midX - self.modesSpace:midX + self.modesSpace, midY - self.modesSpace:midY + self.modesSpace, :self.modesTime] = \
            #    self.compl_mul3d(x_ft[..., midX - self.modesSpace:midX + self.modesSpace, midY - self.modesSpace:midY + self.modesSpace, :self.modesTime], self.weights)
            x_slice = x_ft[..., midX - self.modesSpace:midX + self.modesSpace, 
                            midY - self.modesSpace:midY + self.modesSpace, 
                            midZ - self.modesSpace:midZ + self.modesSpace]
            out_slice = torch.einsum(
                "bixyz,io->boxyz", x_slice, _as_complex(self.weight_channel)
            )

            #switch the order of the weights to match the einsum
            wx = _as_complex(self.weight_x).view(1, self.out_neurons, self.modesSpace * 2, 1, 1)
            wy = _as_complex(self.weight_y).view(1, self.out_neurons, 1, self.modesSpace * 2, 1)
            wz = _as_complex(self.weight_z).view(1, self.out_neurons, 1, 1, self.modesSpace * 2)
            out_slice = out_slice * wx * wy * wz
            out_ft[..., midX - self.modesSpace:midX + self.modesSpace, 
                    midY - self.modesSpace:midY + self.modesSpace, 
                    midZ - self.modesSpace:midZ + self.modesSpace] = out_slice
            del x_ft, out_slice, x_slice

            #iFFT
            out_ft = torch.fft.fftshift(out_ft, dim=(-3, -2, -1))
            out_ft = torch.fft.ifftn(out_ft, dim=[-3, -2, -1]).real
        return out_ft
    
class FourierLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # 频域保留的模态数
        self.scale = (1 / (in_channels * out_channels))
        # 频域复数权重
        self.weights = _complex_parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # 输入维度: [B, in_channels, Freq_Points] (例如 [B, 128, 601])
        B, C, F = x.shape
        
        # Step 1: 沿频率轴进行一维快速傅里叶变换 (RFFT)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.rfft(x.float(), dim=-1)
        
            # Step 2: 初始化频域输出张量
            out_ft = torch.zeros(B, self.out_channels, F // 2 + 1, device=x.device, dtype=torch.cfloat)
            
            # Step 3: 乘上滤波器权重（只对低频的前 modes 个模态进行矩阵乘法，高频置零以平滑曲线）
            out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], _as_complex(self.weights))
            
            # Step 4: 一维逆快速傅里叶变换 (IRFFT)，精确还原回 F 个频点
            x = torch.fft.irfft(out_ft, n=F, dim=-1)
        return x.to(x.dtype)