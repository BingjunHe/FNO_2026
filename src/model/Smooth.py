import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqSmoothnessModule(nn.Module):
    def __init__(self, channels=4, freq_points=601):
        super().__init__()
        self.freq_points = freq_points
        # rfft 对 601 点单边减半后是 301 个频域系数
        self.freq_features = freq_points // 2 + 1
        
        # 可学习的频域复数滤波器，初始化为全通(1.0)
        self.R_theta = nn.Parameter(
            torch.ones(channels, self.freq_features, dtype=torch.cfloat)
        )
        
    def forward(self, x):
        # 输入 x 形状: [B, 4, 601]
        
        # 1. 变换到频域
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # 2. 自适应滤波 (压制导致曲线起毛刺的高频噪声)
        x_ft_smooth = x_ft * self.R_theta
        
        # 3. 逆变换回 601 个频点
        x_smooth = torch.fft.irfft(x_ft_smooth, n=self.freq_points, dim=-1)
        
        # 4. 微波曲线带有负值，这里使用 GELU 或不加激活均可，建议不加激活保持 S 参数原貌
        return x_smooth