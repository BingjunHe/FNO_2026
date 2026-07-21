import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from .Loss import *
from .Smooth import *
from .MLP import *
from .FourierLayer import *
from .Utilities import *
# from LossFunction import *


class ModelCallbacks(L.Callback):
    def on_train_start(self, trainer):
        # RichModelSummary(max_depth=1)
        pass
        
    def on_train_end(self, trainer):
        pass


class FNOModel(L.LightningModule):
    def __init__(self, num_layers, in_neurons, hidden_neurons, out_neurons, modesSpace, modesTime, input_size, learning_rate, restart_at_epoch_n, train_loader, loss_function):
        super().__init__()
        #self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.restart_at_epoch_n = restart_at_epoch_n
        #self.padding = time_padding # set padding here based on input_size
        if train_loader is not None:
            self.n_batches = len(train_loader)
            self.n_training_samples = len(train_loader.dataset)
        else:
            self.n_batches = 0
            self.n_training_samples = 0
        self.loss_name = loss_function
        #train_batch, _ = next(iter(train_loader))
        #x_shape = train_batch.size()
        #self.register_buffer("meshgrid", get_meshgrid(x_shape))
        self.l = num_layers # number of layers
        # Network architechture
        self.p = nn.Linear(input_size, out_neurons)
        
        self.fourier = nn.ModuleList([FourierLayer(in_neurons, out_neurons, modesSpace, modesTime) for _ in range(self.l)])
        self.mlp = nn.ModuleList([MLP(out_neurons, hidden_neurons, out_neurons, kernel_size=1) for _ in range(self.l)])
        self.w = nn.ModuleList([nn.Conv3d(in_neurons, out_neurons, kernel_size=1) for _ in range(self.l)])
        
        #self.q = MLP(in_neurons, 4 * hidden_neurons, 1, kernel_size=1) 
        # 替代原先粗暴的 torch.mean，用轻量卷积层提取特征
        self.spatial_downsample = nn.Sequential(
            nn.Conv3d(out_neurons, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)) # 将残余空间压缩为 1x1x1，输出维度: [B, 128, 1, 1, 1]
        )
        
        # ==================== 1D 频域段架构 ====================
        # 输入通道 = 128 (空间特征) + 1 (动态生成的频率坐标 f) = 129
        self.freq_p = nn.Linear(129, 128)
        
        # 1D 频域傅里叶层：这里的 modes 可以设为 16 或 32（代表频域曲线的复杂程度，越小越平滑）
        self.freq_fourier = FourierLayer1D(in_channels=128, out_channels=128, modes=24)
        self.freq_mlp = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=1)
        )
        self.freq_w = nn.Conv1d(128, 128, kernel_size=1)
        
        # 最后的输出映射：将隐藏通道映射到 4 个 S 参数通道 (s11_r, s11_i, s21_r, s21_i)
        self.freq_decoder = nn.Linear(128, 4)

        self.freq_sm = FreqSmoothnessModule(channels=4, freq_points=601)
        
        if loss_function == 'L2':
            self.loss_function = SParamHybridLoss(alpha=0.2)
        elif loss_function == 'MSE':
            self.loss_function = F.mse_loss
        elif loss_function == 'MAE':
            self.loss_function = F.l1_loss
    
            
    def forward(self, x): 
        #[B, 1, X, Y, Z] -> [B, X, Y, Z, 1]
        x = x.permute(0, 2, 3, 4, 1)
        # input dim: [B, X, Y, Z, C]
        meshgrid = get_meshgrid(x.shape).to(self.device)
        x = torch.concat((x, meshgrid), dim=-1) # [B, X, Y, Z, 3 + C]
        del meshgrid
        x = self.p(x) # [B, X, Y, Z, H]
        x = x.permute(0, 4, 1, 2, 3) # [B, H, X, Y, Z]
        #x = F.pad(x, [0, self.padding]) # Zero-pad
        for fourier_layer, mlp_layer, w_layer in zip(self.fourier, self.mlp, self.w):           
            x1 = fourier_layer(x)  
            x1 = mlp_layer(x1)    
            x2 = w_layer(x)     
            x = F.gelu(x1 + x2)
        
        #x = x[..., :-self.padding] # Unpad zeros
        '''
        x = self.q(x) # [B, 1, X, Y, Z]
        x = x.permute(0, 2, 3, 4, 1)  # [B, X, Y, Z, 1]
        x = x.squeeze_(dim=-1)
        return x
        '''
        # ---------------- 2. 空间特征高级压缩 ----------------
        x_spatial = self.spatial_downsample(x) # [B, 128, 1, 1, 1]
        x_spatial = x_spatial.squeeze(-1).squeeze(-1).squeeze(-1) # 展平空间维度 -> [B, 128]
        
        # ---------------- 3. 动态构建连续频域流 ----------------
        B = x_spatial.shape[0]
        num_freqs = 601
        
        # 动态创建归一化频率坐标网格 [-1, 1]，形状: [B, 601, 1]
        freq_mesh = torch.linspace(-1, 1, num_freqs, device=x.device).view(1, num_freqs, 1).repeat(B, 1, 1)
        
        # 将空间几何特征沿着频率轴复制 601 次，形状: [B, 601, 128]
        x_spatial_expanded = x_spatial.unsqueeze(1).repeat(1, num_freqs, 1)
        
        # 将几何特征和频率坐标拼接，形状: [B, 601, 129]
        x_freq = torch.cat([x_spatial_expanded, freq_mesh], dim=-1)
        
        # ---------------- 4. 1D 频域 FNO 传播 ----------------
        x_freq = self.freq_p(x_freq)         # [B, 601, 128]
        x_freq = x_freq.permute(0, 2, 1)     # 转换为标准 1D 卷积形状: [B, 128, 601]
        
        # 频域神经算子核心计算
        x1_f = self.freq_fourier(x_freq)  
        x1_f = self.freq_mlp(x1_f)    
        x2_f = self.freq_w(x_freq)     
        x_freq = F.gelu(x1_f + x2_f)         # [B, 128, 601]
        
        # ---------------- 5. 映射输出 ----------------
        x_freq = x_freq.permute(0, 2, 1)     # [B, 601, 128]
        out = self.freq_decoder(x_freq)      # [B, 601, 4]
        
        # 转换回你原本定义的 [Batch, 4, 601] 格式
        out = out.permute(0, 2, 1)           # [B, 4, 601]
        
        # 通过频域平滑模块处理输出
        out = self.freq_sm(out)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x) # 此时输出形状已更新为: [B, 4, 601] (4通道分别代表 S11/S21 的实部与虚部)
        
        # 1. 计算常规的数据驱动监督 Loss (MSE 或 L2)
        data_loss = self.loss_function(y_hat, y) 
        
        # 2. 嵌入物理告知 (PINO)：微波网络的无源性约束 (|S11|^2 + |S21|^2 <= 1)
        # 提取各个分量的平方项
        s11_mag_sq = y_hat[:, 0, :]**2 + y_hat[:, 1, :]**2
        s21_mag_sq = y_hat[:, 2, :]**2 + y_hat[:, 3, :]**2
        
        # 计算每个频点输出的总能量
        total_energy = s11_mag_sq + s21_mag_sq
        
        # 物理定律铁律：总能量不能大于 1.0。若大于 1.0，通过 ReLU 激活函数捕获物理残差
        physics_residual = F.relu(total_energy - 1.0)
        physics_loss = F.mse_loss(physics_residual, torch.zeros_like(physics_residual))
        
        # 3. 复合总损失 (alpha 为物理损失权重，建议设在 0.1 ~ 0.5 之间，可根据收敛情况微调)
        alpha = 0.5  
        train_loss = data_loss + alpha * physics_loss
        
        # 4. 多维度指标记录与可视化
        train_mse = F.mse_loss(y_hat, y)
        log_dict = {
            'train_loss': train_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss,
            'mse_loss': train_mse
        }
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)
        return train_loss 


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # [B, 4, 601][cite: 1]
        
        # 1. 计算常规的数据驱动监督 Loss
        data_loss = self.loss_function(y_hat, y)
        
        # 2. 嵌入相同的物理告知 (PINO) 约束，检查验证集是否守恒
        s11_mag_sq = y_hat[:, 0, :]**2 + y_hat[:, 1, :]**2
        s21_mag_sq = y_hat[:, 2, :]**2 + y_hat[:, 3, :]**2
        total_energy = s11_mag_sq + s21_mag_sq
        
        physics_residual = F.relu(total_energy - 1.0)
        physics_loss = F.mse_loss(physics_residual, torch.zeros_like(physics_residual))
        
        # 3. 复合总验证损失 (alpha 权重必须与 training_step 严格一致)
        alpha = 0.5  
        val_loss = data_loss + alpha * physics_loss
        
        # 4. 对称记录指标，方便在 TensorBoard 中直接拉通对比
        log_dict = {
            'val_loss': val_loss,
            'val_data_loss': data_loss,
            'val_physics_loss': physics_loss
        }
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
        
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # [B, 4, 601][cite: 1]
        
        # 测试集同步保持物理复合公式
        data_loss = self.loss_function(y_hat, y)
        
        s11_mag_sq = y_hat[:, 0, :]**2 + y_hat[:, 1, :]**2
        s21_mag_sq = y_hat[:, 2, :]**2 + y_hat[:, 3, :]**2
        total_energy = s11_mag_sq + s21_mag_sq
        
        physics_residual = F.relu(total_energy - 1.0)
        physics_loss = F.mse_loss(physics_residual, torch.zeros_like(physics_residual))
        
        alpha = 0.5
        test_loss = data_loss + alpha * physics_loss
        
        log_dict = {
            'test_loss': test_loss,
            'test_data_loss': data_loss,
            'test_physics_loss': physics_loss
        }
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        return test_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.restart_at_epoch_n)
        
        # 使用 Lightning 标准推荐的配置字典返回，规范数据流管理
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }