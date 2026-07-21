import torch
import torch.nn as nn
import torch.nn.functional as F

class SParamHybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # dB 损失的权重

    def forward(self, y_hat, y):
        """
        y_hat, y 形状: [B, 4, 601]
        通道: 0: s11_re, 1: s11_im, 2: s21_re, 3: s21_im
        """
        B = y.size(0)
        
        # ================= 1. 计算 FNO 标配的 相对 L2 损失 =================
        # 将通道和频点展平为一维向量计算范数
        y_hat_flat = y_hat.view(B, -1)
        y_flat = y.view(B, -1)
        
        diff_norms = torch.norm(y_hat_flat - y_flat, p=2, dim=1)
        y_norms = torch.norm(y_flat, p=2, dim=1)
        
        # 避免除以 0（加上微弱 eps）
        rel_l2_loss = torch.mean(diff_norms / (y_norms + 1e-6))

        # ================= 2. 计算 显式 dB 域 MSE 损失 =================
        # 计算预测和真实的各参数幅值平方 (加上 1e-8 防止 log 爆炸)
        s11_mag_sq_hat = y_hat[:, 0, :]**2 + y_hat[:, 1, :]**2 + 1e-8
        s21_mag_sq_hat = y_hat[:, 2, :]**2 + y_hat[:, 3, :]**2 + 1e-8
        
        s11_mag_sq_true = y[:, 0, :]**2 + y[:, 1, :]**2 + 1e-8
        s21_mag_sq_true = y[:, 2, :]**2 + y[:, 3, :]**2 + 1e-8
        
        # 转换为 dB 域: 10 * log10(mag^2) 
        s11_db_hat = 10.0 * torch.log10(s11_mag_sq_hat)
        s21_db_hat = 10.0 * torch.log10(s21_mag_sq_hat)
        
        s11_db_true = 10.0 * torch.log10(s11_mag_sq_true)
        s21_db_true = 10.0 * torch.log10(s21_mag_sq_true)
        
        # 限制 dB 域的底线（例如微波里超过 -60dB 的深谷对关注度要求降低，防止负无穷大影响训练）
        s11_db_hat = torch.clamp(s11_db_hat, min=-60.0)
        s11_db_true = torch.clamp(s11_db_true, min=-60.0)
        s21_db_hat = torch.clamp(s21_db_hat, min=-60.0)
        s21_db_true = torch.clamp(s21_db_true, min=-60.0)
        
        # 计算 dB 域的 MSE
        db_loss = F.mse_loss(s11_db_hat, s11_db_true) + F.mse_loss(s21_db_hat, s21_db_true)

        # ================= 3. 混合总损失 =================
        return rel_l2_loss + self.alpha * db_loss