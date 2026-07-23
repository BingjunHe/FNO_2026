import torch
import torch.nn as nn
import torch.nn.functional as F

class SParamHybridLoss(nn.Module):
    def __init__(self, alpha=0.5, valley_weight=5.0):
        super().__init__()
        self.alpha = alpha  # dB 损失的权重
        self.valley_weight = valley_weight  # 对深谷区域的追加惩罚倍数

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
        
        # 计算每个频点的绝对误差平方
        s11_squared_error = (s11_db_hat - s11_db_true) ** 2
        s21_squared_error = (s21_db_hat - s21_db_true) ** 2
        
        # 动态创建权重矩阵：如果真实值低于 -15 dB（说明进入了敏感的阻带/谐振谷），权重放大 valley_weight 倍
        s11_weight = torch.where(s11_db_true < -15.0, self.valley_weight, 1.0)
        s21_weight = torch.where(s21_db_true < -15.0, self.valley_weight, 1.0)
        
        # 计算加权后的 MSE
        db_loss = torch.mean(s11_squared_error * s11_weight) + torch.mean(s21_squared_error * s21_weight)
        
        # ================= 4. 【追加高级防御】引入无穷范数（最大误差惩罚） =================
        # 强迫网络去拉低那个全波段误差最大的点（通常就是那个没对齐的尖峰）
        max_error = torch.mean(torch.max(s11_squared_error, dim=-1)[0]) + torch.mean(torch.max(s21_squared_error, dim=-1)[0])

        # 最终混合总损失 (额外拨出 0.1 的权重给最大误差项)
        return rel_l2_loss + self.alpha * db_loss + 0.1 * max_error