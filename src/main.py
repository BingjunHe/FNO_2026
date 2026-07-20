import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from model.FNOModel import FNOModel

class HexFilterDataset(Dataset):
    def __init__(self, geometry_path, target_path=None):
        """
        直接加载离线预处理好的 3D 几何张量 X 与对应的目标物理场数据 Y。
        
        geometry_path: 预处理生成的几何张量文件路径 (如 'data/fno_input_geometry_dataset.pt')
        target_path: 你的 HFSS 仿真真实场数据张量文件路径 (若尚未导出，会自动启用调试模式生成 Dummy 数据占位)
        """
        if not os.path.exists(geometry_path):
            raise FileNotFoundError(
                f"⚠️ 未找到预处理几何文件: {geometry_path}，请确保先运行几何预处理脚本生成该文件！"
            )
        
        # 读取高精度 3D 几何张量，直接加载至 CPU 内存中以备 DataLoader 调用。
        # 形状为 [Batch_Size, Channel=1, Nx, Ny, Nz]，完全保留了预处理时的多位小数高保真精度，中途无任何四舍五入。
        self.X = torch.load(geometry_path, map_location="cpu").float()
        
        # 载入训练的目标真值 Y (例如：HFSS 中导出的 3D 真实电磁场强度分布)
        if target_path and os.path.exists(target_path):
            self.Y = torch.load(target_path, map_location="cpu").float()
        else:
            # 调试备用：如果尚未准备好物理场真实标签 Y，自动生成与 X 形状一致的张量
            # 这样可以确保你在缺少完整仿真数据的情况下，依然能将模型、优化器以及 FP16 精度控制整体跑通
            print(f"⚠️ 未检测到真实标签文件 '{target_path}'，已自动创建同形状的测试标签以确保流水线畅通。")
            self.Y = torch.randn_like(self.X) * 10.0

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        # 此时返回的 X[idx] 形状为 [1, Nx, Ny, Nz]
        return self.X[idx], self.Y[idx]
    
def main():
    # 固定随机种子，确保你的偏微分方程（PDE）求解实验完全可复现
    L.seed_everything(42)

    # 指向你在上一步运行几何预处理代码后生成的 .pt 文件
    GEOMETRY_FILE = "../data/fno_input_geometry_dataset.pt"
    TARGET_FILE = "../data/fno_target_fields.pt"  # 预留你的真实仿真电场数据路径

    # 实例化自定义数据集
    # 此时网格分辨率直接继承自你预处理脚本里生成的尺寸 (如 64x64x256)
    dataset = HexFilterDataset(geometry_path=GEOMETRY_FILE, target_path=TARGET_FILE)

    # 划分训练集与验证集 (按照 80% 训练，20% 验证的比例)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 构建数据加载器
    # 提示：pin_memory=True 可以加速数据从内存拷贝到显卡的过程
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True  # Lightning 推荐设置，避免每个 Epoch 重启 DataLoader 线程造成的 CPU 等待开销
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )

    model = FNOModel(
        num_layers=4,
        in_neurons=1,
        hidden_neurons=32,
        out_neurons=1,
        modesSpace=12,
        modesTime=12,
        input_size=4,
        learning_rate=1e-3,
        restart_at_epoch_n=50,
        train_loader=train_loader,
        loss_function='MSE'
    )

    # 4. 配置自动保存权重的回调函数 (Callbacks)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",       # 监控验证集损失
        dirpath="./model/model-checkpoint/",   # 权重文件保存目录
        filename="fno-3d-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,             # 只保留效果最好的 3 个模型权重
        mode="min",
    )
    
    # 监控学习率变换的回调函数
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 5. 定义强大的 Trainer（闪电训练器）
    # 这里的参数是让 3D FNO 稳定运行且不爆显存的关键
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",       # 自动检测并使用 GPU/TPU/CPU
        devices=1,                # 指定使用 1 张显卡
        
        # 【核心优化】开启 16 位混合精度训练（AMP）
        # 这会让 3D 空间的前向和反向传播显存开销直接砍掉近 50%，同时成倍提升计算速度
        precision="16-mixed",     
        
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        
        # 【备用显存大招】如果你的 3D 网格太大导致哪怕 batch_size=1 都爆显存：
        # 可以解开下面这行。它代表每 4 个 batch 累加一次梯度才更新权重，变相把 batch_size 扩大 4 倍
        # accumulate_grad_batches=4, 
    )

    # 6. 一键启动超级赛亚人模式
    print("🚀 正在启动 PyTorch Lightning 训练流水线...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("🎉 3D FNO 模型训练顺利完成！最优权重已存入 checkpoints/ 目录。")


if __name__ == "__main__":
    main()