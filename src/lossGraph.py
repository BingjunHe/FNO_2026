import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. 自动寻找最新的 Lightning 实验日志版本
log_dir = "./lightning_logs/"
versions = [
    d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))
]
latest_version = sorted(versions, key=lambda x: int(x.split("_")[-1]))[-1]
csv_path = os.path.join(log_dir, latest_version, "metrics.csv")

print(f"📈 正在读取日志文件: {csv_path}")
df = pd.read_csv(csv_path)

# 2. 分离训练损失与验证损失（因为它们的记录频次不同，分开提取能避免画出折线毛刺）
# 按 epoch 进行聚合取均值，使曲线更平滑
train_df = df.dropna(subset=["train_loss_epoch"])
val_df = df.dropna(subset=["val_loss"])

# 3. 开始绘制全流程损失演进图
plt.figure(figsize=(10, 6), dpi=300)

# 绘制总损失
plt.plot(
    train_df["epoch"],
    train_df["train_loss_epoch"],
    label="Train Total Loss",
    color="#1f77b4",
    linewidth=2,
)
plt.plot(
    val_df["epoch"],
    val_df["val_loss"],
    label="Val Total Loss",
    color="#ff7f0e",
    linestyle="--",
    linewidth=2,
)

# 4. 重点观测你的物理约束（PINO）和数据损失收敛情况
if "data_loss_epoch" in df.columns:
    plt.plot(
        train_df["epoch"],
        train_df["data_loss_epoch"],
        label="Data Loss",
        color="#2ca02c",
        alpha=0.7,
    )
if "physics_loss_epoch" in df.columns:
    plt.plot(
        train_df["epoch"],
        train_df["physics_loss_epoch"],
        label="Physics Loss",
        color="#d62728",
        alpha=0.7,
    )

# 5. 图表精细化美化
plt.title("FNO Model Training Process - Loss Curves", fontsize=14, pad=15)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss Value", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="upper right")

# 如果损失值跨度巨大，建议取消下方这行的注释，开启对数坐标轴
plt.yscale('log')

# 6. 保存到本地
output_image_path = "../output/loss_curve.png"
plt.savefig(output_image_path, bbox_inches="tight")
plt.show()

print(f"🎉 损失函数曲线图已成功导出至: {os.path.abspath(output_image_path)}")