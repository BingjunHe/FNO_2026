import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把 src 加入路径
from model.FNOModel import FNOModel
from processing.geometry_generator import generate_geometry_tensor  

def main():
    # ==================== 1. 环境与路径配置 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "../../data/formal3_training_dataset_reim_long.csv"
    #formal2_training_dataset_complex_long.csv 小数据
    #formal3_training_dataset_reim_long.csv 大数据
    checkpoint_path = "../model/model-checkpoint/fno-3d-epoch=49-val_loss=2.9017.ckpt"
    target_design = "Design0148"  # 目标对比设计
    
    print(f"🚀 开始自动化对比流程，目标样本: {target_design}")

    # ==================== 2. 从 CSV 自动提取真实数据与几何参数 ====================
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到原始数据文件: {csv_path}")
        
    df = pd.read_csv(csv_path)
    # 兼容处理：防止 design 列存的是纯数字 35 或字符串 design035
    sub_df = df[df['design'].astype(str) == target_design].sort_values('Freq')
    
    if len(sub_df) == 0:
        # 尝试匹配数字 35
        sub_df = df[df['design'].astype(str) == target_design].sort_values('Freq')
        
    if len(sub_df) == 0:
        raise ValueError(f"在 CSV 中未找到设计名称为 {target_design} 的数据，请检查名称是否匹配！")

    print(f"📊 成功提取真值数据，共 {len(sub_df)} 个频点。")

    # A. 提取用于画图横坐标的真实频率轴 (5.4 - 6.0 GHz)
    freq_axis = sub_df['Freq'].values

    # B. 提取并计算真实 S 参数的线性幅度 (利用实部和虚部计算模值)
    s11_true_linear = np.hypot(sub_df['s11_real'].values, sub_df['s11_img'].values)
    s21_true_linear = np.hypot(sub_df['s21_real'].values, sub_df['s21_img'].values)

    # C. 自动获取该设计对应的 6 个核心几何参数（取第一行即可）
    geo_row = sub_df.iloc[0]
    cw12, cw23, cw34 = geo_row['F3_Cw12'], geo_row['F3_Cw23'], geo_row['F3_Cw34']
    rl1, rl2, rl3 = geo_row['F3_Rl1'], geo_row['F3_Rl2'], geo_row['F3_Rl3']
    print(f"📐 自动提取几何参数 -> Cw: [{cw12}, {cw23}, {cw34}] | Rl: [{rl1}, {rl2}, {rl3}]")

    # ==================== 3. 几何引擎渲染与模型预测 ====================
    print("⏳ 正在通过几何引擎生成 3D 掩膜张量...")
    # 生成 FNO 模型需要的 [1, 1, 64, 64, 256] 拓扑张量
    input_tensor = generate_geometry_tensor(cw12, cw23, cw34, rl1, rl2, rl3).to(device)

    print("🔮 正在加载 FNO 模型权重并进行前向传播...")
    # 1. 先实例化当前版本的模型
    model = FNOModel(
        num_layers=8,
        in_neurons=1,
        hidden_neurons=32,
        out_neurons=1,
        modesSpace=12,
        input_size=4,
        learning_rate=1e-3,
        restart_at_epoch_n=50,
        train_loader=None,
        loss_function='L2'
    ).to(device)

    # 2. 手动加载 checkpoint 的 state_dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # 3. 把所有末尾带 2 维的权重转成复数张量
    new_state_dict = {}
    for key, param in state_dict.items():
        # 判断是否是实虚部分开的权重（最后一维是2）
        if param.shape[-1] == 2 and param.dtype != torch.complex64:
            # [..., 2] -> 转成复数张量 [...], dtype=complex64
            param_complex = torch.view_as_complex(param.contiguous())
            new_state_dict[key] = param_complex
        else:
            new_state_dict[key] = param

    # 4. 加载转换后的权重
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        # 模型预测输出，形状通常与 Target 一致为 [1, 4, 601]
        output = model(input_tensor)
        output = output.cpu().numpy().squeeze()  # 降维成 [4, 601]

    # D. 提取并计算预测 S 参数的线性幅度 (4个通道分别为: s11_real, s11_img, s21_real, s21_img)
    s11_pred_linear = np.hypot(output[0, :], output[1, :])
    s21_pred_linear = np.hypot(output[2, :], output[3, :])

    # ==================== 4. 统一转换为标准的 dB (分贝) ====================
    s11_true_db = 20 * np.log10(s11_true_linear + 1e-8)
    s21_true_db = 20 * np.log10(s21_true_linear + 1e-8)
    
    s11_pred_db = 20 * np.log10(s11_pred_linear + 1e-8)
    s21_pred_db = 20 * np.log10(s21_pred_linear + 1e-8)

    # ==================== 5. 绘制高级对比图表 ====================
    print("🎨 正在绘制预测 vs 真实 对比曲线图...")
    plt.figure(figsize=(12, 7), dpi=300)

    # --- 绘制 S11 对比曲线 ---
    plt.plot(freq_axis, s11_pred_db, label=f"$S_{{11}}$ Predicted", color="#D62728", linewidth=2.5)
    plt.plot(freq_axis, s11_true_db, '--', label=f"$S_{{11}}$ HFSS True", color="#8C564B", linewidth=2.0, alpha=0.8)

    # --- 绘制 S21 对比曲线 ---
    plt.plot(freq_axis, s21_pred_db, label=f"$S_{{21}}$ Predicted", color="#1F77B4", linewidth=2.5)
    plt.plot(freq_axis, s21_true_db, '--', label=f"$S_{{21}}$ HFSS True", color="#17BECF", linewidth=2.0, alpha=0.8)

    # --- 视觉美化配置 ---
    plt.title(f"FNO 3D Prediction vs HFSS Ground Truth ({target_design})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Frequency (GHz)", fontsize=12, labelpad=8)
    plt.ylabel("Magnitude (dB)", fontsize=12, labelpad=8)
    
    plt.xlim(5.4, 6.0)  # 严格限制在你的频率框内
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="lower left", frameon=True, shadow=True)

    # 保存并直观展示
    output_png = f"../../output/fno_vs_truth_{target_design}.png"
    plt.savefig(output_png, bbox_inches='tight')
    print(f"🎉 拟合对比图已成功保存至: {output_png}")
    plt.show()

if __name__ == "__main__":
    main()