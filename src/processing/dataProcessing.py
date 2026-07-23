import os
import numpy as np
import pandas as pd
import torch

# ==================== 1. 定义空间网格分辨率与边界 ====================
x_min, x_max = -15.0, 45.0
y_min, y_max = 0.0, 15.0
z_min, z_max = -10.0, 150.0

Nx, Ny, Nz = 64, 64, 256
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
z = np.linspace(z_min, z_max, Nz)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# ==================== 2. 生成目标张量 (Target Tensor) ====================
def generate_target_tensor(csv_path, output_dir="data"):
    # 1. 读取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 2. 获取所有独立的设计 (共 80.0 个)
    designs = sorted(df['design'].unique())
    
    os.makedirs(output_dir, exist_ok=True)
    Y_list = []
    
    print(f"🚀 正在解析 {len(designs)}.0 个设计的扫频 S 参数...")
    for d in designs:
        # 提取当前设计，并严格按照频率升序排序，确保 601.0 个频点对齐
        sub_df = df[df['design'] == d].sort_values('Freq')
        
        # 提取 4.0 个通道: s11_real, s11_img, s21_real, s21_img
        s_params = sub_df[['s11_real', 's11_img', 's21_real', 's21_img']].values  # 形状: (601, 4)
        
        # 转置为 (4, 601)，以便将通道维度放在前面
        Y_list.append(s_params.T)
        
    # 3. 组合并保存
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32)  # 形状: [80.0, 4.0, 601.0]
    
    output_path = "../../data/fno2_target_fields.pt"
    torch.save(Y_tensor, output_path)
    
    print(f"✅ Target 文件打包成功！已保存至: {output_path}")
    print(f"📐 目标数据维度: {Y_tensor.shape}")

# ==================== 2. 几何掩膜辅助数学函数 ====================

def get_hexagon_mask(X, Z, xc, zc, R):
    """
    计算正六边形腔体掩膜 (Flat-topped 扁平顶结构)
    R: 六边形外接圆半径 (即 cx)
    """
    """
    【修正】R 是外接圆半径 (cx)。
    HFSS 中顶点位于 X 轴，因此平滑的边垂直于 Z 轴。
    """
    dx = X - xc
    dz = Z - zc
    
    sin60 = np.sqrt(3.0) / 2.0
    cos60 = 0.5
    
    # 距离中心的 Z 方向最大距离为 R * sin60 (即 Cx_cos)
    cond1 = np.abs(dz) <= (R * sin60)
    cond2 = np.abs(cos60 * dz + sin60 * dx) <= (R * sin60)
    cond3 = np.abs(-cos60 * dz + sin60 * dx) <= (R * sin60)
    return cond1 & cond2 & cond3

def get_rotated_window_mask(X, Z, p1, p2, width, thickness):
    x1, z1 = p1
    x2, z2 = p2
    xm, zm = (x1 + x2) / 2.0, (z1 + z2) / 2.0
    
    vx, vz = x2 - x1, z2 - z1
    dist = np.hypot(vx, vz)
    if dist < 1e-9:
        return np.zeros_like(X, dtype=bool)
        
    ux, uz = vx / dist, vz / dist
    perpx, perpz = -uz, ux
    
    dx = X - xm
    dz = Z - zm
    
    d_along = dx * ux + dz * uz
    d_across = dx * perpx + dz * perpz
    
    cond_along = np.abs(d_along) <= (thickness / 2.0)
    cond_across = np.abs(d_across) <= (width / 2.0)
    return cond_along & cond_across

# ==================== 3. 参数配置与严格精度坐标计算 ====================
csv_path = "../../data/formal2_training_dataset_complex_long.csv"
df = pd.read_csv(csv_path)
df_unique = df.drop_duplicates(subset=['design']).sort_values('design')
print(f"📊 成功读取原始数据，共检测到 {len(df_unique)} 个独特的设计几何(Design)")

# 【核心修正】提取自 HFSS 的原始变量，全程保持浮点最高精度，不进行任何中间四舍五入
cx = 18.0
ch = 12.0
fw = 1.5           # HFSS中的真实耦合窗厚度
rad_p = 6.0
port_out_rad = 2.0447
port_len = 2.0

Cx_cos = cx * np.cos(np.pi / 6.0)
Cx_sin = cx * np.sin(np.pi / 6.0)
Len = 2.0 * Cx_cos + fw

cavity_centers = []
for i in range(7):
    # 奇数腔体(Idx 0,2,4,6)的 X 为 0；偶数腔体(Idx 1,3,5)的 X 平移
    xc = 0.0 if i % 2 == 0 else (Len * np.cos(np.pi / 6.0))
    zc = Cx_cos + i * Len * np.sin(np.pi / 6.0)
    cavity_centers.append((xc, zc))

# ==================== 4. 批量执行几何变换 ====================
X_tensors = []

for i, (index, row) in enumerate(df_unique.iterrows()):
    print(f"\r🚀 [进度] 正在绘制第 {i + 1}/{len(df_unique)} 个 design...", end="", flush=True)
    
    grid_3d = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    
    # --- 步骤 A: 绘制 7 个六角形空气腔体 ---
    for j, (xc, zc) in enumerate(cavity_centers):
        hex_2d_mask = get_hexagon_mask(X, Z, xc, zc, cx)
        hex_3d_mask = hex_2d_mask & (Y >= 0.0) & (Y <= ch)
        grid_3d[hex_3d_mask] = 1.0
        
    # --- 步骤 B: 绘制 6 个倾斜的耦合窗通道 ---
    cws = [row['F3_Cw12'], row['F3_Cw23'], row['F3_Cw34'], 
           row['F3_Cw34'], row['F3_Cw23'], row['F3_Cw12']]
    for j in range(6):
        p1 = cavity_centers[j]
        p2 = cavity_centers[j+1]
        cw_width = cws[j]
        
        # 传入正确的厚度 fw
        win_2d_mask = get_rotated_window_mask(X, Z, p1, p2, cw_width, fw)
        win_3d_mask = win_2d_mask & (Y >= 0.0) & (Y <= ch)
        grid_3d[win_3d_mask] = 1.0

    # --- 步骤 C: 植入 7 个金属调谐螺杆 ---
    rls = [row['F3_Rl1'], row['F3_Rl2'], row['F3_Rl3'], row['F3_Rl4'], 
           row['F3_Rl5'], row['F3_Rl6'], row['F3_Rl7']]
    for j, (xc, zc) in enumerate(cavity_centers):
        post_length = rls[j]
        post_2d_mask = ((X - xc)**2 + (Z - zc)**2) <= (rad_p**2)
        post_3d_mask = post_2d_mask & (Y >= (ch - post_length)) & (Y <= ch)
        grid_3d[post_3d_mask] = 0.0

    # --- 步骤 D: 绘制输入与输出端口 ---
    # 【修正】悬浮在腔体高度中间 (Y - ch/2)，紧贴腔体平坦边缘
    # Port 1 连接于 Z = 0
    port1_mask = ((X - cavity_centers[0][0])**2 + (Y - ch/2.0)**2 <= port_out_rad**2) & \
                 (Z >= (cavity_centers[0][1] - Cx_cos - port_len)) & (Z <= (cavity_centers[0][1] - Cx_cos))
    grid_3d[port1_mask] = 1.0
    
    # Port 2 连接于 Z 末端 (即 Z_center + Cx_cos)
    port2_zc = cavity_centers[6][1] + Cx_cos
    port2_mask = ((X - cavity_centers[6][0])**2 + (Y - ch/2.0)**2 <= port_out_rad**2) & \
                 (Z >= port2_zc) & (Z <= (port2_zc + port_len))
    grid_3d[port2_mask] = 1.0

    X_tensors.append(torch.from_numpy(grid_3d).unsqueeze(0))

# ==================== 5. 合并并存储为 FNO 输入张量 ====================
print("\n打包数据集...")
X_dataset = torch.stack(X_tensors)
output_path = "../../data/fno2_input_geometry_dataset.pt"
torch.save(X_dataset, output_path)

print(f"🎉 转换完成! FNO 输入张量维度 (Batch, Channel, Nx, Ny, Nz): {X_dataset.shape}")