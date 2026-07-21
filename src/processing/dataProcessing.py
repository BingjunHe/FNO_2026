import os
import numpy as np
import pandas as pd
import torch

# ==================== 1. 定义空间网格分辨率与边界 ====================
# 请根据你模型的实际包络框（Bounding Box）进行微调
x_min, x_max = -30.0, 30.0
y_min, y_max = 0.0, 15.0
z_min, z_max = -10.0, 170.0

# 空间网格分辨率 (Nx, Ny, Nz) -> 推荐为 FNO 友好的 2 的幂次方
Nx, Ny, Nz = 64, 64, 256

x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
z = np.linspace(z_min, z_max, Nz)

# 生成 3D 坐标矩阵 (采用 'ij' 索引，保证维度与 X, Y, Z 一一对应)
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

def get_hexagon_mask(X, Z, xc, zc, W):
    """
    计算正六边形腔体掩膜 (Flat-topped 扁平顶结构)
    W: 六边形平行边之间的距离 (即 cx)
    """
    dx = X - xc
    dz = Z - zc
    
    # 动态精确计算 sqrt(3)/2，防止硬编码引入四舍五入误差
    sin60 = np.sqrt(3.0) / 2.0
    cos60 = 0.5
    
    # 三个方向的带状区域限制
    cond1 = np.abs(dx) <= (W / 2.0)
    cond2 = np.abs(cos60 * dx + sin60 * dz) <= (W / 2.0)
    cond3 = np.abs(-cos60 * dx + sin60 * dz) <= (W / 2.0)
    
    return cond1 & cond2 & cond3

def get_rotated_window_mask(X, Z, p1, p2, width, thickness):
    """
    计算旋转矩形耦合窗掩膜
    p1, p2: 相邻两个腔体中心的 (x, z) 坐标
    width: 耦合窗宽度 (Cw)
    thickness: 耦合窗厚度 (Iris Thickness)
    """
    x1, z1 = p1
    x2, z2 = p2
    xm, zm = (x1 + x2) / 2.0, (z1 + z2) / 2.0
    
    # 连线方向的向量与距离
    vx, vz = x2 - x1, z2 - z1
    dist = np.hypot(vx, vz)
    
    if dist < 1e-9:
        return np.zeros_like(X, dtype=bool)
        
    # 单位方向向量及其法向量
    ux, uz = vx / dist, vz / dist
    perpx, perpz = -uz, ux
    
    # 计算网格点相对于窗口中心的投影距离
    dx = X - xm
    dz = Z - zm
    
    d_along = dx * ux + dz * uz
    d_across = dx * perpx + dz * perpz
    
    cond_along = np.abs(d_along) <= (thickness / 2.0)
    cond_across = np.abs(d_across) <= (width / 2.0)
    
    return cond_along & cond_across

# ==================== 3. 读取数据集与参数配置 ====================
csv_path = "../../data/formal3_training_dataset_reim_long.csv"
df = pd.read_csv(csv_path)

df_unique = df.drop_duplicates(subset=['design']).sort_values('design')
print(f"📊 成功读取原始数据，共检测到 {len(df_unique)} 个独特的设计几何(Design)")

# 【核心配置】 7个腔体在 AEDT 中测出的精确 (X, Z) 中心坐标 (保留完整小数位)
# 根据你截图中的折叠形态，X 坐标应该呈现交错分布（例如负、正、负、正...）
cavity_centers = [
    (-11.54701, 15.0),    # Cavity 1
    (11.54701, 35.0),     # Cavity 2
    (-11.54701, 55.0),    # Cavity 3
    (11.54701, 75.0),     # Cavity 4
    (-11.54701, 95.0),    # Cavity 5
    (11.54701, 115.0),    # Cavity 6
    (-11.54701, 135.0)    # Cavity 7
]

iris_thickness = 3.0  # 耦合窗沿传播通道的物理厚度 (mm)

# ==================== 4. 批量执行几何变换 ====================
X_tensors = []

for i, (index, row) in enumerate(df_unique.iterrows()):
    print(f"\r[进度] 正在绘制第 {i + 1}/{len(df_unique)} 个 design: {row['design']} ...", end="", flush=True)
    # 1. 初始化 3D 矩阵：0.0 代表金属实体（背景）
    grid_3d = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    
    cx = 18
    ch = 12
    rad_p = 6
    
    # --- 步骤 A: 绘制 7 个六角形空气腔体 (赋值为 1.0) ---
    for i, (xc, zc) in enumerate(cavity_centers):
        # Y 方向限制在 0 到腔体高度 ch 之间
        hex_2d_mask = get_hexagon_mask(X, Z, xc, zc, cx)
        hex_3d_mask = hex_2d_mask & (Y >= 0.0) & (Y <= ch)
        grid_3d[hex_3d_mask] = 1.0
        
    # --- 步骤 B: 绘制 6 个倾斜的耦合窗通道 (赋值为 1.0) ---
    cws = [
        row['F3_Cw12'], row['F3_Cw23'], row['F3_Cw34'], 
        row['F3_Cw34'], row['F3_Cw23'], row['F3_Cw12']
    ]
    for i in range(6):
        p1 = cavity_centers[i]
        p2 = cavity_centers[i+1]
        cw_width = cws[i]
        
        # 计算连接当前两个腔体的旋转耦合窗
        win_2d_mask = get_rotated_window_mask(X, Z, p1, p2, cw_width, iris_thickness)
        win_3d_mask = win_2d_mask & (Y >= 0.0) & (Y <= ch)
        grid_3d[win_3d_mask] = 1.0

    # --- 步骤 C: 植入 7 个金属调谐螺杆 (覆盖抹回 0.0) ---
    rls = [
        row['F3_Rl1'], row['F3_Rl2'], row['F3_Rl3'], 4.52, 
        row['F3_Rl3'], row['F3_Rl2'], row['F3_Rl1']
    ]
    for i, (xc, zc) in enumerate(cavity_centers):
        post_length = rls[i]
        
        # 调谐螺杆在 X-Z 平面是圆柱体，从顶部 Y = ch 向下插入
        post_2d_mask = ((X - xc)**2 + (Z - zc)**2) <= (rad_p**2)
        post_3d_mask = post_2d_mask & (Y >= (ch - post_length)) & (Y <= ch)
        grid_3d[post_3d_mask] = 0.0

    # --- 步骤 D: 绘制输入与输出端口 (赋值为 1.0) ---
    # 假设输入/输出端口分别垂直连接在 Cavity 1 和 Cavity 7 处
    p_rad_in = 0.635
    p_rad_out = 2.0447
    p_len = 2
    
    # 示例输入端口：连接在第 1 腔体中心，向 Z 轴负方向延伸
    port1_mask = ((X - cavity_centers[0][0])**2 + Y**2 <= p_rad_in**2) & \
                 (Z >= (cavity_centers[0][1] - cx/2.0 - p_len)) & (Z <= cavity_centers[0][1])
    grid_3d[port1_mask] = 1.0
    
    # 示例输出端口：连接在第 7 腔体中心，向 Z 轴正方向延伸
    port2_mask = ((X - cavity_centers[6][0])**2 + Y**2 <= p_rad_out**2) & \
                 (Z >= cavity_centers[6][1]) & (Z <= (cavity_centers[6][1] + cx/2.0 + p_len))
    grid_3d[port2_mask] = 1.0

    # 2. 将当前的样本 Tensor 放入数据集列表中
    # 增加通道维度，形成 (1, Nx, Ny, Nz)
    X_tensors.append(torch.from_numpy(grid_3d).unsqueeze(0))

# ==================== 5. 合并并存储为 FNO 输入张量 ====================
# 最终堆叠成形状为 (Batch_Size, Channel=1, Nx, Ny, Nz) 的 PyTorch 张量
X_dataset = torch.stack(X_tensors)

# 保存张量文件到本地，供 FNO 训练直接读取
output_path = "../../data/fno2_input_geometry_dataset.pt"
torch.save(X_dataset, output_path)

generate_target_tensor(csv_path)

print(f"🎉 转换完成！")
print(f"数据集已保存至: {output_path}")
print(f"FNO 输入张量维度 (Batch, Channel, Nx, Ny, Nz): {X_dataset.shape}")