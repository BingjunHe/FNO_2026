import numpy as np
import torch

# ==================== 1. 全局初始化：空间网格分辨率与边界 ====================
# 在模块加载时仅初始化一次，避免函数重复调用时的计算开销
x_min, x_max = -30.0, 30.0
y_min, y_max = 0.0, 15.0
z_min, z_max = -10.0, 170.0

Nx, Ny, Nz = 64, 64, 256

_x = np.linspace(x_min, x_max, Nx)
_y = np.linspace(y_min, y_max, Ny)
_z = np.linspace(z_min, z_max, Nz)

# 生成 3D 坐标全局矩阵
X, Y, Z = np.meshgrid(_x, _y, _z, indexing='ij')

# 固定物理结构常量
cavity_centers = [
    (-11.54701, 15.0),    # Cavity 1
    (11.54701, 35.0),     # Cavity 2
    (-11.54701, 55.0),    # Cavity 3
    (11.54701, 75.0),     # Cavity 4
    (-11.54701, 95.0),    # Cavity 5
    (11.54701, 115.0),    # Cavity 6
    (-11.54701, 135.0)    # Cavity 7
]

iris_thickness = 3.0  # 耦合窗物理厚度 (mm)
cx = 18               # 六边形平行边距
ch = 12               # 腔体高度
rad_p = 6             # 螺杆半径
p_rad_in = 0.635      # 输入端口半径
p_rad_out = 2.0447    # 输出端口半径
p_len = 2             # 端口延伸长度


# ==================== 2. 几何掩膜辅助数学函数 ====================

def _get_hexagon_mask(X, Z, xc, zc, W):
    """计算正六边形腔体掩膜"""
    dx = X - xc
    dz = Z - zc
    sin60 = np.sqrt(3.0) / 2.0
    cos60 = 0.5
    
    cond1 = np.abs(dx) <= (W / 2.0)
    cond2 = np.abs(cos60 * dx + sin60 * dz) <= (W / 2.0)
    cond3 = np.abs(-cos60 * dx + sin60 * dz) <= (W / 2.0)
    return cond1 & cond2 & cond3

def _get_rotated_window_mask(X, Z, p1, p2, width, thickness):
    """计算旋转矩形耦合窗掩膜"""
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


# ==================== 3. 核心导出接口 ====================

def generate_geometry_tensor(F3_Cw12, F3_Cw23, F3_Cw34, F3_Rl1, F3_Rl2, F3_Rl3):
    """
    根据传入的 6 个核心设计参数，直接构建并返回用于 FNO 模型推理的 3D 几何张量。
    
    参数:
        F3_Cw12 (float): 1-2 及 6-7 腔体间的耦合窗宽度
        F3_Cw23 (float): 2-3 及 5-6 腔体间的耦合窗宽度
        F3_Cw34 (float): 3-4 及 4-5 腔体间的耦合窗宽度
        F3_Rl1  (float): 1 及 7 号腔体调谐螺杆深度
        F3_Rl2  (float): 2 及 6 号腔体调谐螺杆深度
        F3_Rl3  (float): 3 及 5 号腔体调谐螺杆深度
        
    返回:
        torch.Tensor: 形状为 [1, 1, 64, 64, 256] 的 float32 类型 PyTorch 张量
    """
    # 1. 初始化 3D 矩阵：0.0 代表金属实体（背景背景）
    grid_3d = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    
    # --- 步骤 A: 绘制 7 个六角形空气腔体 (赋值为 1.0) ---
    for xc, zc in cavity_centers:
        hex_2d_mask = _get_hexagon_mask(X, Z, xc, zc, cx)
        hex_3d_mask = hex_2d_mask & (Y >= 0.0) & (Y <= ch)
        grid_3d[hex_3d_mask] = 1.0
        
    # --- 步骤 B: 绘制 6 个倾斜的耦合窗通道 (赋值为 1.0) ---
    cws = [F3_Cw12, F3_Cw23, F3_Cw34, F3_Cw34, F3_Cw23, F3_Cw12]
    for i in range(6):
        p1 = cavity_centers[i]
        p2 = cavity_centers[i+1]
        cw_width = cws[i]
        
        win_2d_mask = _get_rotated_window_mask(X, Z, p1, p2, cw_width, iris_thickness)
        win_3d_mask = win_2d_mask & (Y >= 0.0) & (Y <= ch)
        grid_3d[win_3d_mask] = 1.0

    # --- 步骤 C: 植入 7 个金属调谐螺杆 (覆盖抹回 0.0) ---
    # 4.52 为 4 号中间腔体的固定螺杆深度
    rls = [F3_Rl1, F3_Rl2, F3_Rl3, 4.52, F3_Rl3, F3_Rl2, F3_Rl1]
    for i, (xc, zc) in enumerate(cavity_centers):
        post_length = rls[i]
        
        post_2d_mask = ((X - xc)**2 + (Z - zc)**2) <= (rad_p**2)
        post_3d_mask = post_2d_mask & (Y >= (ch - post_length)) & (Y <= ch)
        grid_3d[post_3d_mask] = 0.0

    # --- 步骤 D: 绘制输入与输出端口 (赋值为 1.0) ---
    # 输入端口：连接在第 1 腔体中心，向 Z 轴负方向延伸
    port1_mask = ((X - cavity_centers[0][0])**2 + Y**2 <= p_rad_in**2) & \
                 (Z >= (cavity_centers[0][1] - cx/2.0 - p_len)) & (Z <= cavity_centers[0][1])
    grid_3d[port1_mask] = 1.0
    
    # 输出端口：连接在第 7 腔体中心，向 Z 轴正方向延伸
    port2_mask = ((X - cavity_centers[6][0])**2 + Y**2 <= p_rad_out**2) & \
                 (Z >= cavity_centers[6][1]) & (Z <= (cavity_centers[6][1] + cx/2.0 + p_len))
    grid_3d[port2_mask] = 1.0

    # 2. 转化为 PyTorch 张量并增加 Batch 与 Channel 维度 -> 形成 [1, 1, Nx, Ny, Nz]
    tensor_5d = torch.from_numpy(grid_3d).unsqueeze(0).unsqueeze(0)
    
    return tensor_5d


# ==================== 4. 测试运行验证 ====================
if __name__ == "__main__":
    print("⏳ 正在测试生成单结构几何张量...")
    # 传入一组测试数值
    test_tensor = generate_geometry_tensor(
        F3_Cw12=4.2, 
        F3_Cw23=3.8, 
        F3_Cw34=3.5, 
        F3_Rl1=2.1, 
        F3_Rl2=5.4, 
        F3_Rl3=3.3
    )
    print("✅ 测试成功！")
    print(f"📊 生成的张量数据类型: {test_tensor.dtype}")
    print(f"📐 输出张量标准维度 (Batch, Channel, Nx, Ny, Nz): {list(test_tensor.shape)}")