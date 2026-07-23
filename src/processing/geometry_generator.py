import numpy as np
import torch

# ==================== 1. 全局初始化：空间网格分辨率与边界 ====================
# 根据真实交错排布，调整包络框使其居中，包容 X: 0 ~ 28.3, Z: -2 ~ 132.6
x_min, x_max = -15.0, 45.0
y_min, y_max = 0.0, 15.0
z_min, z_max = -10.0, 150.0

Nx, Ny, Nz = 64, 64, 256

_x = np.linspace(x_min, x_max, Nx)
_y = np.linspace(y_min, y_max, Ny)
_z = np.linspace(z_min, z_max, Nz)

# 生成 3D 坐标全局矩阵
X, Y, Z = np.meshgrid(_x, _y, _z, indexing='ij')

# ----------------- 物理结构常量与高精度坐标推导 -----------------
cx = 18.0             # 六边形外接圆半径 (顶点到中心距离)
ch = 12.0             # 腔体高度
fw = 1.5              # 耦合窗真实物理厚度 (mm)
rad_p = 6.0           # 螺杆半径
port_out_rad = 2.0447 # 输入输出端口统一半径
port_len = 2.0        # 端口延伸长度

# 中间步严格不进行四舍五入，保留全精度
Cx_cos = cx * np.cos(np.pi / 6.0)
Cx_sin = cx * np.sin(np.pi / 6.0)
Len = 2.0 * Cx_cos + fw

# 动态生成 7 个腔体的中心坐标
cavity_centers = []
for i in range(7):
    # 奇数腔体(Idx 0,2,4,6)的 X 为 0；偶数腔体(Idx 1,3,5)的 X 平移
    _xc = 0.0 if i % 2 == 0 else (Len * np.cos(np.pi / 6.0))
    _zc = Cx_cos + i * Len * np.sin(np.pi / 6.0)
    cavity_centers.append((_xc, _zc))


# ==================== 2. 几何掩膜辅助数学函数 ====================

def _get_hexagon_mask(X, Z, xc, zc, R):
    """计算正六边形腔体掩膜 (平坦边垂直于 Z 轴)"""
    dx = X - xc
    dz = Z - zc
    sin60 = np.sqrt(3.0) / 2.0
    cos60 = 0.5
    
    cond1 = np.abs(dz) <= (R * sin60)
    cond2 = np.abs(cos60 * dz + sin60 * dx) <= (R * sin60)
    cond3 = np.abs(-cos60 * dz + sin60 * dx) <= (R * sin60)
    return cond1 & cond2 & cond3

def _get_rotated_window_mask(X, Z, p1, p2, width, thickness):
    """计算倾斜矩形耦合窗掩膜"""
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
    
    返回:
        torch.Tensor: 形状为 [1, 1, 64, 64, 256] 的 float32 类型 PyTorch 张量
    """
    grid_3d = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    
    # --- 步骤 A: 绘制 7 个六角形空气腔体 ---
    for xc, zc in cavity_centers:
        hex_2d_mask = _get_hexagon_mask(X, Z, xc, zc, cx)
        hex_3d_mask = hex_2d_mask & (Y >= 0.0) & (Y <= ch)
        grid_3d[hex_3d_mask] = 1.0
        
    # --- 步骤 B: 绘制 6 个倾斜的耦合窗通道 ---
    cws = [F3_Cw12, F3_Cw23, F3_Cw34, F3_Cw34, F3_Cw23, F3_Cw12]
    for i in range(6):
        p1 = cavity_centers[i]
        p2 = cavity_centers[i+1]
        cw_width = cws[i]
        
        win_2d_mask = _get_rotated_window_mask(X, Z, p1, p2, cw_width, fw)
        win_3d_mask = win_2d_mask & (Y >= 0.0) & (Y <= ch)
        grid_3d[win_3d_mask] = 1.0

    # --- 步骤 C: 植入 7 个金属调谐螺杆 ---
    rls = [F3_Rl1, F3_Rl2, F3_Rl3, 4.52, F3_Rl3, F3_Rl2, F3_Rl1]
    for i, (xc, zc) in enumerate(cavity_centers):
        post_length = rls[i]
        post_2d_mask = ((X - xc)**2 + (Z - zc)**2) <= (rad_p**2)
        post_3d_mask = post_2d_mask & (Y >= (ch - post_length)) & (Y <= ch)
        grid_3d[post_3d_mask] = 0.0

    # --- 步骤 D: 绘制输入与输出端口 (对齐至腔体 Y=6.0 中心) ---
    port1_mask = ((X - cavity_centers[0][0])**2 + (Y - ch/2.0)**2 <= port_out_rad**2) & \
                 (Z >= (cavity_centers[0][1] - Cx_cos - port_len)) & (Z <= (cavity_centers[0][1] - Cx_cos))
    grid_3d[port1_mask] = 1.0
    
    port2_zc = cavity_centers[6][1] + Cx_cos
    port2_mask = ((X - cavity_centers[6][0])**2 + (Y - ch/2.0)**2 <= port_out_rad**2) & \
                 (Z >= port2_zc) & (Z <= (port2_zc + port_len))
    grid_3d[port2_mask] = 1.0

    tensor_5d = torch.from_numpy(grid_3d).unsqueeze(0).unsqueeze(0)
    return tensor_5d


# ==================== 4. 测试运行验证 ====================
if __name__ == "__main__":
    print("⏳ 正在测试生成单结构几何张量...")
    test_tensor = generate_geometry_tensor(
        F3_Cw12=4.2, F3_Cw23=3.8, F3_Cw34=3.5, 
        F3_Rl1=2.1, F3_Rl2=5.4, F3_Rl3=3.3
    )
    print("✅ 测试成功！")
    print(f"📊 生成的张量数据类型: {test_tensor.dtype}")
    print(f"📐 输出张量标准维度 (Batch, Channel, Nx, Ny, Nz): {list(test_tensor.shape)}")
    
    print("\n🔍 腔体中心点坐标校验:")
    for idx, (xc, zc) in enumerate(cavity_centers):
        print(f"  Cavity {idx+1}: X = {xc:.5f}, Z = {zc:.5f}")