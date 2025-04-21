import torch
import itertools

def build_lattice_matrix(lattice_params):
    """
    将6个晶格参数转换为晶格矩阵

    参数：
        lattice_params: [B, 6]
            前3个为 1/a, 1/b, 1/c（单位：1/Å）
            后3个为 α/180, β/180, γ/180

    返回：
        lattice: [B, 3, 3] 晶格矩阵，单位 Å
    """
    B = lattice_params.shape[0]
    inv_abc = lattice_params[:, :3]
    abc = 1.0 / inv_abc  # 还原为 a, b, c，单位 Å
    angles = lattice_params[:, 3:] * 180.0  # 恢复到度

    a = abc[:, 0]
    b = abc[:, 1]
    c = abc[:, 2]

    alpha = angles[:, 0]
    beta = angles[:, 1]
    gamma = angles[:, 2]

    alpha_rad = torch.deg2rad(alpha)
    beta_rad = torch.deg2rad(beta)
    gamma_rad = torch.deg2rad(gamma)

    cos_alpha = torch.cos(alpha_rad)
    cos_beta = torch.cos(beta_rad)
    cos_gamma = torch.cos(gamma_rad)
    sin_gamma = torch.sin(gamma_rad)

    # 构造晶格矩阵的列向量
    a_col = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=1)
    b_col = torch.stack([b*cos_gamma, b*sin_gamma, torch.zeros_like(b)], dim=1)
    
    # 计算c列
    temp = (cos_alpha - cos_beta*cos_gamma) / (sin_gamma + 1e-8)
    c_x = c * cos_beta
    c_y = c * temp
    c_z_sq = c**2 * (1 - cos_beta**2 - temp**2)
    c_z = torch.sqrt(torch.clamp(c_z_sq, min=1e-8))  # 防止负数

    c_col = torch.stack([c_x, c_y, c_z], dim=1)

    # 组合成3x3晶格矩阵
    lattice = torch.stack([a_col, b_col, c_col], dim=2)
    return lattice

def compute_relative_features(pos, lattice, mask):
    """
    计算考虑周期性边界的真实原子间距离

    参数:
        pos: 张量，形状 [B, L_atom, 3]，分数坐标
        lattice: 张量，形状 [B, 3, 3]，晶格矩阵
        mask: 张量，形状 [B, L_atom]，原子掩码

    返回:
        distances: [B, L_atom, L_atom] 欧氏距离
    """
    B, L, _ = pos.shape
    
    # 将分数坐标转换为笛卡尔坐标
    pos_cart = torch.einsum('bij,bjk->bik', pos, lattice)  # [B, L, 3]
    
    # 生成所有可能的平移向量 T
    shifts = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), 
                         device=pos.device, dtype=torch.float32)  # [27, 3]
    T = torch.einsum('ij,bjk->bik', shifts, lattice)  # [B, 27, 3]
    
    # 计算所有原子对在所有平移下的距离
    diff_cart = pos_cart.unsqueeze(2) - pos_cart.unsqueeze(1)  # [B, L, L, 3]
    diff_cart_all = diff_cart.unsqueeze(3) + T.unsqueeze(1).unsqueeze(2)  # [B, L, L, 27, 3]
    distances_all = torch.norm(diff_cart_all, dim=-1)  # [B, L, L, 27]
    
    # 选择每个原子对的最小距离
    distances, _ = torch.min(distances_all, dim=-1)  # [B, L, L]
    
    # 应用掩码，忽略填充原子
    mask_matrix = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, L, L]
    distances = distances * mask_matrix.float() + (1 - mask_matrix.float()) * 1e9
    
    return distances