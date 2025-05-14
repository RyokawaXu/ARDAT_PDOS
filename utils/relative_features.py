# utils/relative_features.py
import torch

def compute_relative_features(pos):
    """
    计算原子对之间的相对特征

    参数:
        pos: 张量，形状 [B, L_atom, 3]，原子坐标

    返回:
        distances: [B, L_atom, L_atom] 欧氏距离
        direction: [B, L_atom, L_atom, 3] 差值归一化后的方向
    """
    B, L, _ = pos.shape
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, L, L, 3]
    distances = torch.norm(diff, dim=-1)         # [B, L, L]
    direction = diff / (distances.unsqueeze(-1) + 1e-8)
    return distances, direction
