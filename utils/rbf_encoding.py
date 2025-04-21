# utils/rbf_encoding.py
import torch
import torch.nn as nn

class RBFEncoding(nn.Module):
    def __init__(self, num_centers=16, cutoff=10.0):
        """
        参数:
            num_centers: RBF 中心个数
            cutoff: 距离截断值
        """
        super().__init__()
        self.num_centers = num_centers
        centers = torch.linspace(0, cutoff, num_centers)
        self.register_buffer('centers', centers)
        self.width = cutoff / num_centers

    def forward(self, distances):
        """
        参数:
            distances: [B, L, L]
        返回:
            rbf: [B, L, L, num_centers]
        """
        # distances: [B, L, L]
        rbf = torch.exp(-((distances.unsqueeze(-1) - self.centers)**2) / self.width)
        return rbf  # 返回 [B, L, L, num_centers]