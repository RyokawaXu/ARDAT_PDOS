# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from utils.atom_feature import AtomFeatureEncoder
from utils.relative_features import build_lattice_matrix, compute_relative_features
from utils.rbf_encoding import RBFEncoding

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 使用 GELU 激活函数代替 ReLU
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
        
class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, kernel_size=3, padding=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # 添加卷积层
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,  # 修改为更小的卷积核大小
                    padding=padding
                )
            )

            # 非最后一层添加GELU和Dropout
            if i < num_layers - 1:
                self.layers.append(nn.GELU())    # 替换ReLU
                self.layers.append(nn.Dropout(0.1))  # 新增Dropout

    def forward(self, x):
        # 输入形状: [B, D, L]
        for layer in self.layers:
            x = layer(x)
        return x  # 输出形状: [B, output_dim, L]


class Transformer(nn.Module):

    def __init__(self, token_num=100, d_model=512, nhead=8, dos_num=40, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False):
        super().__init__()
        # 创建一个可学习的比例参数，初始值设置为 0.7
        #self.alpha_lattice = nn.Parameter(torch.tensor(0.5))  # 融合比例可学习参数
        
        # 初始化 RBF 编码器，num_centers 可根据需要设置
        self.rbf_encoder = RBFEncoding(num_centers=64, cutoff=10)
        # 当构造编码器层时，将 rbf_encoder 传递进去
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                rbf_encoder=self.rbf_encoder)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.query_embed = nn.Parameter(torch.zeros(dos_num, d_model))
        self.tgt = nn.Parameter(torch.zeros(dos_num, d_model))
        self.tok_emb = nn.Embedding(token_num, d_model//2)
        
        #self.out_embed = MLP(d_model, d_model, 1, 5)
        self.out_embed = CNN(d_model, d_model*3, output_dim=1, num_layers=4)
        #self.out_embed = nn.Sequential(
            #CNN(d_model, d_model*3, output_dim=1, num_layers=6),
            #nn.Sigmoid()  # 强制输出在[0,1]
        #)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.atom_fea_encoder = AtomFeatureEncoder(3, d_model//2)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, src, mask, pos):
        """
        参数：
          src: [B, L]，包含晶格信息和原子信息的 token 序列（前两个 token为晶格信息）
          mask: [B, L]，对应的掩码
          pos: [B, L, 3]，其中 pos[:, :2, :] 为晶格参数，pos[:, 2:, :] 为原子坐标
        """
        B = pos.shape[0]
        
        # 1. 提取晶格信息
        lattice_params = pos[:, :2, :].reshape(B, 6)  # [B, 6]
        lattice = build_lattice_matrix(lattice_params)  # [B, 3, 3]

        # 2. 移除晶格 token，保留原子部分
        src = src[:, 2:]        # [B, L-2]：原子信息（去除晶格信息）
        pos = pos[:, 2:, :]  # [B, L-2, 3]：原子坐标（从分数坐标提取）
    
        mask = mask[:, 2:]      # [B, L-2]：掩码
        B, L0, _ = pos.shape    # L0 = 原子数
        
        # 3. 原子 token 嵌入
        src_data = self.tok_emb(src)        # [B, L0, d_model/2]
        atom_fea = self.atom_fea_encoder(src)         # [B, L0, d_model/2]
        atom_src = torch.cat([src_data, atom_fea], dim=-1)  # [B, L0, d_model]
        
        # 4. 计算相对特征（使用周期性边界条件）
        #distances = compute_relative_features_pbc(pos, lattice)  # [B, L0, L0]
        distances = compute_relative_features(pos, lattice, mask)  # [B, L0, L0]
        rel_features = self.rbf_encoder(distances)  # [B, L0, L0, num_centers]
        '''
        import pdb; pdb.set_trace()
        '''

        # 5. 编码器：包含 global token 的输入序列
        memory = self.encoder(
            src=atom_src,
            src_key_padding_mask=mask,
            pos=pos,
            rel_features=rel_features
        )  # [B, L0, d_model]
        
        # 6. 解码器：使用 memory 作为上下文进行预测
        query_embed = self.query_embed.unsqueeze(0).repeat(B, 1, 1)  # [B, dos_num, d_model]
        tgt = self.tgt.unsqueeze(0).repeat(B, 1, 1)                  # [B, dos_num, d_model]
        hs, attention_v = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos, 
            query_pos=query_embed
        )
        
        # 7. 输出：经过 CNN decoder 得到 DOS 序列
        hs = hs.permute(0, 2, 1)         # [B, d_model, dos_num]
        res = self.out_embed(hs)         # [B, 1, dos_num]
        res = res.permute(0, 2, 1)       # [B, dos_num, 1]
    
        return res, attention_v

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            rel_features=None):
        
        output = src
    
        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos,
                           rel_features=rel_features)
        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output, attention = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="leaky_relu", normalize_before=False, rbf_encoder=None):
        super().__init__()
        self.activation = _get_activation_fn(activation)
        self.dim = d_model
        self.nhead = nhead
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 用于前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 保存 RBF 编码器，用于获取相对特征的维度信息
        self.rbf_encoder = rbf_encoder
        self.rel_proj = nn.Linear(self.rbf_encoder.num_centers, d_model)
        self.alpha_rel = nn.Parameter(torch.tensor(1.0))

    def forward_post(self, src, src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None,
                     rel_features=None):
        """
        参数:
            src: [B, L, d_model]
            rel_features: [B, L, L, num_centers]，预计算的 RBF 编码
        """
        B, L, _ = src.size()
        q, k, v = src, src, src
        attn_output, attn_weights = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        

        if rel_features is not None:
            
            # 1) 映射相对特征
            rel_emb = self.rel_proj(rel_features)  # [B, L, L, d_model]
            d_k = self.dim // self.nhead
            rel_emb = rel_emb.view(B, L, L, self.nhead, d_k)
            
            # 2) 拆头
            q_heads = q.view(B, L, self.nhead, d_k)
            
            # 3) 计算 rel_scores
            rel_scores = (q_heads.unsqueeze(2) * rel_emb).sum(-1)  # [B, L, L, nhead]
            rel_scores = rel_scores.permute(0,3,1,2).reshape(B*self.nhead, L, L)
 
            # 4) 重算 base_scores 并对两部分都做 √d_k 缩放
            q_heads2 = q.view(B, L, self.nhead, d_k).permute(0,2,1,3).reshape(B*self.nhead, L, d_k)
            k_heads  = k.view(B, L, self.nhead, d_k).permute(0,2,1,3).reshape(B*self.nhead, L, d_k)
            base_scores = torch.bmm(q_heads2, k_heads.transpose(1,2)) / math.sqrt(d_k)
            rel_scores  = (self.alpha_rel * rel_scores) / math.sqrt(d_k)
            alpha = self.alpha_rel
            total_scores = (base_scores + alpha * rel_scores) / math.sqrt(1.0 + alpha * alpha)

            # 5) softmax + 重算 attention output
            attn_weights_new = F.softmax(total_scores, dim=-1)
            v_heads = v.view(B, L, self.nhead, d_k).permute(0, 2, 1, 3).reshape(B * self.nhead, L, d_k)
            attn_output_new = torch.bmm(attn_weights_new, v_heads)
            attn_output_new = attn_output_new.view(B, self.nhead, L, d_k).permute(0, 2, 1, 3).reshape(B, L, self.dim)
            src2 = attn_output_new
        else:
            src2 = attn_output

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None, rel_features=None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, rel_features)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = tgt
        k = tgt
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        q = tgt
        k = memory
        tgt2, attention_v = self.multihead_attn(query=q, key=k, value=memory,
                                                attn_mask=memory_mask,
                                                key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attention_v

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)  # 添加 LeakyReLU 支持
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
