import numpy as np
from torch.nn import functional as F
from .utils import PositionWiseFeedForward, save_freq
import torch
from torch import nn
from .attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, m=0):

        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        if m == 0:
            att = self.lnorm(queries + self.dropout(att))
        elif m == 1:
            att = self.lnorm1(queries + self.dropout(att))
        else:  # m == 2 (detect特征)
            att = self.lnorm2(queries + self.dropout(att))  # 暂时与pixel共用lnorm1，也可以添加新的lnorm2
        ff = self.pwff(att, m)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, Lf, T, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.N = N
        self.Lf = Lf
        self.T = T
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.fusion_gate = nn.Linear(3 * d_model, 3)  # 输入维度需要是3倍特征维度
        nn.init.xavier_uniform_(self.fusion_gate.weight)

        self.projection = nn.Linear(d_model, 64)  # d_in是输入维度
        self.pooler = nn.Linear(d_model, d_model)  # 可学习的特征聚合
        nn.init.orthogonal_(self.pooler.weight)  # 正交初始化提升多样性
        self.temperature = nn.Parameter(torch.tensor([1.0]))

        self.padding_idx = padding_idx

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def criterion(self, out_1, out_2, tau_plus, beta):
        # neg score
        batch_size = out_1.size(0)
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = self.get_negative_mask(batch_size).to(out_1.device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        N = batch_size * 2 - 2
        imp = (beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()

        return loss

    def forward(self, input, pixel, detect, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        pixel_attention_mask = (torch.sum(pixel, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        detect_attention_mask = (torch.sum(detect, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # 为detect添加注意力掩码
        mix_attention_mask = (torch.sum(detect, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
        out = input
        out1 = pixel
        out2 = detect

        x1 = input
        x2 = pixel
        x3 = detect
        x_all = x1+x2+x3

        for i, l in enumerate(self.layers):
            if i < self.Lf:
                for t in range(self.T):
                    out = l(out, out, out, attention_mask, attention_weights, m=0)
                    out1 = l(out1, out1, out1, pixel_attention_mask, attention_weights, m=1)
                    out2 = l(out2, out2, out2, detect_attention_mask, attention_weights, m=2)

                x1 = l(x1, x_all, x_all, mix_attention_mask, attention_weights, m=0)
                x2 = l(x2, x_all, x_all, mix_attention_mask, attention_weights, m=0)
                x3 = l(x3, x_all, x_all, mix_attention_mask, attention_weights, m=0)

            elif i == self.Lf:
                combined = torch.cat([out, out1, out2], dim=-1)  # 拼接特征
                gate = torch.softmax(self.fusion_gate(combined), dim=-1)  # 形状(b_s, seq_len, 3)
                # 分解门控权重到各个模态
                gate1 = gate[..., 0].unsqueeze(-1)
                gate2 = gate[..., 1].unsqueeze(-1)
                gate3 = gate[..., 2].unsqueeze(-1)

                # 加权融合（保持梯度流）
                feat1 = gate1 * out + gate2 * out1 + gate3 * out2
                # 添加残差连接
                feat2 = feat1 + x_all  # 使用早期融合特征作为残差
                out = l(feat2, feat2, feat2, attention_mask, attention_weights, m=0)
            else:
                out = l(out, out, out, attention_mask, attention_weights, m=0)

        # 在特征计算后添加对比学习
        z1 = F.normalize(self.projection(x1.mean(1)), dim=-1)  # 添加pooler层
        z2 = F.normalize(self.projection(x2.mean(1)), dim=-1)
        z3 = F.normalize(self.projection(x3.mean(1)), dim=-1)
        # 计算三组对比损失
        contrastive_loss = (self.criterion(z1, z2, 0.1, 0.5)+
                            self.criterion(z1, z3, 0.1, 0.5)+
                            self.criterion(z2, z3, 0.1, 0.5))/3

        return out, attention_mask, contrastive_loss


class DifnetEncoder(MultiLevelEncoder):
    def __init__(self, Lf, T, N, padding_idx, d_in=2048, **kwargs):
        super(DifnetEncoder, self).__init__(Lf, T, N, padding_idx, **kwargs)

    def forward(self, input, pixel, detect, attention_weights=None):

        return super(DifnetEncoder, self).forward(input, pixel, detect, attention_weights=attention_weights)
