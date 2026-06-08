# utils/fed_vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTAggregator(nn.Module):
    def __init__(
        self,
        feat_dim,                  # 特征维度（来自参数统计特征）
        num_clients=None,          # 客户端数量，用于构造每客户端一个编码器
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        temperature=0.7,
        avg_loss_weight=1.0,       # w_avg：平均监督损失权重
        self_loss_weight=0.2,      # w_self：自身保持损失权重
        ent_weight=1e-3,           # w_ent：注意力熵约束权重
        self_mix=True,
        use_personal_encoders=True # 是否启用“每客户端一个编码器”
    ):
        super().__init__()
        self.avg_loss_weight = float(avg_loss_weight)
        self.self_loss_weight = float(self_loss_weight)
        self.ent_weight = float(ent_weight)
        self.self_mix = self_mix

        self.num_clients = num_clients
        self.use_personal_encoders = use_personal_encoders

        # ========= 特征编码部分 =========
        if self.use_personal_encoders:
            assert self.num_clients is not None, \
                "use_personal_encoders=True 时必须指定 num_clients"

            # 为每个客户端分配一个独立编码器：F -> H
            self.client_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feat_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.num_clients)
            ])
            self.feat_proj = None
        else:
            # 退回到原来的共享编码器
            self.client_encoders = None
            self.feat_proj = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # ========= Transformer 编码客户端间的交互 =========
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                # batch_first=True
            ),
            num_layers=num_layers
        )

        # 生成注意力的 Q/K
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 自适应 self-mix 门控
        if self_mix:
            self.beta_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            self.beta_head = None

        # 可学习的温度，控制注意力的“锐度”
        self.temperature = nn.Parameter(torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, client_params, client_feats, avg_target=None, prior=None):
        """
        client_params: [N, D] 扁平参数
        client_feats:  [N, F] 每个客户端的统计特征（小维度）
        avg_target:    [N]    可选的加权 AVG 权重（如样本量），若为 None 则等权
        prior:         [N]    可选先验（如样本量），将作为 log-bias 加到注意力 logits 上

        return:
          - out_params: [N, D] 个性化聚合后的参数
          - attn:       [N, N] 行归一的注意力矩阵（第 i 行表示目标客户端 i 对源客户端 j 的权重）
          - losses:     dict(total, avg, self, entropy)
        """
        N, D = client_params.shape
        if self.num_clients is not None:
            assert N == self.num_clients, f"N mismatch: got {N}, expected {self.num_clients}"

        # ========= 特征编码：每个客户端单独编码后再堆叠 =========
        if self.use_personal_encoders and (self.client_encoders is not None):
            encoded = []
            for i in range(N):
                hi = self.client_encoders[i](client_feats[i])  # [H]
                encoded.append(hi)
            H = torch.stack(encoded, dim=0)  # [N, H]
        else:
            # 共享编码器
            H = self.feat_proj(client_feats)  # [N, H]

        # ========= Transformer 编码客户端间的交互 =========
        Z = self.encoder(H.unsqueeze(1).transpose(0, 1)).transpose(0, 1).squeeze(1)  # [N, H]

        # ========= 计算注意力 logits =========
        d = Z.size(-1)
        Q = self.q(Z)                               # [N, H]
        K = self.k(Z)                               # [N, H]
        scale = torch.sqrt(torch.tensor(d, dtype=torch.float32, device=Z.device))
        logits = (Q @ K.t()) / (scale * torch.clamp(self.temperature, min=1e-2))

        # 不允许自注意（avoid identity shortcut），强制跨客户端融合
        logits = logits.masked_fill(torch.eye(N, device=Z.device, dtype=torch.bool), float("-inf"))

        # 可选：加入先验（如样本量）作为 log-bias
        if prior is not None:
            prior = prior.float().clamp_min(1e-6)
            prior = prior / prior.sum()
            logits = logits + prior.log().unsqueeze(0)  # 广播到每一行

        # ========= 行归一注意力 =========
        attn = F.softmax(logits, dim=-1)  # [N, N]

        # ========= 个性化聚合 =========
        mixed = attn @ client_params      # [N, D]

        # 自适应 self-mix（在自己和邻居之间平衡）
        if self.self_mix and self.beta_head is not None:
            beta = torch.sigmoid(self.beta_head(Z))        # [N, 1], in (0,1)
            out_params = beta * client_params + (1.0 - beta) * mixed
        else:
            out_params = mixed

        # ========= 内部损失：稳定 vs 个性化 =========
        # 1) AVG 监督：聚合后参数的加权平均 ≈ 原参数的加权平均
        """
        if avg_target is not None:
           w = avg_target.to(client_params.device).float()
           w = w / torch.clamp(w.sum(), min=1e-8)            # [N]
        else:
            w = torch.full((N,), 1.0 / N, dtype=torch.float32, device=client_params.device)

        mean_src = (w.unsqueeze(0) @ client_params).squeeze(0)  # [D]
        mean_out = (w.unsqueeze(0) @ out_params).squeeze(0)     # [D]
        loss_avg = F.mse_loss(mean_out, mean_src)
        """
        # 1) 强 AVG 监督：所有客户端的聚合结果都被拉向“原参数的平均模型”
        src_avg = client_params.mean(dim=0)  # [D]
        src_avg_mat = src_avg.unsqueeze(0).expand_as(out_params)  # [N, D]
        loss_avg = F.mse_loss(out_params, src_avg_mat)
        # 2) 自保持：不要离自己太远
        loss_self = F.mse_loss(out_params, client_params)

        # 3) 注意力熵（越大越均匀，这里通常希望稍微小一点，让注意力更“个性化”）
        entropy = -(attn * (attn + 1e-12).log()).sum(dim=-1).mean()

        total = (
            self.avg_loss_weight * loss_avg +
            self.self_loss_weight * loss_self +
            self.ent_weight * entropy
        )

        return out_params, attn, {
            "total": total,
            "avg": loss_avg,
            "self": loss_self,
            "entropy": entropy
        }