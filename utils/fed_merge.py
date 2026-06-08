# utils/fed_merge.py
import torch
from .fed_vit import ViTAggregator


def dict_weight(dict1, weight):
    for k, v in dict1.items():
        dict1[k] = weight * v
    return dict1


def dict_add(dict1, dict2):
    for k, v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1


def FedAvg(global_model, local_models, client_weight, bn=True):
    new_model_dict = None

    if bn:
        for client_idx in range(len(local_models)):
            local_dict = local_models[client_idx].state_dict()
            if new_model_dict is None:  # init
                new_model_dict = dict_weight(local_dict, client_weight[client_idx])
            else:
                new_model_dict = dict_add(new_model_dict, dict_weight(local_dict, client_weight[client_idx]))
        global_model.load_state_dict(new_model_dict)

    else:
        for key in global_model.state_dict().keys():
            if 'bn' not in key:
                temp = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(len(client_weight)):
                    temp += client_weight[client_idx] * local_models[client_idx].state_dict()[key]
                global_model.state_dict()[key].data.copy_(temp)

    return global_model


def FedUpdate(global_model, local_models, bn=True):
    if bn:
        global_dict = global_model.state_dict()
        for client_idx in range(len(local_models)):
            local_models[client_idx].load_state_dict(global_dict)
    else:
        for key in global_model.state_dict().keys():
            if 'bn' not in key:
                for client_idx in range(len(local_models)):
                    local_models[client_idx].state_dict()[key].data.copy_(global_model.state_dict()[key])
    return local_models


def _extract_client_feats(local_models, param_names, device):
    """
    用轻量统计特征来表征每个客户端的模型。
    返回: [N, F]，F = 4 * len(param_names)（mean/std/L1/L2）
    """
    feats = []
    for model in local_models:
        s = model.state_dict()
        per_layer_stats = []
        for k in param_names:
            v = s[k].float().to(device).view(-1)
            if v.numel() == 0:
                per_layer_stats.extend([torch.tensor(0., device=device)] * 4)
                continue
            mean = v.mean()
            std = v.std()
            l1 = v.abs().mean()
            l2 = v.norm(p=2) / torch.sqrt(torch.tensor(v.numel(), device=device, dtype=torch.float32) + 1e-8)
            per_layer_stats.extend([mean, std, l1, l2])
        feats.append(torch.stack(per_layer_stats))
    feats = torch.stack(feats)  # [N, F]
    return feats


def FedViT(global_model, local_models, args, bn=True, avg_target=None, apply=True):
    """
    单次 ViT 聚合步骤（使用 AVG + self + entropy 监督）：
      - 从 local_models 抽取展平参数矩阵 P: [N, D]；
      - 计算统计特征 feats: [N, F]；
      - 前向：out_personal = A @ P，经 beta self-mix 得到 out_params；
      - ViTAggregator 内部监督：
          * loss_avg: 加权平均前后的一致性（AVG 监督）
          * loss_self: 聚合结果不要离自己太远
          * entropy: 注意力熵正则
      - 若 apply=True:
          * 将 out_params[i] 回传写回第 i 个客户端模型；
          * 全局模型更新为 out_params 的简单平均；
      - 返回:
          global_model, local_models, attn_weights(列平均), avg_loss(=total), losses(dict)。
    """
    device = next(global_model.parameters()).device

    # 1) 选择需要聚合的参数
    param_names = [k for k in global_model.state_dict().keys() if ('bn' not in k or bn)]

    # 2) 扁平化参数 [N, D]
    client_params = []
    for m in local_models:
        p = torch.cat([m.state_dict()[k].flatten().to(device) for k in param_names])
        client_params.append(p)
    client_params = torch.stack(client_params).to(device)  # [N, D]
    N, D = client_params.shape

    # 3) 统计特征 [N, F]
    client_feats = _extract_client_feats(local_models, param_names, device)
    feat_dim = client_feats.shape[1]

    # 4) 初始化聚合器
    if not hasattr(FedViT, 'aggregator'):
        FedViT.aggregator = ViTAggregator(
            feat_dim=feat_dim,
            num_clients=N,
            num_heads=args.vit_num_heads,
            hidden_dim=args.vit_hidden_dim,
            num_layers=2,
            temperature=0.7,
            avg_loss_weight=args.vit_avg_weight,  # w_avg
            self_loss_weight=0.2,                 # 可按需调节
            ent_weight=1e-3,                      # 可按需调节
            self_mix=True,
            use_personal_encoders=True
        ).to(device)
        FedViT.optimizer = torch.optim.Adam(FedViT.aggregator.parameters(), lr=1e-4)

    aggregator = FedViT.aggregator

    # 5) 前向：得到 out_personal、attn、losses（total/avg/self/entropy）
    out_personal, attn_mat, losses = aggregator(
        client_params=client_params,
        client_feats=client_feats,
        avg_target=avg_target  # 可选：用于加权平均
    )
    avg_loss = losses["total"]

    # 6) 若需要，将个性化聚合后的参数回传 + 更新全局模型
    if apply:
        # 回传到各自客户端模型
        for i in range(N):
            local_dict = local_models[i].state_dict()
            ptr = 0
            for k in param_names:
                numel = local_dict[k].numel()
                local_dict[k].data.copy_(out_personal[i, ptr:ptr + numel].view_as(local_dict[k]))
                ptr += numel
            local_models[i].load_state_dict(local_dict)

        # 全局模型更新为 out_personal 的简单平均
        avg_param = out_personal.mean(dim=0)
        gdict = global_model.state_dict()
        ptr = 0
        for k in param_names:
            numel = gdict[k].numel()
            gdict[k].data.copy_(avg_param[ptr:ptr + numel].view_as(gdict[k]))
            ptr += numel
        global_model.load_state_dict(gdict)

    # 7) 从注意力矩阵得到列平均贡献度（每个源客户端的平均贡献）
    attn_weights = attn_mat.mean(dim=0).detach()  # [N]

    return global_model, local_models, attn_weights, avg_loss, losses