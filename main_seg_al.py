# main_seg_al.py
import numpy as np
import argparse
import os
import time
import random
import logging
import sys
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model.unet2d import Unet2D as Model
from data.dataset import generate_dataset
from utils.fed_merge import FedAvg, FedUpdate, FedViT
from utils.utils import scoring_func
from utils.seg.val import val
from utils.seg.test import test

parser = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser.add_argument('--dataset', type=str, default='Polyp', help='dataset')

parser.add_argument('--fl_method', type=str, default='FedEvi', help='federated method')
parser.add_argument('--max_round', type=int, default=200, help='maximum round number to train')
parser.add_argument('--max_epoch', type=int, default=2, help='maximum epoch number to train')
parser.add_argument('--server_inner_epochs', type=int, default=10, help='server inner loop epochs per round')
parser.add_argument('--norm', type=str, default='bn', help='normalization type')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--deterministic', type=bool, default=False, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=3, help='random seed')

parser.add_argument('--kl_weight', type=float, default=0.01, help='edl kl weight')
parser.add_argument('--ratio', type=float, default=1.0, help='ratio')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--annealing_step', type=int, default=10, help='annealing_step')

parser.add_argument('--num_classes', type=int, default=2, help='class num')
# ViT 聚合器参数
parser.add_argument('--vit_avg_weight', type=float, default=1.0, help='weight for AVG supervision loss')
parser.add_argument('--vit_num_heads', type=int, default=4, help='number of attention heads in ViT aggregator')
parser.add_argument('--vit_hidden_dim', type=int, default=128, help='hidden dimension in ViT aggregator')

args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if __name__ == '__main__':
    # log
    localtime = time.localtime(time.time())
    ticks = '{:>02d}{:>02d}{:>02d}{:>02d}{:>02d}'.format(
        localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min, localtime.tm_sec
    )
    snapshot_path = "UNet_{}/{}_{}_{}/".format(args.dataset.lower(), args.dataset, args.fl_method, ticks)

    os.makedirs(snapshot_path, exist_ok=True)
    os.makedirs(os.path.join(snapshot_path, 'model'), exist_ok=True)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Outputs root (snapshot_path): {os.path.abspath(snapshot_path)}")
    logging.info(f"Log file: {os.path.abspath(os.path.join(snapshot_path, 'log.txt'))}")
    logging.info(f"TensorBoard dir: {os.path.abspath(os.path.join(snapshot_path, 'log'))}")
    logging.info(f"Test result file: {os.path.abspath(os.path.join(snapshot_path, 'global_test_result.txt'))}")

    # init
    dataset = args.dataset
    assert dataset in ['Polyp']
    fl_method = args.fl_method
    assert fl_method in ['FedEvi']

    batch_size = args.batch_size
    base_lr = args.base_lr
    max_round = args.max_round
    server_inner_epochs = args.server_inner_epochs
    norm = args.norm

    if fl_method == 'FedEvi':
        from utils.seg.train_fedevi import train
        bn = False
        norm = 'in'
        val_batch_size = 1

    if dataset == 'Polyp':
        c = 3
        client_num = 4

    logging.info(str(args))

    # random seed
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # local dataloaders and models
    local_models = []
    local_train_loaders, local_val_loaders, local_test_loaders = [], [], []
    train_num = []

    for client_idx in range(client_num):
        data_train, data_val, data_test = generate_dataset(dataset=dataset, fl_method=fl_method, client_idx=client_idx)
        train_num.append(len(data_train))
        print(f"Client {client_idx} 训练样本数: {len(data_train)}")

        train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset=data_val, batch_size=val_batch_size, shuffle=False, num_workers=2,
                                pin_memory=True)
        test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False, num_workers=2,
                                 pin_memory=True)

        local_train_loaders.append(train_loader)
        local_val_loaders.append(val_loader)
        local_test_loaders.append(test_loader)

        model = Model(c=c, num_classes=args.num_classes, norm=norm).cuda()
        local_models.append(model)

    writer = SummaryWriter(snapshot_path + '/log')

    result_after_avg = np.zeros(client_num)
    best_val = 9999.0
    best_dice = 0.0

    global_model = Model(c=c, num_classes=args.num_classes, norm=norm).cuda()

    local_optimizers, local_schedulers = [], []
    for client_idx in range(client_num):
        optimizer = torch.optim.Adam(local_models[client_idx].parameters(), lr=base_lr, betas=(0.9, 0.99),
                                     weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        local_optimizers.append(optimizer)
        local_schedulers.append(scheduler)

    train_num = np.array(train_num, dtype=float)
    # 作为 AVG 监督目标（1×N）：样本比例；若想等权，改成 np.ones(client_num)/client_num
    client_weight_base = train_num / np.sum(train_num)
    client_weight = client_weight_base.copy()

    with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
        print('train num: {}'.format(train_num.tolist()), file=f)
        print('init weight: {}'.format(client_weight.tolist()), file=f)

    u_dis = np.zeros((client_num, max_round))
    u_data = np.zeros((client_num, max_round))

    # 重置 FedViT 的静态变量
    if hasattr(FedViT, 'aggregator'):
        del FedViT.aggregator
    if hasattr(FedViT, 'optimizer'):
        del FedViT.optimizer

    for round_idx in range(max_round):
        logging.info("=" * 70)
        logging.info(f"[Round {round_idx + 1}/{max_round}] START")

        # 1) 分发全局模型到各客户端
        local_models = FedUpdate(global_model, local_models, bn=bn)

        # 2) 客户端本地训练（这里只训练）
        for cid in range(client_num):
            logging.info(f"[Client {cid}] Train start")
            local_models[cid] = train(
                round_idx=round_idx + 1,
                client_idx=cid,
                model=local_models[cid],
                dataloader=local_train_loaders[cid],
                optimizer=local_optimizers[cid],
                args=args
            )
            local_schedulers[cid].step()
            logging.info(f"[Client {cid}] Train done (lr={local_optimizers[cid].param_groups[0]['lr']:.2e})")

        # 3) 服务器内循环：聚合 + 优化聚合器（整轮执行一次）
        logging.info(f"[Server] Aggregation + optimizer loop (inner epochs={server_inner_epochs})")
        for server_epoch in range(server_inner_epochs):
            logging.info(f"[Server] Inner epoch {server_epoch + 1}/{server_inner_epochs}")

            avg_target_t = torch.tensor(client_weight_base, dtype=torch.float32,
                                        device=next(global_model.parameters()).device)

            # 只有最后一次回传（apply=True），前面几次只训练聚合器（apply=False）
            apply_now = (server_epoch == server_inner_epochs - 1)
            global_model, local_models, attn_weights, avg_loss, loss_dict = FedViT(
                global_model, local_models, args, bn=bn, avg_target=avg_target_t, apply=apply_now
            )

            # 打印三项分损到终端/日志
            step = round_idx * server_inner_epochs + server_epoch
            logging.info(
                f"[Server] inner {server_epoch + 1}/{server_inner_epochs} | "
                f"loss_total={avg_loss.item():.6f} | "
                f"loss_global={loss_dict['global'].item():.6f} | "
                f"loss_personal_avg={loss_dict['personal_avg'].item():.6f} | "
                f"loss_self={loss_dict['self'].item():.6f}"
            )

            # TensorBoard 记录
            writer.add_scalar('vit_loss/total',        avg_loss.item(),                  step)
            writer.add_scalar('vit_loss/global',       loss_dict['global'].item(),       step)
            writer.add_scalar('vit_loss/personal_avg', loss_dict['personal_avg'].item(), step)
            writer.add_scalar('vit_loss/self',         loss_dict['self'].item(),         step)

            # 优化聚合器
            FedViT.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(FedViT.aggregator.parameters(), max_norm=1.0)
            FedViT.optimizer.step()

        # 3.3（可选）基于不确定性的权重调整（不影响上面的监督）
        for cid in range(client_num):
            u_dis[cid, round_idx], u_data[cid, round_idx] = scoring_func(
                global_model, local_models[cid], local_val_loaders[cid], client_idx=cid, args=args
            )
        client_weight = attn_weights.detach().cpu().numpy()
        client_weight = np.clip(client_weight, a_min=1e-3, a_max=None)
        client_weight += args.ratio * u_dis[:, round_idx] / np.maximum(u_data[:, round_idx], 1e-8)
        client_weight /= client_weight.sum()
        client_weight = np.clip(client_weight, a_min=1e-3, a_max=None)

        # 4) 验证每个客户端（整轮只做一次），并打印一次平均
        for cid in range(client_num):
            result_after_avg[cid] = val(model=local_models[cid],
                                        dataloader=local_val_loaders[cid],
                                        args=args)
        avg_val = result_after_avg.mean()
        logging.info(f"Round {round_idx + 1} average val loss: {avg_val:.6f}")
        writer.add_scalar('val_loss', avg_val, round_idx)

        # 5) 若验证更好则测试并保存
        if avg_val < best_val:
            best_val = avg_val
            with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                print('FL round {}'.format(round_idx + 1), file=f)
                print('weight: {}'.format(client_weight.tolist()), file=f)

            avg_dice = 0.0
            avg_hd95 = 0.0
            logging.info("[Test] New best validation, running test ...")
            for cid in range(client_num):
                dice_score, hd95_score = test(
                    dataset=dataset, model=local_models[cid],
                    dataloader=local_test_loaders[cid], client_idx=cid, args=args
                )
                avg_dice += dice_score.mean()
                avg_hd95 += hd95_score.mean()
                logging.info(f"[Test] Client {cid} Dice={dice_score[0]:.5f} HD95={hd95_score[0]:.3f}")
                with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                    print('client {}.\tDice\t{:.5f}\tHD95\t{:.3f}'.format(cid, dice_score[0], hd95_score[0]), file=f)
            avg_dice /= client_num
            avg_hd95 /= client_num
            writer.add_scalar('avg_dice', avg_dice, round_idx)
            writer.add_scalar('avg_hd95', avg_hd95, round_idx)
            logging.info(f"[Test] Avg Dice={avg_dice:.5f} Avg HD95={avg_hd95:.3f}")

            # 保存模型（全局 + 个性化 + 聚合器）
            save_model_path = os.path.join(snapshot_path + '/model/best_seed{}_global.pth'.format(args.seed))
            torch.save(global_model.state_dict(), save_model_path)
            for cid in range(client_num):
                torch.save(local_models[cid].state_dict(),
                           os.path.join(snapshot_path + f'/model/best_seed{args.seed}_client{cid}.pth'))
            if hasattr(FedViT, 'aggregator'):
                torch.save(FedViT.aggregator.state_dict(),
                           os.path.join(snapshot_path + '/model/best_aggregator.pth'))

        logging.info(f"[Round {round_idx + 1}/{max_round}] END")

    writer.close()