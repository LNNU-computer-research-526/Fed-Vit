import numpy as np
import os
import torch
from data.dataset import generate_dataset  # 导入数据集生成函数
from datetime import datetime

# -------------------------- 配置参数（适配“元组嵌套字典”格式） --------------------------
# 目标保存路径
TARGET_SAVE_DIR = r"C:\Users\Administrator\Desktop\FedEvi-master\UNet_polyp"
DATASET = "Polyp"
FL_METHOD = "FedEvi"
CLIENT_NUM = 4

# 1. 元组中“数据字典”的索引（样本格式：(样本ID, 数据字典) → 索引1）
DATA_DICT_INDEX_IN_TUPLE = 1
# 2. 数据字典中“标签”的键名（需根据实际字典键调整，默认"label"）
LABEL_KEY_IN_DICT = "label"


# ---------------------------------------------------------------------------------------------

def count_label_distribution(dataset, dataset_type, client_idx):
    """统计标签分布（适配“元组嵌套字典”格式：(样本ID, {'image': ..., 'label': ...})）"""
    label_count = {}  # 类别 → 总像素数
    sample_with_label = {}  # 类别 → 包含该类别的样本索引列表

    for sample_idx, sample in enumerate(dataset):
        # -------------------------- 核心修改：分步提取标签 --------------------------
        # 步骤1：从元组中提取“数据字典”（索引1）
        data_dict = sample[DATA_DICT_INDEX_IN_TUPLE]
        # 验证数据字典是否包含标签键
        if LABEL_KEY_IN_DICT not in data_dict:
            raise KeyError(f"数据字典中缺少标签键 '{LABEL_KEY_IN_DICT}'，请检查键名！")

        # 步骤2：从数据字典中提取标签张量
        label = data_dict[LABEL_KEY_IN_DICT]
        # 步骤3：标签格式统一（转为Tensor，移除多余维度）
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)  # 无需强制GPU，避免设备不匹配
        if label.dim() > 2:
            label = label.squeeze()  # 如 [1,H,W] → [H,W]
        # -------------------------------------------------------------------------------------

        # 1. 统计各类别像素数（分割任务核心）
        unique_cls, cls_counts = torch.unique(label, return_counts=True)
        for cls, cnt in zip(unique_cls, cls_counts):
            cls_int = int(cls)  # 类别转为整数（避免Tensor作为字典键）
            if cls_int not in label_count:
                label_count[cls_int] = 0
            label_count[cls_int] += cnt.item()  # 累加像素数（转Python数值）

        # 2. 统计包含各类别的样本数（按样本统计）
        for cls in unique_cls:
            cls_int = int(cls)
            if cls_int not in sample_with_label:
                sample_with_label[cls_int] = []
            if sample_idx not in sample_with_label[cls_int]:
                sample_with_label[cls_int].append(sample_idx)

    # 基础统计量计算（避免除以0）
    total_samples = len(dataset)
    total_pixels = sum(label_count.values()) if label_count else 0
    sample_count_per_cls = {k: len(v) for k, v in sample_with_label.items()}

    return {
        "client_idx": client_idx,
        "dataset_type": dataset_type,
        "total_samples": total_samples,
        "total_pixels": total_pixels,
        "label_pixel_count": label_count,
        "label_pixel_ratio": {k: v / total_pixels for k, v in label_count.items()} if total_pixels != 0 else {},
        "sample_with_label_count": sample_count_per_cls,
        "sample_with_label_ratio": {k: v / total_samples for k, v in
                                    sample_count_per_cls.items()} if total_samples != 0 else {}
    }


def write_dist_to_file(all_distributions, save_dir):
    """写入统计结果到目标路径"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"已确认/创建保存目录：{save_dir}")

    # 带时间戳的文件名（避免覆盖）
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file_name = f"polyp_label_dist_{time_str}.txt"
    save_file_path = os.path.join(save_dir, save_file_name)

    # 写入结果（utf-8编码防乱码）
    with open(save_file_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + " 客户端样本标签分布统计报告 " + "=" * 60 + "\n")
        f.write(f"数据集：{DATASET}\n")
        f.write(f"联邦方法：{FL_METHOD}\n")
        f.write(f"统计时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本格式：元组嵌套字典 → (样本ID, {{'image': ..., '{LABEL_KEY_IN_DICT}': ...}})\n")
        f.write(f"结果保存路径：{save_file_path}\n")
        f.write("=" * 130 + "\n\n")

        # 按客户端分组输出
        for client_idx in range(CLIENT_NUM):
            f.write(f"【客户端 {client_idx}】\n")
            client_dists = [d for d in all_distributions if d["client_idx"] == client_idx]
            for dist in client_dists:
                f.write(f"  数据类型：{dist['dataset_type']} 集\n")
                f.write(f"    - 总样本数：{dist['total_samples']} 个\n")
                f.write(f"    - 总像素数：{dist['total_pixels']:,} 个（千位分隔）\n")

                # 像素分布
                f.write(f"    - 类别像素分布：\n")
                if dist["label_pixel_count"]:
                    for cls in sorted(dist["label_pixel_count"].keys()):
                        cnt = dist["label_pixel_count"][cls]
                        ratio = dist["label_pixel_ratio"][cls]
                        f.write(f"      类别 {cls}：{cnt:,} 像素（占比 {ratio:.4f}）\n")
                else:
                    f.write(f"      无有效标签数据\n")

                # 样本分布
                f.write(f"    - 含类别样本分布：\n")
                if dist["sample_with_label_count"]:
                    for cls in sorted(dist["sample_with_label_count"].keys()):
                        cnt = dist["sample_with_label_count"][cls]
                        ratio = dist["sample_with_label_ratio"][cls]
                        f.write(f"      类别 {cls}：{cnt} 样本（占比 {ratio:.4f}）\n")
                else:
                    f.write(f"      无有效样本标签数据\n")

                f.write("\n")
            f.write("-" * 120 + "\n\n")

    return save_file_path


if __name__ == "__main__":
    print("=" * 50 + " 客户端标签分布统计工具 " + "=" * 50 + "\n")
    print(f"目标保存路径：{TARGET_SAVE_DIR}")
    print(f"样本格式配置：")
    print(f"  1. 元组中数据字典的索引：{DATA_DICT_INDEX_IN_TUPLE}（当前样本：(ID, 数据字典)）")
    print(f"  2. 数据字典中标签的键名：{LABEL_KEY_IN_DICT}\n")

    # 收集所有客户端的标签分布
    all_label_dists = []
    for client_idx in range(CLIENT_NUM):
        try:
            print(f"正在处理客户端 {client_idx}...")
            # 1. 加载数据集
            data_train, data_val, data_test = generate_dataset(
                dataset=DATASET,
                fl_method=FL_METHOD,
                client_idx=client_idx
            )
            print(f"  数据集加载成功：train({len(data_train)}), val({len(data_val)}), test({len(data_test)})")

            # 2. 打印样本结构详情（帮助确认配置）
            sample_example = data_train[0]
            data_dict_example = sample_example[DATA_DICT_INDEX_IN_TUPLE]
            print(f"  样本结构详情：")
            print(f"    - 元组长度：{len(sample_example)} → 索引0（样本ID）：{type(sample_example[0])}")
            print(f"    - 数据字典包含的键：{list(data_dict_example.keys())}")
            print(f"    - 标签（{LABEL_KEY_IN_DICT}）类型：{type(data_dict_example[LABEL_KEY_IN_DICT])}")
            print(
                f"    - 标签形状：{data_dict_example[LABEL_KEY_IN_DICT].shape if hasattr(data_dict_example[LABEL_KEY_IN_DICT], 'shape') else '无shape'}")

            # 3. 统计标签分布
            train_dist = count_label_distribution(data_train, "train", client_idx)
            val_dist = count_label_distribution(data_val, "val", client_idx)
            test_dist = count_label_distribution(data_test, "test", client_idx)
            all_label_dists.extend([train_dist, val_dist, test_dist])
            print(f"  客户端 {client_idx} 统计完成！\n")

        except Exception as e:
            print(f"  客户端 {client_idx} 统计失败：{str(e)}\n")
            continue

    # 4. 写入结果文件
    if all_label_dists:
        save_path = write_dist_to_file(all_label_dists, TARGET_SAVE_DIR)
        print("=" * 50 + " 统计完成 " + "=" * 50)
        print(f"✅ 结果已保存至：{save_path}")
        # 预览前8行
        print("\n📄 结果预览（前8行）：")
        with open(save_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 8:
                    print(line.strip())
                else:
                    print("...（其余内容省略）")
                    break
    else:
        print("❌ 未收集到数据，请优先检查：")
        print(f"  1. 数据字典的键是否包含 '{LABEL_KEY_IN_DICT}'（当前配置）；")
        print(f"  2. 标签是否为Tensor/numpy数组（可转换为整数类别）；")
        print(f"  3. 标签是否为2D格式（如 [H,W]，无通道维度）。")