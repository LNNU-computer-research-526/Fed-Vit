from glob import glob
import json
import numpy as np
import os


def split_prostate():
    if not os.path.exists('data_split/Polyp/'):
        os.makedirs('data_split/Polyp/')

    for i in range(6):
        # 读取数据路径（确保路径正确）
        data_list = glob(
            'C:/Users/Administrator/Desktop/FedEvi-master/Dataset/Polyp_npy/client{}/data_npy/*'.format(i + 1))
        np.random.shuffle(data_list)
        data_len = len(data_list)
        print(f"客户端{i + 1}原始数据量：{data_len}")  # 新增：打印原始数据量，验证是否正确读取

        # 测试集：总数据的20%（向上取整，避免0）
        test_len = int(np.ceil(0.2 * data_len))
        test_list = data_list[:test_len]
        train_val_list = data_list[test_len:]  # 剩余80%用于训练+验证

        # 验证集：剩余数据的12.5%（即总数据的10%）
        val_len = int(np.ceil(0.125 * len(train_val_list)))
        val_list = train_val_list[:val_len]
        train_list = train_val_list[val_len:]

        # 处理切片文件（如果是按case划分后再取切片）
        train_slice_list = []
        for train_case in train_list:
            train_slice_list.extend(glob('{}/*'.format(train_case)))
        val_slice_list = []
        for val_case in val_list:
            val_slice_list.extend(glob('{}/*'.format(val_case)))
        test_slice_list = []
        for test_case in test_list:
            test_slice_list.extend(glob('{}/*'.format(test_case)))

        # 打印顺序：训练集 → 验证集 → 测试集（符合常规逻辑）
        print(
            f"客户端{i + 1}划分后：训练集{len(train_slice_list)}，验证集{len(val_slice_list)}，测试集{len(test_slice_list)}")

        # 保存文件
        with open("data_split/Polyp/client{}_train.txt".format(i + 1), "w") as f:
            json.dump(train_slice_list, f)
        with open("data_split/Polyp/client{}_val.txt".format(i + 1), "w") as f:
            json.dump(val_slice_list, f)
        with open("data_split/Polyp/client{}_test.txt".format(i + 1), "w") as f:
            json.dump(test_slice_list, f)


# split_prostate()

def split_dataset(dataset, client_num):
    if not os.path.exists('data_split/{}'.format(dataset)):
        os.makedirs('data_split/{}'.format(dataset))

    for i in range(client_num):
        # 读取数据路径（确认路径中的client编号与实际文件夹一致）
        data_list = glob(
            'C:/Users/Administrator/Desktop/FedEvi-master/Dataset/{}_npy/client{}/data_npy/*'.format(dataset, i + 1))
        np.random.shuffle(data_list)
        data_len = len(data_list)
        print(f"客户端{i + 1}原始数据量：{data_len}")  # 新增：验证是否读取到正确数量的原始数据

        # 测试集：20% of 总数据
        test_len = int(np.ceil(0.2 * data_len))
        test_list = data_list[:test_len]
        train_val_list = data_list[test_len:]  # 剩余80%

        # 验证集：12.5% of 剩余数据（即总数据的10%）
        val_len = int(np.ceil(0.125 * len(train_val_list)))
        val_list = train_val_list[:val_len]
        train_list = train_val_list[val_len:]

        # 修正打印顺序：训练集 → 验证集 → 测试集
        print(f"客户端{i + 1}划分后：训练集{len(train_list)}，验证集{len(val_list)}，测试集{len(test_list)}")

        # 保存文件
        with open("data_split/{}/client{}_train.txt".format(dataset, i + 1), "w") as f:
            json.dump(train_list, f)
        with open("data_split/{}/client{}_val.txt".format(dataset, i + 1), "w") as f:
            json.dump(val_list, f)
        with open("data_split/{}/client{}_test.txt".format(dataset, i + 1), "w") as f:
            json.dump(test_list, f)


# 运行时指定数据集和客户端数量（确保client_num与实际文件夹数量一致）
split_dataset(dataset='Polyp', client_num=4)