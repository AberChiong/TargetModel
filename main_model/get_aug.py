# 为了进行模型的基准性测试
# 此版本更新了分别储存XP和XL，并将储存格式设置为dataset。
# 作者：常怡彬 
# 最后更新：2023年10月16日
# 更改为PDBbind2020数据库
# 最后更新：2023年11月29日

import os
import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import rotate

# 获取x和y
def get_xy(file_path, dataset):
    with h5py.File(file_path, 'r') as f:
        dataxp = f[dataset+'_xp'][:]
        dataxl = f[dataset+'_xl'][:]
        datay = f[dataset+'_y'][:]
        f.close()
    return dataxp, dataxl, datay

# 定义单次个观察面的四次旋转
def ones_rotate(data):
    data_ones = data
    arrg = [90, 180, 270]
    axes = (1, 3)
    for i in arrg:
        data_one = rotate(data, angle=i, axes=axes, order=0)
        data_ones= np.concatenate([data_ones, data_one], axis=0)
    return data_ones

# 张量转换并保存
def tensor_save(xp, xl, y, PATH):
    xp = torch.from_numpy(xp)
    xl = torch.from_numpy(xl)
    y = torch.from_numpy(np.tile(y, 4))
    dataset = TensorDataset(xp, xl, y)
    torch.save(dataset, PATH)


# 定义主要扩增程序
def data_augmentation(path_dir, from_name, data_type):
    path_dir = path_dir
    file_path = os.path.join(path_dir, from_name)
    dataset_name = data_type

    dataxp, dataxl, datay = get_xy(file_path, dataset_name)

    j = 0
    save_path = os.path.join(path_dir, dataset_name+str(j)+'.pt')

    # 0号观察面旋转
    dataxp_0 = ones_rotate(dataxp)
    dataxl_0 = ones_rotate(dataxl)
    tensor_save(dataxp_0, dataxl_0, datay, save_path)

    def ones(dataxp, dataxl, datay, axes, save_path):
        dataxp_1 = rotate(dataxp, angle=i, axes=axes, order=0)
        dataxp_1 = ones_rotate(dataxp)
        dataxl_1 = rotate(dataxl, angle=i, axes=axes, order=0)
        dataxl_1 = ones_rotate(dataxl)
        tensor_save(dataxp_1, dataxl_1, datay, save_path)

    # 基于0号，左、右、后三个观察面
    for i in [90, 180, 270]:
        j += 1
        save_path = os.path.join(path_dir, dataset_name+str(j)+'.pt')
        ones(dataxp, dataxl, datay, axes=(1, 2), save_path=save_path)

    # 基于0号，上下两个观察面
    for i in [-90, 90]:
        j += 1
        save_path = os.path.join(path_dir, dataset_name+str(j)+'.pt')
        ones(dataxp, dataxl, datay, axes=(2, 3), save_path=save_path)


def main():
    path_dir = './data/'
    from_name = 'PDBbind2020.h5'

    print('#'* 50)
    print('开始测试集的数据扩增！')
    data_augmentation(path_dir, from_name, 'test')
    print('测试集的数据扩增完毕！')

    print('#'* 50)
    print('开始验证集的数据扩增！')
    data_augmentation(path_dir, from_name, 'valid')
    print('验证集的数据扩增完毕！')

    print('#'* 50)
    print('开始训练集的数据扩增，请耐心等待并注意内存占用！')
    data_augmentation(path_dir, from_name, 'train')
    print('训练集的数据扩增完毕！程序已退出！')    

if __name__=='__main__':main()