# 此版本为优化版本的分支版本，将x拆分成了xp和xl进行训练。
# 通过无效化xp，只保留xl设置阴性对照实验，来评估模型性能，探究主模型整体的鲁棒性。
# 作者：常怡彬
# 最后更新：2023年10月18日

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import pandas as pd
import h5py
from tqdm import *
from scipy.ndimage import rotate
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

# 环境检测
'''
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())
'''
# modle Net
class FireBlock(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireBlock, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv3d(in_channels, squeeze_channels, kernel_size=1),
            nn.ReLU()
        )
        self.expand1 = nn.Sequential(
            nn.Conv3d(squeeze_channels, expand_channels, kernel_size=1),
            nn.ReLU()
        )
        self.expand2 = nn.Sequential(
            nn.Conv3d(squeeze_channels, expand_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        squeezed = self.squeeze(x)
        expanded1 = self.expand1(squeezed)
        expanded2 = self.expand2(squeezed)
        return torch.cat((expanded1, expanded2), 1)

class CNNScore(nn.Module):
    def __init__(self):
        super(CNNScore, self).__init__()
        self.conv1 = nn.Conv3d(8, 48, kernel_size=1, stride=2)
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()

        layer_configs = [
            (48, 8, 32),
            (64, 8, 32),
            (64, 16, 64),
            (128, 16, 64),
            (128, 24, 96),
            (192, 24, 96),
            (192, 32, 128)
        ]

        for in_channels, squeeze_channels, expand_channels in layer_configs[:3]:
            self.layers1.append(FireBlock(in_channels, squeeze_channels, expand_channels))
        for in_channels, squeeze_channels, expand_channels in layer_configs[3:]:
            self.layers2.append(FireBlock(in_channels, squeeze_channels, expand_channels))

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2)
        self.avg_pool = nn.AvgPool3d(kernel_size=3, padding=1)
        self.dense1 = nn.Linear(512 * 2 * 2 * 2, 1)

    def _expand_layers(self, x):
        x = F.relu(self.conv1(x))
        for layer in self.layers1:
            x = layer(x)
        x = self.pool(x)

        for layer in self.layers2:
            x = layer(x)
        x = self.avg_pool(x)
        return x

    def forward(self, x1, x2):
        x1 = self._expand_layers(x1)
        x2 = self._expand_layers(x2)
        x = torch.cat((x1, x2), 1)
        x = x.view(-1, 512 * 2 * 2 * 2)
        x = self.dense1(x)
        x = x.view(-1)
        return x

# 设置Adam优化器
def create_optimizer(net, lr):
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    return optimizer

# 反向传播
def train(device, num_epochs, net, optimizer, batch_size=128, shuffle=True, batch_print_interval=16):  
    # 设置回归损失函数
    criterion = nn.MSELoss()
    
    for epoch in tqdm(range(num_epochs)):
        print('\n')
        running_loss = 0.0
        
        # 遍历不同的数据文件
        for file_idx in range(6):
            running_loss = 0.0
            file_path = f'./data/train{file_idx}.pt'
            data_set = torch.load(file_path)
            data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
            
            for batch_idx, (_, inputsl, labels) in enumerate(data_loader):  # 丢弃原本的xp信息
                inputsl = inputsl.permute(0, 4, 1, 2, 3)
                inputsp = torch.zeros_like(inputsl) # 重新生成形状等于xl的零张量xp
                inputsp, inputsl, labels = inputsp.to(device), inputsl.to(device), labels.to(device)
                
                optimizer.zero_grad()

                outputs = net(inputsp,inputsl)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                if (batch_idx + 1) % batch_print_interval == 0:
                    avg_loss = running_loss / batch_print_interval
                    print(f'[Epoch {epoch + 1}, Data {file_idx + 1}, Batch {batch_idx + 1}] Loss: {avg_loss:.3f}')
                    running_loss = 0.0
    
    return net

# 模型性能评估
def evaluate(device, model, batch_size=128, shuffle=False):
    # 定义子集划分函数
    def split_data(data, cut_num):
        # param cut_num: 分割的子行数
        chunk_size = len(data) // cut_num
        num_chunks = len(data) // chunk_size

        data_matrix = np.array(data[:num_chunks * chunk_size]).reshape(-1, chunk_size)
        means = np.mean(data_matrix, axis=0)

        return means
    
    # 将模型设置为评估模式
    model.eval()

    # 初始化用于保存真实值和预测值的列表
    true_values = []
    pred_values = []

    # 遍历不同的数据文件
    for file_idx in tqdm(range(6)):
        file_path = f'./data/train{file_idx}.pt'
        data_set = torch.load(file_path)
        data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

        true_ones = []
        pred_ones = []

        with torch.no_grad():
            for batch in data_loader:
                inputsp, inputsl, labels = batch
                inputsp, inputsl = inputsp.permute(0, 4, 1, 2, 3), inputsl.permute(0, 4, 1, 2, 3)
                inputsp, inputsl, labels = inputsp.to(device), inputsl.to(device), labels.to(device)

                # 获取模型的预测值
                predictions = model(inputsp, inputsl)

                # 将真实值和预测值添加到列表中
                true_ones.extend(labels.cpu().numpy())
                pred_ones.extend(predictions.cpu().numpy())

        # 切割列表并求平均值
        true_ones = split_data(true_ones, 4)
        pred_ones = split_data(pred_ones, 4)

        # 将6个文件中的4次平均值添加到总列表中
        true_values.extend(true_ones)
        pred_values.extend(pred_ones)

    # 分割六个文件并取平均值
    true_values = split_data(true_values, 6)
    pred_values = split_data(pred_values, 6)

    # 组合结果矩阵
    result_matrix = np.column_stack((true_values, pred_values))

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))

    # 计算斯皮尔曼相关系数
    spearman_corr, _ = spearmanr(true_values, pred_values)

    # 计算皮尔森相关系数
    pearson_corr, _ = pearsonr(true_values, pred_values)

    return {
        "RMSE": rmse,
        "Spearman Correlation": spearman_corr,
        "Pearson Correlation": pearson_corr
    }, result_matrix


def main():
    # 环境检测
    if torch.cuda.is_available():
        print("cuda 正常，程序继续运行。")
        device_cpu = torch.device("cpu")
        device_gpu = torch.device("cuda:0")
    else:
        print("cuda 异常，请检测显卡运行状况，程序已经退出。")
        sys.exit(0)

    path = './pretest1.h5' 

    # 定义模型
    cnn_mix = CNNScore()

    # GPU 训练
    print('#' * 50)
    print("GPU 训练开始！")

    net_gpu = cnn_mix.to(device_gpu)
    opt_gpu = create_optimizer(net_gpu, 0.0001)

    # 训练模型
    net_gpu_tr = train(device_gpu, num_epochs=100, net=net_gpu, optimizer=opt_gpu)

    # 保存模型
    test_num = '1_lig'
    model = 'model_'+'gpu_'+'100_'+test_num+'.pth'

    torch.save(net_gpu_tr, model)
    print('模型训练完毕并储存在%s 文件中！' %(model))

    # 评估模型
    print('#' * 50)
    print('进行模型评估！')

    result, res_m = evaluate(device_gpu, net_gpu_tr)
    print('模型性能如下 \n', result)

    # 保存 matrix 矩阵到 CSV 文件
    matrix = 'matrix_'+'gpu_'+'100_'+test_num+'.csv'

    res_m_df = pd.DataFrame(res_m)
    res_m_df.to_csv(matrix, index=False)
    print('模型评估完毕并储存在%s 文件中！' %(matrix))

    print('程序运行完毕，程序已经退出！')

if __name__ == "__main__":
    main()