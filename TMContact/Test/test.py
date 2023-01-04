# Author QFIUNE
# coding=utf-8
# @Time: 2022/3/1 16:15
# @File: test.py
# @Software: PyCharm
# @contact: 1760812842@qq.com
import glob
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


testID = '/lustre/home/qfchen/ContactMap/TMContact/dataset/testID.txt'

import torch
from resnet import ResNet18
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F

def get_path():
    testID = "/lustre/home/qfchen/ContactMap/TMContact/dataset/testID.txt"
    lable_dir = "/lustre/home/qfchen/ContactMap/TMContact/lable/"
    feature_dir = "/lustre/home/qfchen/ContactMap/TMContact/onehot_pair/"
    images_path = []
    lable_path = []
    with open(testID, 'r') as fo:
        lines = fo.readlines()
        for line in lines:
            pdb_id = line.replace('\n', '')
            images_path.append(os.path.join(feature_dir, pdb_id + '.npy'))
            lable_path.append(os.path.join(lable_dir, pdb_id + '.npy'))

        return images_path, lable_path

class mydataset(Data.Dataset):
    """Create Dataset"""
    def __init__(self, feature_path, label_path):
        self.images = feature_path
        self.targets = label_path

    def __getitem__(self, index):
        data_path, label_path = self.images[index], self.targets[index]
        data = np.load(data_path)
        feature = np.array(data)
        # print(feature.shape)

        width = feature.shape[0]
        heigth = feature.shape[1]
        depth = feature.shape[2]
        feature = feature.reshape([depth, width, heigth])
        feature = torch.from_numpy(feature)

        label_data = np.load(label_path)

        max_value = 200
        new_label = np.zeros([label_data.shape[0], label_data.shape[1]], float)

        # Min-Max Normalization
        for i in range(label_data.shape[0]):
            for j in range(label_data.shape[1]):
                new_label[i][j] = label_data[i][j] / max_value

        width, height, depth = new_label.shape[0], new_label.shape[1], 1
        new_label = new_label.reshape([depth, width, height])
        new_label = torch.from_numpy(new_label)

        return feature, new_label

    def __len__(self):
        return len(self.images)


def pcc(y_pred,y_true):
    """caculate pcc"""
    y_true = y_true.reshape(-1).detach().numpy()
    y_pred = y_pred.reshape(-1).detach().numpy()
    cc = np.corrcoef(y_true, y_pred)
    return cc[0][1]

def r2(y_true, y_pred):
    """calculate r2"""
    y_true = y_true.reshape(-1).detach().numpy()
    y_pred = y_pred.reshape(-1).detach().numpy()
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r_2 = 1 - sse / sst
    return r_2


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1023)  # 设置CPU生成随机数的种子,方便下次复现实验结果.

    X,Y = get_path()
    test = mydataset(X, Y)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=1)

    w_loss, w_mae, w_mse, w_r2, w_cc = 0, 0, 0, 0, 0
    criterion = nn.SmoothL1Loss(reduction='mean')
    MAE_fn = nn.L1Loss(reduction='mean')
    MSE_fn = nn.MSELoss(reduction='mean')

    model_path = '/lustre/home/qfchen/ContactMap/TMContact/model/new_Normalization_onehot_pair_16_90'

    model = ResNet18(400)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    for step, (x, y_true) in enumerate(test_loader):
        x = x.to(torch.float32).to(device)
        y_true = y_true.to(torch.float32).to(device)
        y_pred = model(x)
        train_loss = criterion(y_pred, y_true)
        train_mae = MAE_fn(y_pred, y_true)
        train_mse = MSE_fn(y_pred, y_true)
        train_r2 = r2(y_pred, y_true)
        train_cc = pcc(y_pred, y_true)
        print("trian_loss: {:.4f}, trian_mae: {:.4f}, "
              "trian_mse:{:.4f}, trian_r2: {:.4f}, trian_cc: {:.4f}"
              .format(train_loss,train_mae, train_mse,train_r2, train_cc))

        w_mae += train_mae.detach().item()
        w_loss += train_loss.detach().item()
        w_mse += train_mse.detach().item()
        w_r2 += train_r2
        w_cc += train_cc

    w_mae /= step + 1
    w_loss /= step + 1
    w_mse /= step + 1
    w_r2 /= step + 1
    w_cc /= step + 1


    # evaluating indicator
    print("Total_loss: {:.4f}, Total_mae: {:.4f}, "
          "Total_mse:{:.4f}, Total_r2: {:.4f}, Total_cc: {:.4f}"
          .format(w_loss, w_mae, w_mse, w_r2, w_cc))



if __name__ == '__main__':
    predict()