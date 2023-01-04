'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/12/24 14:25
@Author : Qiufen.Chen
@FileName: copulanet_train.py
@Software: PyCharm
'''

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# from sklearn.preprocessing import MinMaxScaler
import torch
import torch as th
# from torchsummary import summary
# from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import processing
from torch import nn
import torch.utils.data as Data
from resnet import ResNet18
import numpy as np
from itertools import chain
import math
# scaler = MinMaxScaler(feature_range=[0, 1])

# ----------------------------------------------------------------------------
def pcc(y_pred,y_true):
    """caculate pcc"""
    y_true = y_true.reshape(-1).detach().cpu().numpy()
    y_pred = y_pred.reshape(-1).detach().cpu().numpy()
    cc = np.corrcoef(y_true, y_pred)
    return cc[0][1]

def r2(y_true, y_pred):
    """calculate r2"""
    y_true = y_true.reshape(-1).detach().cpu().numpy()
    y_pred = y_pred.reshape(-1).detach().cpu().numpy()
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r_2 = 1 - sse / sst
    return r_2

def mask(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    y_true = torch.squeeze(y_true)
    y_pred = torch.squeeze(y_pred)
    # print(y_true.shape, y_pred.shape)

    mask = y_true > 0
    y_true = torch.masked_select(y_true, mask)
    y_pred = torch.masked_select(y_pred, mask)
    # print(y_true.shape, y_pred.shape)
    return y_true, y_pred

# ----------------------------------------------------------------------------
class mydataset(Data.Dataset):
    """Create Dataset"""
    def __init__(self, feature_path, label_path):
        self.images = feature_path
        self.targets = label_path

    def __getitem__(self, index):
        data_path, label_path = self.images[index], self.targets[index]
        data = np.load(data_path)

        if data.ndim == 2:
            result = []
            for a in data:
                for b in data:
                    row = np.concatenate((a, b))
                    result.append(row)

            feature = np.array(result)

            width = feature.shape[0]
            width = int(math.sqrt(width))
            depth = feature.shape[1]
            feature = feature.reshape([depth, width, width])
            feature = torch.from_numpy(feature)

        label_data = np.load(label_path)

        max_value = 100
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


# ---------------------------------------------------------------------------
batch_size = 1
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1023)


con = processing.Contact()
x_train, y_train, x_val, y_val, x_test, y_test = con.main('onehot')
train = mydataset(x_train, y_train)
val = mydataset(x_val, y_val)
test = mydataset(x_test, y_test)
train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=1)
val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=1)
test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=1)


def main(n):
    """train model"""
    model = ResNet18(n).to(device)
    print('model', model)

    w_loss, w_mae, w_mse, w_r2, w_cc = 0, 0, 0, 0, 0
    criterion = nn.SmoothL1Loss(reduction='mean')
    MAE_fn = nn.L1Loss(reduction='mean')
    MSE_fn = nn.MSELoss(reduction='mean')

    optimizer = th.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, threshold=0.001, threshold_mode='rel', cooldown=0,
        min_lr=0.001, eps=1e-08, verbose=False)

    model.train()
    for epoch in range(epochs):
        for step, (x, y_true) in enumerate(train_loader):
            x = x.to(torch.float32).to(device)
            y_true = y_true.to(torch.float32).to(device)
            y_pred = model(x)

            y_true, y_pred = mask(y_true, y_pred)

            train_loss = criterion(y_pred, y_true)
            train_mae = MAE_fn(y_pred, y_true)
            train_mse = MSE_fn(y_pred, y_true)
            train_r2 = r2(y_pred, y_true)
            train_cc = pcc(y_pred, y_true)

            if np.isnan(train_cc):
                train_cc = 0
            if math.isinf(train_r2):
                train_r2 = 0

            w_mae += train_mae.detach().item()
            w_loss += train_loss.detach().item()
            w_mse += train_mse.detach().item()
            w_r2 += train_r2
            w_cc += train_cc

             # Back propagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        w_mae /= step + 1
        w_loss /= step + 1
        w_mse /= step + 1
        w_r2 /= step + 1
        w_cc /= step + 1
        scheduler.step(w_loss)

        # evaluating indicator
        print("Epoch {:2d}, trian_loss: {:.4f}, trian_mae: {:.4f}, "
              "trian_mse:{:.4f}, trian_r2: {:.4f}, trian_cc: {:.4f}"
              .format(epoch, w_loss, w_mae, w_mse, w_r2, w_cc))

        # Validation
        model.eval()
        val_loss, val_mae, val_mse, val_r2, val_cc = 0, 0, 0, 0, 0
        with torch.no_grad():
            for index, (x, y_true) in enumerate(val_loader):
                x = x.to(torch.float32).to(device)
                y_true = y_true.to(torch.float32).to(device)

                y_pred = model(x)
                y_true, y_pred = mask(y_true, y_pred)
                loss_1 = criterion(y_pred, y_true)
                mae_1 = MAE_fn(y_pred, y_true)
                mse_1 = MSE_fn(y_pred, y_true)
                r2_1 = r2(y_pred, y_true)
                cc_1 = pcc(y_pred, y_true)


                if np.isnan(cc_1):
                    cc_1 = 0
                if math.isinf(r2_1):
                    r2_1 = 0
                    
                val_loss += loss_1.detach().item()
                val_mae += mae_1.detach().item()
                val_mse += mse_1.detach().item()
                val_r2 += r2_1
                val_cc += cc_1

        val_mae /= index + 1
        val_loss /= index + 1
        val_mse /= index + 1
        val_r2 /= index + 1
        val_cc /= index + 1

        print("Epoch {:2d}, val_loss: {:.4f}, val_mae: {:.4f},"
              " val_mse: {:.4f}, val_r2: {:.4f}, val_cc: {:.4f}"
              .format(epoch, val_loss, val_mae, val_mse, val_r2, val_cc))

        if epoch % 10 == 0:
            th.save(model.state_dict(), "/lustre/home/qfchen/ContactMap/TMContact/model/" + "18_mask_new_Normalization_onehot_" + str(epoch))


if __name__ == '__main__':
    n = 40
    main(n)