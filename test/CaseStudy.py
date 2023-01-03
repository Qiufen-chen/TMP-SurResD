# Author QFIUNE
# coding=utf-8
# @Time: 2022/6/6 11:26
# @File: CaseStudy.py
# @Software: PyCharm
# @contact: 1760812842@qq.com


import os
import numpy as np
import matplotlib.pyplot as plt
import torch


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

plt.rcParams['axes.linewidth'] = 1
plt.rcParams['figure.dpi'] = 300
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 20}
plt.rc('font', **font)

result = '/lustre/home/qfchen/ContactMap/SurContact/output/'
for root, dirs, files in os.walk(result):
    for file in files:
        pdb_id = file.split('.')[0]
        pred_file = os.path.join(root, file)
        data = np.load(pred_file)
        # print(data.files)
        true_lable = data['y_true']
        pred_lable = data['y_pred']

        true_lable = torch.from_numpy(true_lable)
        pred_lable = torch.from_numpy(pred_lable)
        true_lable, pred_lable = mask(true_lable, pred_lable)

        # Max = 200
        #
        # true_lable = true_lable * 200
        # pred_lable = pred_lable * 200

        y_true = true_lable.reshape(-1)
        y_pred = pred_lable.reshape(-1)
        plt.figure(figsize=(10,10), dpi=600)
        # plt.plot(np.arange(y_true.shape[0]), y_true, color='r', label='True y')
        # plt.plot(np.arange(y_true.shape[0]), y_pred, color='g', label='Predicted y')
        # plt.plot(y_true,y_pred,'go')
        # plt.scatter(y_true,y_pred, c='#8A2BE2')
        # plt.scatter(y_true,y_pred, c='#008B8B')
        # plt.scatter(y_true,y_pred, c='#228B22')
        plt.scatter(y_true,y_pred, c='#008080')

        parameter = np.polyfit(y_pred, y_true, 1)
        y = parameter[0]*y_pred + parameter[1]
        
        plt.plot(y_pred, y, color='#000000', linewidth=3)

        plt.xlim(min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true)))
        plt.ylim(min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true)))
        

        # plt.title('Regression result comparision')
        plt.legend(loc='upper right', labels=['fitted value', pdb_id])
        plt.title(pdb_id)
        plt.xlabel('Real value')
        plt.ylabel('Predicted value')

        # plt.show()
        plt.savefig("./picture/" + pdb_id + '.jpg')





