'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2022/3/21 23:59
@Author : Qiufen.Chen
@FileName: PSICOV.py
@Software: PyCharm
'''

import math
import os

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
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

ACC = 0
PRE = 0
REC = 0
F1 = 0
MCC = 0

lable_dir = '/lustre/home/qfchen/ContactMap/SurContact/output/' 
psicov_dir = '/lustre/home/qfchen/ContactMap/SurContact/Test/DEEPCON_result/'
for root, dirs, files in os.walk(psicov_dir):
    n = len(files)
    for file in files:
        pdb_id = file.split('.')[0]
        print(pdb_id)
        data = np.load(os.path.join(lable_dir, pdb_id+'.npz'))
        true_lable = data['y_true']
        Max = 200
        true_lable = true_lable * 20

        y_true = np.zeros(true_lable.shape[0])

        for i in range(true_lable.shape[0]):
            if true_lable[i] < 8 and true_lable[i]>0:
                y_true[i] = 1

        # y_true, y_pred = mask(y_true, y_pred)

        m = int(math.sqrt(true_lable.shape[0]))
        print(m)
        psicov_matr = np.zeros([m, m])
        print(psicov_matr.shape)
        psicov_file = os.path.join(root, file)

        with open(psicov_file, 'r') as fo:
            lines = fo.readlines()
            for line in lines[1:]:
                arr = line.split()
                i = int(arr[0])
                j = int(arr[1])
                if float(arr[4]) > 0.5:
                    psicov_matr[i - 1][j - 1] = 1
                    psicov_matr[j - 1][i - 1] = 1

        y_pred = psicov_matr.flatten()
        y_true, y_pred = mask(torch.tensor(y_true), torch.tensor(y_pred))
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        # print(len(y_true))
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        ACC += acc
        PRE += prec
        REC += recall
        F1 += f1
        MCC += mcc
    print("Accuracy: {:.4f}, Precision: {:.4f}, "
          "TRecall:{:.4f}, F1: {:.4f}, MCC: {:.4f}"
          .format(ACC / n, PRE / n, REC / n, F1 / n, MCC / n))
