'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2022/3/22 0:24
@Author : Qiufen.Chen
@FileName: deepcon.py
@Software: PyCharm
'''

import math
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

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

deepcon_dir = '/lustre/home/qfchen/ContactMap/SurContact/Test/DEEPCON_result/'
for root, dirs, files in os.walk(deepcon_dir):
    n =0
    for file in files:
        pdb_id = file.split('.')[0]
        data = np.load(os.path.join(lable_dir, pdb_id+'.npz'))
        true_lable = data['y_true']
        Max = 200
        true_lable = true_lable * 20

        y_true = np.zeros(true_lable.shape[0])

        for i in range(true_lable.shape[0]):
            if true_lable[i] < 8 and true_lable[i]>0:
                y_true[i] = 1
 
        m = int(math.sqrt(true_lable.shape[0]))
        deepcon_matr = np.zeros([m, m])

        deepcon_file = os.path.join(root, file)

        with open(deepcon_file, 'r') as fo:
            lines = fo.readlines()
            print(len(lines))
            
            n += 1
            for line in lines[1:]:
                arr = line.split()
                i = int(arr[0])
                j = int(arr[1])
                if float(arr[4]) > 0.5:
                    deepcon_matr[i - 1][j - 1] = 1
                    deepcon_matr[j - 1][i - 1] = 1

            y_pred = deepcon_matr.flatten()
            y = torch.tensor(y_true)
            y_hat = torch.tensor(y_pred)
            y_true, y_pred = mask(y, y_hat)

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