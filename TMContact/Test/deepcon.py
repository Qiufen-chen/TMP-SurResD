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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

ACC = 0
PRE = 0
REC = 0
F1 = 0
MCC = 0

lable_dir = '/lustre/home/qfchen/ContactMap/TMContact/output/'

deepcon_dir = '/lustre/home/qfchen/ContactMap/TMContact/Test/deepcon_result/'
for root, dirs, files in os.walk(deepcon_dir):
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
            if true_lable[i] < 7 and true_lable[i]>0:
                y_true[i] = 1

        m = int(math.sqrt(true_lable.shape[0]))
        print(m)
        deepcon_matr = np.zeros([m, m])
        print(deepcon_matr.shape)
        deepcon_file = os.path.join(root, file)

        with open(deepcon_file, 'r') as fo:
            lines = fo.readlines()
            for line in lines[1:]:
                arr = line.split()
                i = int(arr[0])
                j = int(arr[1])
                if float(arr[4]) > 0.9:
                    deepcon_matr[i - 1][j - 1] = 1
                    deepcon_matr[j - 1][i - 1] = 1

        y_pred = deepcon_matr.flatten()
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