'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2022/3/17 16:21
@Author : Qiufen.Chen
@FileName: r2r.py
@Software: PyCharm
'''
import json
import math
import os

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

r2r_dir = '/lustre/home/qfchen/ContactMap/TMContact/Test/R2R_result/'
lable_dir = '/lustre/home/qfchen/ContactMap/TMContact/output/'
ACC = 0
PRE = 0
REC = 0
F1 = 0
MCC = 0

for root, dirs, files in os.walk(r2r_dir):
    n = len(files)
    print(n)
    for file in files:
        pdb_id = file.split('.')[0]
        # print(pdb_id)
        data = np.load(os.path.join(lable_dir, pdb_id+'.npz'))
        true_lable = data['y_true']
        Max = 200
        true_lable = true_lable * 20

        y_true = np.zeros(true_lable.shape[0])

        for i in range(true_lable.shape[0]):
            if true_lable[i] < 7 and true_lable[i]>0:
                y_true[i] = 1

        m = int(math.sqrt(true_lable.shape[0]))
        # print(m)
        r2r_matr = np.zeros([m, m])

        r2r_file = os.path.join(root, file)

        with open(r2r_file, 'r') as fo:
            lines = fo.readlines()
            n = 0
            new = lines
            for line in lines:
                arr = line.split()
                i = int(arr[0])
                j = int(arr[2])
                # print(i,j)
                if float(arr[4]) > 0.4:
                    # print(arr[4])
                    r2r_matr[i - 1][j - 1] = 1
                    r2r_matr[j - 1][i - 1] = 1

        y_pred = r2r_matr.flatten()
        # print(y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        # print(acc,prec)

        ACC += acc
        PRE += prec
        REC += recall
        F1 += f1
        MCC += mcc
    # print(ACC)
    print("Accuracy: {:.4f}, Precision: {:.4f}, "
          "TRecall:{:.4f}, F1: {:.4f}, MCC: {:.4f}"
          .format(ACC / 15, PRE / 15, REC / 15, F1 /15, MCC / 15))