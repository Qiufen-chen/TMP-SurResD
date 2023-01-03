import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, matthews_corrcoef

pred_data = '/lustre/home/qfchen/ContactMap/SurContact/output/'
ccmpred_data = '/lustre/home/qfchen/ContactMap/SurContact/Test/ccmpred_test/'

ACC = 0
PRE = 0
REC = 0
F1 = 0
MCC = 0
for root, dirs, files in os.walk(pred_data):
    n = len(files)
    for file in files:
        pdb_id = file.split('.')[0]
        pred_file = os.path.join(root, file)
        data = np.load(pred_file)
        # print(data.files)
        true_lable = data['y_true']

        ccmpred_lable = np.loadtxt(os.path.join(ccmpred_data, pdb_id + '.mat'))
        ccmpred_lable = ccmpred_lable.reshape(-1)

        Max = 200
        true_lable = true_lable * 200
        # pred_lable = pred_lable * 200

        y_true = np.zeros(true_lable.shape[0], dtype=int)
        y_pred = np.zeros(true_lable.shape[0], dtype=int)
        print(true_lable.shape[0] == ccmpred_lable.shape[0])

        for i in range(true_lable.shape[0]):
            if true_lable[i] < 7 and true_lable[i]>0:
                y_true[i] = 1

            if ccmpred_lable[i] >= 0.9:
                y_pred[i] = 1

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
          .format(ACC/n, PRE/n, REC/n, F1/n , MCC/n))