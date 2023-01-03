# Author QFIUNE
# coding=utf-8
# @Time: 2022/3/9 16:05
# @File: unNormalization.py
# @Software: PyCharm
# @contact: 1760812842@qq.com


import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

data_dir = "/lustre/home/qfchen/ContactMap/SurContact/output/"
for root,dirs, _files in os.walk(data_dir):
    n = 0
    for file in _files:
        n = n+1
        data_file = os.path.join(root, file)
        data = np.load(data_file)
        # npz文件保存的文件包括： ['arr_0', 'arr_1'] === y_true, y_pred
        # print('npz文件保存的文件包括：', data.files)
        y_true = data['arr_0']*200
        y_pred = data['arr_1']*200
        print(y_true[:10])
        print(y_pred[:10])

        odd = abs(y_true-y_pred)
        # print(y_true.shape)

        # 画残差图
        # import matplotlib.pyplot as plt
        # plt.scatter(y_pred[:200],odd[:200])
        # plt.savefig("/lustre/home/qfchen/ContactMap/11.jpg")



        # 给任务单独分配随机种子
        np.random.seed(sum(map(ord, "anscombe")))
        import seaborn as sns

        # Plot
        plt.figure(figsize=(10, 8), dpi=300)
        sns.residplot(x=odd, y=y_pred)
        save_path = '/lustre/home/qfchen/ContactMap/Residule_map/'
        plt.savefig(save_path + str(n)+ '.jpg')









