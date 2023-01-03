'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/11/22 20:30
@Author : Qiufen.Chen
@FileName: distance_TM.py
@Software: PyCharm
'''

import os
import numpy as np
import math

def make_dir(path):
    """
    purpose: Created path folder
    :param path:
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + " Created folder sucessful!")
        return True
    else:
        print("This path is exist！")
        return False


def get_matrix(input_dir, save_dir, threshold):
    file_num = 0
    count = 0
    for (root, dirs, files) in os.walk(input_dir):
        for file_name in files:
            file_num = file_num + 1
            with open(os.path.join(root, file_name), 'r') as f1:
                lines = f1.readlines()
                n = len(lines)

                initial_matrix = np.zeros((n, n))
                str1 = [0 for i in range(len(lines))]  # 初始化0列表
                k = 0
                for line in lines:
                    str1[k] = line[0:88]  # 为列表赋值
                    k = k + 1

                for i in range(0, len(lines)):
                    # print(file_name, str1[i][84:88])
                    if str1[i][84:88].strip() != 'T':
                        continue

                    x_1 = float(str1[i][30:38])
                    y_1 = float(str1[i][38:46])
                    z_1 = float(str1[i][46:56])

                    for j in range(0, len(lines)):
                        if str1[i][84:88].strip() != 'T':
                            continue

                        x_2 = float(str1[j][30:38])
                        y_2 = float(str1[j][38:46])
                        z_2 = float(str1[j][46:56])

                        # 计算氨基酸之间的欧氏距离
                        ans = math.sqrt(pow(x_1 - x_2, 2) + pow(y_1 - y_2, 2) + pow(z_1 - z_2, 2))
                        # if ans <= threshold:
                        #     initial_matrix[i][j] = ans  # 做成分类问题
                        # 回归问题
                        initial_matrix[i][j] = ans

                # print(initial_matrix.shape)
                print(initial_matrix.shape, initial_matrix.max(), initial_matrix.min())
            count = count + 1

            np.save(save_dir + '/' + str(file_name).split('.')[0], initial_matrix)

        print("Finished %d contact files." % count)


# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
     input_atom_dir = "/lustre/home/qfchen/ContactMap/TMContact/atom_topo/"
     output_matrix_dir = "/lustre/home/qfchen/ContactMap/TMContact/new_lable/"
     make_dir(input_atom_dir)

     get_matrix(input_atom_dir, output_matrix_dir, 8)







