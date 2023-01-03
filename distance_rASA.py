'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/12/20 10:54
@Author : Qiufen.Chen
@FileName: distance_rASA.py
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
        print("This path is exist!")
        return False


def get_matrix(input_dir, save_dir, threshold):
    file_num = 0
    count = 0
    for (root, dirs, files) in os.walk(input_dir):
        for file_name in files:
            print(file_name)
            file_num = file_num + 1
            with open(os.path.join(root, file_name), 'r') as f1:
                lines = f1.readlines()
                # print(len(lines))
                n = len(lines)

                initial_matrix = np.zeros((n, n))
                str1 = [0 for i in range(len(lines))]
                k = 0
                for line in lines:
                    str1[k] = line[0:88]
                    k = k + 1

                for i in range(0, len(lines)):
                    if float(str1[i][80:110].strip()) < 0.2:
                        continue

                    x_1 = float(str1[i][30:38])
                    # print(x_1)
                    y_1 = float(str1[i][38:46])
                    z_1 = float(str1[i][46:56])

                    for j in range(0, len(lines)):
                        if float(str1[i][80:110].strip()) < 0.2:
                            continue

                        x_2 = float(str1[j][30:38])
                        # print(x_2)
                        y_2 = float(str1[j][38:46])
                        z_2 = float(str1[j][46:56])

                    
                        ans = math.sqrt(pow(x_1 - x_2, 2) + pow(y_1 - y_2, 2)
                                        + pow(z_1 - z_2, 2))
                        # if ans <= 8:
                        #
                        #     initial_matrix[i][j] = 1
                        initial_matrix[i][j] = ans

                # print(initial_matrix.shape)
                print(file_name, initial_matrix.max(), initial_matrix.min())
            count = count + 1

            np.save(save_dir + '/' + str(file_name).split('.')[0], initial_matrix)

        print("Finished %d contact files." % count)


# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
     input_atom_dir = "/lustre/home/qfchen/ContactMap/SurContact/rASR_atom/"

     output_matrix_dir = "/lustre/home/qfchen/ContactMap/SurContact/lable/"
     make_dir(output_matrix_dir)

     get_matrix(input_atom_dir, output_matrix_dir, 8)







