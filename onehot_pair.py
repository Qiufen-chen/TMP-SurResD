'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/11/21 10:14
@Author : Qiufen.Chen
@FileName: onehot_pair.py
@Software: PyCharm
'''
import pickle
from itertools import permutations

import numpy as np
import os
from tensorflow.keras.utils import to_categorical


def permutation():
    """
    purpose: 实现20种氨基酸残基两两排列
    :return: 400种排列
    """
    resList = []

    amino = 'ARDCQEHIGNLKMFPSTWYV'
    residue = sorted(set(amino))

    for i, j in permutations(residue, 2):
        resList.append((i, j))

    for i in residue:
        resList.append((i,i))

    # 将400个氨基酸对编号
    resDict = dict((v, k) for k, v in enumerate(resList))

    return resDict

def Onehot(fasta_path, save_path, resDict):
    for (root, dirs, files) in os.walk(fasta_path):
        for file_name in files:
            with open(os.path.join(root, file_name), 'r') as get_fasta:
                lines = get_fasta.readlines()
                pdb_id = ""

                for line in lines:
                    codeList = []
                    if (line[0] == ">"):
                        pdb_id = line[1:].strip().split()[0]
                        continue
                    # -------- one-hot encoded (L*L*400) ----------#
                    if '\n' in line:
                        n = len(line)-1
                    else:
                        n = len(line)
                    for i in line:
                        for j in line:
                            if (j != "\n" and i != "\n"):
                                try:
                                    code = resDict[(i.upper(), j.upper())]
                                    codeList.append(code)
                                except KeyError:
                                    print('The sequence of ' + pdb_id + ' contains an invalid character.')

                    data = np.array(codeList)
                    # print('Shape of data (BEFORE encode): %s' % str(data.shape))
                    encoded = to_categorical(data)
                    print(encoded.shape)

                    if encoded.shape[1] < 400:
                        padding_c = np.zeros([encoded.shape[0], 400-encoded.shape[1]], int)
                        encoded = np.c_[encoded, padding_c]

                    # padding_r = np.zeros([1000000 - encoded.shape[0], 400], int)
                    #
                    # encoded = np.r_[encoded, padding_r]
                    print(encoded.shape)
                    encoded = encoded.reshape(n, n, 400)
                    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
                    np.save(save_path + '/' + pdb_id, encoded)
                    # output = open(save_path + '/' + pdb_id + '.pkl', 'wb')
                    # pickle.dump(encoded, output)
                    # output.close()

if __name__ == '__main__':
    fastaDir = "/lustre/home/qfchen/ContactMap/TMContact/fasta/"
    saveDir = "/lustre/home/qfchen/ContactMap/TMContact/onehot_pair/"
    dic = permutation()
    Onehot(fastaDir, saveDir, dic)





