'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/11/5 9:09
@Author : Qiufen.Chen
@FileName: get_onehot.py
@Software: PyCharm
'''
import numpy as np
import os
import argparse
from tensorflow import keras 
#from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical

dict = {'C': 0, 'D': 1, 'S': 2, 'Q': 3, 'K': 4,
        'I': 5, 'P': 6, 'T': 7, 'F': 8, 'N': 9,
        'G': 10, 'H': 11, 'L': 12, 'R': 13, 'W': 14,
        'A': 15, 'V': 16, 'E': 17, 'Y': 18, 'M': 19}


class process:
    def pre_processing(self, fasta_path, save_path):
        for (root, dirs, files) in os.walk(fasta_path):
            for file_name in files:
                with open(os.path.join(root, file_name), 'r') as get_fasta:
                    lines = get_fasta.readlines()
                    n = len(lines)
                    print(n)
                    pdb_id = ""
                    for line in lines:
                        codelist = []
                        if (line[0] == ">"):
                            pdb_id = line[1:].strip().split()[0]
                            continue
                        # -------- one-hot encoded (L * 21) ----------#
                        for i in line:
                            if (i != "\n"):
                                try:
                                    code = dict[i.upper()]
                                    codelist.append(code)
                                except KeyError:
                                    print('The sequence of ' + pdb_id + ' contains an invalid character.')

                        data = np.array(codelist)
                        # print('Shape of data (BEFORE encode): %s' % str(data.shape))
                        encoded = to_categorical(data)
                        if (encoded.shape[1] < 20):
                            column = np.zeros([encoded.shape[0], 20 - encoded.shape[1]], int)
                            encoded = np.c_[encoded, column]

                        print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
                        np.save(save_path + '/' + pdb_id, encoded)


if __name__ == '__main__':
    onehot = "/lustre/home/qfchen/ContactMap/multi_fasta/"
    save_path = "/lustre/home/qfchen/ContactMap/onehot/"
    proc = process()
    proc.pre_processing(onehot, save_path)
