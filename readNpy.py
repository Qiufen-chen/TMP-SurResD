'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/10/23 15:58
@Author : Qiufen.Chen
@FileName: readNpy.py
@Software: PyCharm
'''

import numpy as np
import os

lable_dir = "/lustre/home/qfchen/ContactMap/SurContact/lable/"
hmm_dir = "/lustre/home/qfchen/ContactMap/SurContact/hmm/"
save_dir = '/lustre/home/qfchen/ContactMap/npy/'
ccmpred_dir = "/lustre/home/qfchen/ContactMap/SurContact/ccmpred/"
onehot_dir = "/lustre/home/qfchen/ContactMap/SurContact/onehot/"

id_file = open('./ID.txt', '+a')

id = '/lustre/home/qfchen/ContactMap/surConD_id.txt'
with open(id, 'r') as fin:
    lines = fin.readlines()
    for line in lines:
# names = os.listdir(lable_dir)
# for root, dirs, files in os.walk(hmm_dir):
#     for file in files:
        hmm_file = os.path.join(lable_dir, line.replace('\n', '')+'.npy')
        lable_file = os.path.join(hmm_dir, line.replace('\n', '')+'.npy')
        onehot_file = os.path.join(onehot_dir, line.replace('\n', '')+'.npy')
        ccmpred_file = os.path.join(ccmpred_dir, line.replace('\n', '')+'.mat')


        onehot_data = np.load(onehot_file)
        ccmpred_data = np.loadtxt(ccmpred_file)


        hmm_data = np.load(hmm_file)
        lable_data = np.load(lable_file)

        # print(line.replace('\n', ''), hmm_data.shape[0], lable_data.shape[0],
        #       ccmpred_data.shape[0], onehot_data.shape[0], profold.shape[0])

        if hmm_data.shape[0] == lable_data.shape[0] == ccmpred_data.shape[0] == onehot_data.shape[0]:
            pdb_id = line.replace('\n', '')
            id_file.write(pdb_id + '\n')

        # np.savetxt(save_dir + file.split('.')[0]+'.txt', lable_data, fmt='%s', newline='\n')


def Max(dir):
    """get distance_max value"""
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            data = np.load(path)
            # print(data.shape)
            print(file.split('.')[0],  np.amax(data))

def countID(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            print(file.split('.')[0])

if __name__ == '__main__':
    # dir = "/lustre/home/qfchen/ContactMap/TMContact/lable/"
    # # dir = "/lustre/home/qfchen/ContactMap/TMContact/onehot_pair/"
    # Max(dir)
    # # # countID(dir)
    pass