'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/11/2 20:30
@Author : Qiufen.Chen
@FileName: extract_HHM.py
@Software: PyCharm
'''
import os
import math
import numpy as np

id = "/lustre/home/qfchen/ContactMap/id.txt"
hhblits_dir = "/lustre/home/qfchen/ContactMap/hhblits/"

with open(id, 'r') as fo:
    lines = fo.readlines()
    for line in lines:
        li = line.split()
        print(li)
        pdb_id = li[0]
        length = int(li[1].replace('\n', ''))

        hhm_file = os.path.join(hhblits_dir, pdb_id+'.hhm')
        with open(hhm_file) as hhm:
            hhm_matrix = np.zeros([length, 30], float)
            hhm_line = hhm.readline()
            while hhm_line[0] != '#':
                hhm_line = hhm.readline()
            for i in range(0, 5):
                hhm_line = hhm.readline()
            raw = 1
            while hhm_line:
                if len(hhm_line.split()) == 23:
                    each_item = hhm_line.split()[2:22]
                    for idx, s in enumerate(each_item):
                        if s == '*':
                            each_item[idx] = '99999'
                    for j in range(0, 20):
                        try:
                            hhm_matrix[raw-1, j] = 1/(1 + math.exp(-1 * int(each_item[j])/2000))
                            # hhm_matrix[raw-1, j] = 1.0/(1 + math.exp(-int(each_item[j])))
                        except IndexError:
                            pass
                elif len(hhm_line.split()) == 10:
                    each_item = hhm_line.split()[0:10]
                    for idx, s in enumerate(each_item):
                        if s == '*':
                            each_item[idx] = '99999'
                    for j in range(20, 30):
                        try:
                            hhm_matrix[raw-1, j] = 1/(1 + math.exp(-1 * int(each_item[j-20])/2000))
                            # hhm_matrix[raw-1, j] = 1.0 / (1 + math.exp(-int(each_item[j-20])))
                        except IndexError:
                            pass
                    raw += 1
                hhm_line = hhm.readline()

        print(hhm_matrix)
        save_path = "/lustre/home/qfchen/ContactMap/SurContact/hmm/"
        np.save(os.path.join(save_path) + '/' + pdb_id, hhm_matrix)

#
#
