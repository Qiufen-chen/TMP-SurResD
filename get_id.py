'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2022/5/16 17:38
@Author : Qiufen.Chen
@FileName: get_id.py
@Software: PyCharm
'''

import os
import numpy as np
import math

result_dir = '/lustre/home/qfchen/ContactMap/SurContact/output/'
# atom_dir = '/lustre/home/qfchen/ContactMap/atom/'

fin = './id_length.txt'
w = open(fin, '+a')
for root,dirs,files in os.walk(result_dir):
    for file in files:
        res_file = os.path.join(root,file)
        pdb_id = file.split('.')[0]
        data = np.load(res_file)
        w.write(pdb_id + '\t' + str(int(math.sqrt(data['y_true'].shape[0]))))
        w.write('\n')

        # if os.path.isfile(os.path.join(atom_dir, pdb_id + '.pdb')):
        #     atom_file = open(os.path.join(atom_dir, pdb_id + '.pdb'))
        #     rasa_file = open(os.path.join(root, file))
        #
        #
        #     atoms = atom_file.readlines()
        #     rasas = rasa_file.readlines()
        #
        #     if len(atoms) == len(rasas):
        #         w.write(file.split('.')[0] + '\t' + str(len(atoms)))
        #         w.write('\n')


