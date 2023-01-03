'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/11/22 19:11
@Author : Qiufen.Chen
@FileName: connection_files.py
@Software: PyCharm
'''
import os

"""topo_atom"""
# input_path = '/home/chenqiufen/ContactMap/ContactMap/data/topo_2/'
# for (root, dirs, files) in os.walk(input_path):  # 列出windows目录下的所有文件和文件名
#     file_num = 0
#     for file_name in files:
#         pdb_id = file_name.split('.')[0]
#
#         topo_file = os.path.join(root) + '/' + file_name
#         pdb_file = os.path.join('/home/chenqiufen/ContactMap/data/NR_chainAtom/', pdb_id + '.pdb')
#         save_path = os.path.join('/home/chenqiufen/ContactMap/ContactMap/data/atom_topo/', pdb_id + '.pdb')
#
#         file1 = open(pdb_file, "r", encoding="utf-8")
#         file2 = open(topo_file, "r", encoding="utf-8")
#         file3 = open(save_path, "w", encoding="utf-8")
#
#         while True:
#             mystr1 = file1.readline()  # 表示一次读取一行
#             mystr2 = file2.readline()
#             if not mystr1:
#                 # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
#                 break
#
#             file3.write(mystr1[:-1])
#             file3.write("\t")
#             file3.write(mystr2)
#
#         file1.close()
#         file3.close()
#


"""atom_rASA"""
id_file = "/lustre/home/qfchen/ContactMap/surConD_id.txt"

with open(id_file,'r') as info:
    lines = info.readlines()
    for line in lines:
        pdb_id = line.split('\n')[0]
        # print(pdb_id)
        rASA_file = "/lustre/home/qfchen/ContactMap/dssp_pdb_to_rasa/cullpdb_train_rasa/" + pdb_id + '.rasa'

        pdb_file = os.path.join("/lustre/home/qfchen/ContactMap/atom/", pdb_id + '.pdb')
        save_path = os.path.join("/lustre/home/qfchen/ContactMap/SurContact/rASR_atom/", pdb_id + '.pdb')

        file1 = open(pdb_file, "r", encoding="utf-8")
        file2 = open(rASA_file, "r", encoding="utf-8")
        file3 = open(save_path, "w", encoding="utf-8")
        # n1 = len(file1.readlines())
        # n2 = len(file2.readlines())
        # if n1 == n2:
        while True:
            mystr1 = file1.readline()  # 表示一次读取一行
            mystr2 = file2.readline()
            if not mystr2:
                # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                break

            file3.write(mystr1[:-1])
            file3.write("\t")
            file3.write(mystr2)

        file1.close()
        file3.close()

