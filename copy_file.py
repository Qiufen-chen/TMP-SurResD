'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/12/22 16:08
@Author : Qiufen.Chen
@FileName: copy_file.py
@Software: PyCharm
'''
import os
import shutil
import os

#id_file = "/lustre/home/qfchen/ContactMap/TMContact/TMID.txt"
# id_file = "/lustre/home/qfchen/ContactMap/1057_hhm.txt"
#
# with open(id_file,'r') as info:
#     lines = info.readlines()
#     for line in lines:
#         pdb_id = line.split('\n')[0]
#         print(pdb_id)
#
#         file = os.path.join("/lustre/home/qfchen/ContactMap/atom/", pdb_id + '.pdb')
#         save_file = os.path.join("/lustre/home/qfchen/ContactMap/SurContact/Test/deepMSA/", pdb_id + '.pdb')
#         shutil.copyfile(file, save_file)



# def CreateDir(path):
#     isExists = os.path.exists(path)
#     # 判断结果
#     if not isExists:
#         # 如果不存在则创建目录
#         os.makedirs(path)
#         print(path+' 目录创建成功')
#     else:
#         # 如果目录存在则不创建，并提示目录已存在
#         print(path+' 目录已存在')
#
#
# def CopyFile(filepath, newPath):
#     # 获取当前路径下的文件名，返回List
#     fileNames = os.listdir(filepath)
#     for file in fileNames:
#         # 将文件命加入到当前文件路径后面
#         newDir = filepath + '//' + file
#         # 如果是文件
#         if os.path.isfile(newDir):
#             print(newDir)
#             newFile = newPath + file
#             shutil.copyfile(newDir, newFile)
#         #如果不是文件，递归这个文件夹的路径
#         else:
#             CopyFile(newDir,newPath)
#
# if __name__ == "__main__":
#     path = '/home/chenqiufen/ContactData/'
#     # 创建目标文件夹
#     mkPath = '/home/chenqiufen/ContactMap/data/All_Proteins/'
#     CreateDir(mkPath)
#     CopyFile(path, mkPath)

import shutil
import os

testID = '/lustre/home/qfchen/ContactMap/surConD_id.txt'

with open(testID, 'r') as fo:
    lines = fo.readlines()
    for line in lines:
        pdb_id = line.replace('\n', '')
        
        # hmm_file = os.path.join("/lustre/home/qfchen/ContactMap/hhblits/", pdb_id + '.hhm')
        # save_file = os.path.join('/lustre/home/qfchen/ContactMap/test/hhm/', pdb_id + '.hhm')
        # shutil.copyfile(hmm_file, save_file)


        fasta_file = os.path.join("/lustre/home/qfchen/ContactMap/multi_fasta/", pdb_id + '.fasta')
        save_file = os.path.join('/lustre/home/qfchen/ContactMap/fasta/', pdb_id + '.fasta')
        shutil.copyfile(fasta_file, save_file)

        # ccmpred_file = os.path.join('/lustre/home/qfchen/ContactMap/SurContact/onehot/', pdb_id + '.npy')
        # save_file = os.path.join('/lustre/home/qfchen/ContactMap/train/onehot/',pdb_id + '.npy')
        # shutil.copyfile(ccmpred_file, save_file)

        # msa_file = os.path.join('/lustre/home/qfchen/ContactMap/ccmpred/', pdb_id + '.mat')
        # save_file = os.path.join('/lustre/home/qfchen/ContactMap/test/ccmpred/', pdb_id + '.mat')
        # shutil.copyfile(msa_file, save_file)

        # fasta_file = os.path.join("/lustre/home/qfchen/ContactMap/SurContact/lable/", pdb_id + '.npy')
        # save_file = os.path.join('/lustre/home/qfchen/ContactMap/test/label/', pdb_id + '.npy')
        # shutil.copyfile(fasta_file, save_file)

