# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:30:54 2020

@author: YihangBao
"""
import os

f = open('./fasta/cullpdb_train.fasta','r')
for i in f:
    if i[0]!='>':
        continue
    try:
        name = i[1]+i[2]+i[3]+i[4]
        name = name.lower()
        os.system('dssp -i cullpdb_train_pdb/' + name +'.pdb' + ' -o cullpdb_train_dssp/' + name + '.dssp')
    except:
        fx = open('./ERROR_AutoDssp.txt','a+')
        fx.write(name+'\n')
    
