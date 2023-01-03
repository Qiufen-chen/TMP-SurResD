'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2022/3/20 21:57
@Author : Qiufen.Chen
@FileName: runDEEPCON.py
@Software: PyCharm
'''

import os

def rundeepCon(msa_path, out_path):
    for (root, dirs, files) in os.walk(msa_path):
        for file_name in files:
            pdb_id = file_name.split('.')[0]
            msa_file = os.path.join(root, file_name)
            print(msa_file)

            cmd = 'python ' +\
                  "/lustre/home/qfchen/ContactMap/SurContact/Test/DEEPCON-master/deepcon-covariance/deepcon-covariance.py" + \
                  ' --aln ' + msa_file + \
                  ' --rr ' + out_path + pdb_id + '.rr'

            os.system(cmd)


if __name__ == '__main__':
    fasta_path = "/lustre/home/qfchen/ContactMap/SurContact/Test/MSA_test/"
    out_path = "/lustre/home/qfchen/ContactMap/SurContact/Test/DEEPCON_result/"
    rundeepCon(fasta_path, out_path)

