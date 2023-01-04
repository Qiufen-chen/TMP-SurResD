# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:30:54 2020

@author: YihangBao
"""
residue = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 
                'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 
                'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}

def extract(pdbid,chainId):
    with open('./cullpdb_train_pdb/' + pdbid + '.pdb') as pdb_file:
        model = 0
        flag_resName = ''
        flag_resNo = ''
        pdbline = pdb_file.readline()
        while pdbline:
            if pdbline[:6] == 'ENDMDL' and model == 1:
                break
            if pdbline[:6] != 'ATOM  ' and pdbline[:6] != 'HETATM':
                pdbline = pdb_file.readline()
                continue
            if pdbline[21] != chainId or pdbline[12:16] != ' CA ':
                pdbline = pdb_file.readline()
                continue
            model = 1
            if pdbline[17:20] in residue.keys():
                res_name = residue[pdbline[17:20]]
            else:
                res_name = 'X'
            res_no = pdbline[22:26].strip()
            if flag_resName != res_name or flag_resNo != res_no:
                w = open('./cullpdb_train_No/' + pdbid+'_'+chainId+'.no','a+')
                w.write(res_no + '\n')
            flag_resName = res_name
            flag_resNo = res_no
            pdbline = pdb_file.readline()
            
f = open('./fasta/cullpdb_train.fasta','r')
for item in f:
    if item[0]!='>':
        continue
    s = item[1]+item[2]+item[3]+item[4]
    s = s.lower()
    extract(s, item[6])
