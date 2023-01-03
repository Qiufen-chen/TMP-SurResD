# -*- coding:utf-8 -*-
'''
	File Name：     runHHblits
	Description :   generate HHblits features of fasta files
	Author :        Liu Zhe
	date：          2020/2/21
'''
import os

class HHblits():
    def runHHblits(self, fastapath, outpath):
        names = [name for name in os.listdir(fastapath) if os.path.isfile(os.path.join(fastapath + '//', name))]
        for each_item in names:
            pdb_id = each_item.split('.')[0]
            postfix = each_item.split('.')[1]
            if postfix == 'fasta':
                tool_path = '/lustre/software/anaconda/anaconda3-2019.10-py37/envs/alphafold2/bin/hhblits'
                db_path = '/lustre/software/alphafold/download/uniclust30/uniclust30_2018_08/uniclust30_2018_08'
                print(each_item)
                '''
                database used: uniclust30_2018_08
                link: http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/
                '''
                cmd = tool_path + ' -i '+ fastapath + '/' + each_item + ' -ohhm ' + outpath + '/' + pdb_id + '.hhm -d ' + db_path
                os.system(cmd)



if __name__ == '__main__':
    '''
    samples:
    fastapath = '/home/liuzhe002/HHblits/HHFasta'
    outpath = '/home/RaidDisk/liuzhe002/HHResult'
    You can also check the structure and format of used fasta files in my folder : /home/liuzhe002/HHblits/HHFasta
    Warning : the permissions issue has not been resolved, please use the filepath under /home/RaidDisk/ as your outpath
    '''

    fastapath = '/lustre/home/qfchen/Amyloid_search/data/input/fasta_arp_information/'
    outpath = '/lustre/home/qfchen/Amyloid_search/data/output_hhblits/arp_information/'

    hh = HHblits()
    hh.runHHblits(fastapath, outpath)