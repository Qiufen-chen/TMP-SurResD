import os

"""将多个fasta文件中合并到一个文件中"""
dir_in = '/lustre/home/qfchen/ContactMap/multi_fasta/'
save_path = '/lustre/home/qfchen/ContactMap/dssp_pdb_to_rasa/fasta/cullpdb_train.fasta'
save_file = open(save_path, "w", encoding="utf-8")
for (root, dirs, files) in os.walk(dir_in):
    for file in files:
        fasta_file = os.path.join(root, file)
        with open(fasta_file) as fo:
            lines = fo.readlines()
            for line in lines:
                save_file.write(line)
            # file.write('\n')

                        