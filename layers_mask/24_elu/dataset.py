# Author QFIUNE
# coding=utf-8
# @Time: 2022/2/26 20:07
# @File: dataset.py
# @Software: PyCharm
# @contact: 1760812842@qq.com


import torch
import feature_concat
import torch.utils.data as Data



# ----------------------------------------------------------------------------
class mydataset(Data.Dataset):
    """Create Dataset"""

    def __init__(self, feature_path, label_path):
        self.images = feature_path
        self.targets = label_path



    def __getitem__(self, index):
        pdb_id = self.images[0][index].split('/')[-1][:6]
        # print(pdb_id)
        if len(self.images) ==2:
            # print(self.images[0][index], self.images[1][index])
            if 'onehot' in self.images[0][index] and 'ccmpred' in self.images[1][index]:
                onehot_path = self.images[0][index]
                ccmpred_path = self.images[1][index]
                lable_path = self.targets[index]
                onehot = feature_concat.get_onehot(onehot_path)
                ccmpred = feature_concat.get_ccmpred(ccmpred_path)
                label = feature_concat.get_lable(lable_path)
                feature = torch.cat([onehot, ccmpred], dim=0)
                return feature, label

            if 'hmm' in self.images[0][index] and 'ccmpred' in self.images[1][index]:
                hmm_path = self.images[0][index]
                ccmpred_path = self.images[1][index]
                lable_path = self.targets[index]
                hmm = feature_concat.get_hhm(hmm_path)
                ccmpred = feature_concat.get_ccmpred(ccmpred_path)
                label = feature_concat.get_lable(lable_path)
                feature = torch.cat([hmm, ccmpred], dim=0)
                return feature, label

            if 'ccmpred' in self.images[0][index] and 'profold' in self.images[1][index]:
                ccmpred_path = self.images[0][index]
                profold_path = self.images[1][index]
                lable_path = self.targets[index]
                ccmpred = feature_concat.get_ccmpred(ccmpred_path)
                profold = feature_concat.get_profold(profold_path)
                label = feature_concat.get_lable(lable_path)
                feature = torch.cat([ccmpred, profold], dim=0)
                return feature, label

            if 'onehot' in self.images[0][index] and 'profold' in self.images[1][index]:
                onehot_path = self.images[0][index]
                profold_path = self.images[1][index]
                lable_path = self.targets[index]
                onehot = feature_concat.get_onehot(onehot_path)
                profold = feature_concat.get_profold(profold_path)
                label = feature_concat.get_lable(lable_path)
                feature = torch.cat([onehot, profold], dim=0)
                return feature, label

            if 'hmm' in self.images[0][index] and 'profold' in self.images[1][index]:
                hmm_path = self.images[0][index]
                profold_path = self.images[1][index]
                lable_path = self.targets[index]
                hmm = feature_concat.get_hhm(hmm_path)
                profold = feature_concat.get_profold(profold_path)
                label = feature_concat.get_lable(lable_path)
                feature = torch.cat([hmm, profold], dim=0)
                return feature, label

        elif len(self.images) == 3:
            # print(self.images[0][index],self.images[1][index],self.images[2][index] )
            if 'hmm' in self.images[0][index] and 'ccmpred' in self.images[1][index] and 'profold' in self.images[2][index]:
                hmm_path = self.images[0][index]
                ccmpred_path = self.images[1][index]
                profold_path = self.images[2][index]
                lable_path = self.targets[index]

                hmm = feature_concat.get_hhm(hmm_path)
                ccmpred = feature_concat.get_ccmpred(ccmpred_path)
                profold = feature_concat.get_profold(profold_path)
                label = feature_concat.get_lable(lable_path)

                feature = torch.cat([torch.cat([hmm,ccmpred], dim=0), profold], dim=0)
                # print(feature.shape)
                return feature, label

            if 'onehot' in self.images[0][index] and 'hmm' in self.images[1][index] and 'ccmpred' in self.images[2][
                index]:
                onehot_path = self.images[0][index]
                hmm_path = self.images[1][index]
                ccmpred_path = self.images[2][index]
                lable_path = self.targets[index]

                onehot = feature_concat.get_onehot(onehot_path)
                hmm = feature_concat.get_hhm(hmm_path)
                ccmpred = feature_concat.get_ccmpred(ccmpred_path)
                label = feature_concat.get_lable(lable_path)

                feature = torch.cat([torch.cat([onehot, hmm], dim=0), ccmpred], dim=0)
                # print(feature.shape)
                return feature, label, pdb_id

        elif len(self.images) == 4:
            onehot_path = self.images[0][index]
            hmm_path = self.images[1][index]
            ccmpred_path = self.images[2][index]
            profold_path = self.images[3][index]
            lable_path = self.targets[index]

            onehot = feature_concat.get_onehot(onehot_path)
            hmm = feature_concat.get_hhm(hmm_path)
            ccmpred = feature_concat.get_ccmpred(ccmpred_path)
            profold = feature_concat.get_profold(profold_path)
            label = feature_concat.get_lable(lable_path)

            feature = torch.cat([torch.cat([torch.cat([onehot, hmm], dim=0), ccmpred], dim=0), profold], dim=0)
            # print(feature.shape)

            return feature, label

    def __len__(self):
        return len(self.images[0])
