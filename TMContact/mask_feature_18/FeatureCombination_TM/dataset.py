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
        if len(self.images) == 2:
            onehot_path = self.images[0][index]
            hmm_path = self.images[1][index]
            lable_path = self.targets[index]

            onehot = feature_concat.get_onehot(onehot_path)
            hmm = feature_concat.get_hhm(hmm_path)
            label = feature_concat.get_lable(lable_path)

            feature = torch.cat([onehot, hmm], dim=0)

            # print(feature.shape)
            return feature, label

        elif len(self.images) == 3:
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
            return feature, label

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
