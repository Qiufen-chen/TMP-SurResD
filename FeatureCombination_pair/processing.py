'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/12/22 18:33
@Author : Qiufen.Chen
@FileName: data_split.py
@Software: PyCharm
'''

from random import random
import torch.utils.data as Data
from torch.utils.data import DataLoader
import random
import numpy as np
import os


"""Read Data"""
class Contact():
    # def __init__(self):
    #     self.parent_path = os.path.abspath('..')  # get the current working parent directory
    #     print(self.parent_path)

    def save_file(self, li, save_path):
        str = '\n'
        f = open(save_path, "w")
        f.write(str.join(li))
        f.close()

    def read_csv(self):
        Id_path = "/lustre/home/qfchen/ContactMap/surConD_id.txt"
        # -----------------------------------------------------------------------------------------------
        # data = pd.read_csv(Id_path, usecols=[0], encoding='utf-8', header=None, keep_default_na=False)
        # data = data.values.tolist()
        #
        # dataID = []
        # for item in data:
        #     dataID.append(item[0])
        # -----------------------------------------------------------------------------------------------
        dataID = []
        with open(Id_path, 'r') as fo:
            lines = fo.readlines()
            for line in lines:
                dataID.append(line.replace('\n', ''))

        random.seed(1023)
        random.shuffle(dataID)
        nums = len(dataID)

        # 60% Train Data
        trainID = dataID[:int(0.6 * nums)]
        self.save_file(trainID, '/lustre/home/qfchen/ContactMap/SurContact/dataset/trainID.txt')
        # 20%  Validation Data
        valID = dataID[int(0.6 * nums):int(0.8 * nums)]
        self.save_file(valID, '/lustre/home/qfchen/ContactMap/SurContact/dataset/valID.txt')
        # 20%  Test Data
        testID = dataID[int(0.8 * nums):]
        self.save_file(testID, '/lustre/home/qfchen/ContactMap/SurContact/dataset/testID.txt')

        print('Trian Data: %d   Validation Data: %d  Test Data: %d' % (len(trainID), len(valID), len(testID)))

        return trainID, valID, testID

    def get_feature_label(self, ID, flag):
        onehot = []
        hmm = []
        ccmpred = []
        profold = []

        feature = []
        label = []
        label_dir = "/lustre/home/qfchen/ContactMap/SurContact/lable/"
        # get lable path
        for item in ID:
            label.append(label_dir + item + '.npy')

        if flag == 'onehotpair+hmm':
            onehot_pair_dir = "/lustre/home/qfchen/ContactMap/SurContact/onehot_pair/"
            hmm_dir = "/lustre/home/qfchen/ContactMap/SurContact/hmm/"
            for item in ID:
                onehot.append(onehot_pair_dir + item + '.npy')
                hmm.append(hmm_dir + item + '.npy')

            feature.append(onehot)
            feature.append(hmm)

            return feature, label


        elif flag == 'onehotpair+hmm+ccmpred':
            onehot_pair_dir = "/lustre/home/qfchen/ContactMap/SurContact/onehot_pair/"
            hmm_dir = "/lustre/home/qfchen/ContactMap/SurContact/hmm/"
            ccmpred_dir = "/lustre/home/qfchen/ContactMap/SurContact/ccmpred/"
            for item in ID:
                onehot.append(onehot_pair_dir + item + '.npy')
                hmm.append(hmm_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')
                
            feature.append(onehot)
            feature.append(hmm)
            feature.append(ccmpred)

            return feature, label



        elif flag == 'onehotpair+hmm+ccmpred+profold':
            onehot_pair_dir = "/lustre/home/qfchen/ContactMap/SurContact/onehot_pair/"
            hmm_dir = "/lustre/home/qfchen/ContactMap/SurContact/hmm/"
            ccmpred_dir = "/lustre/home/qfchen/ContactMap/SurContact/ccmpred/"
            profold_dir = "/lustre/home/qfchen/ContactMap/SurContact/profold/"
            for item in ID:
                onehot.append(onehot_pair_dir + item + '.npy')
                hmm.append(hmm_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')
                profold.append(profold_dir + item + '.npz')

            feature.append(onehot)
            feature.append(hmm)
            feature.append(ccmpred)
            feature.append(profold)

            return feature, label



    def main(self, flag):
        """
        :return:Train data of path, Validation data of path,Test data of path
        """
        train, val, test = self.read_csv()
        # print(len(train), len(val), len(test))
        x_train, y_train = self.get_feature_label(train, flag)
        x_val, y_val = self.get_feature_label(val, flag)
        x_test, y_test = self.get_feature_label(test, flag)

        return x_train, y_train, x_val, y_val, x_test, y_test



class mydataset(Data.Dataset):

    def __init__(self, feature_path, label_path):
        # The definition of image's path
        self.images = feature_path
        self.targets = label_path

    def __getitem__(self, index):
        img_path, lab_path = self.images[index], self.targets[index]
        # the usage of __getitem__
        img_data = np.load(img_path)
        img_data = img_data.reshape([])
        lab_data = np.load(lab_path)
        lab_data = lab_data.reshape([])

        return img_data, lab_data

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    con = Contact()
    x_train, y_train, x_val, y_val, x_test, y_test = con.main()
    train = mydataset(x_train, y_train)
    val = mydataset(x_val, y_val)
    test = mydataset(x_test, y_test)
    train_loader = DataLoader(train, batch_size=2, shuffle=False, num_workers=4)
    val_loader = DataLoader(val, batch_size=2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test, batch_size=2, shuffle=False, num_workers=2)

