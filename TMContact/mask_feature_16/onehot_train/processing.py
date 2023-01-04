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
        Id_path = "/lustre/home/qfchen/ContactMap/TMConD_id.txt"
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
        self.save_file(trainID, '/lustre/home/qfchen/ContactMap/TMContact/dataset/trainID.txt')
        # 20%  Validation Data
        valID = dataID[int(0.6 * nums):int(0.8 * nums)]
        self.save_file(valID, '/lustre/home/qfchen/ContactMap/TMContact/dataset/valID.txt')
        # 20%  Test Data
        testID = dataID[int(0.8 * nums):]
        self.save_file(testID, '/lustre/home/qfchen/ContactMap/TMContact/dataset/testID.txt')

        print('Trian Data: %d   Validation Data: %d  Test Data: %d' % (len(trainID), len(valID), len(testID)))

        return trainID, valID, testID

    def get_feature_label(self, ID, flag):

        if flag == 'onehot':
            feature_dir = "/lustre/home/qfchen/ContactMap/TMContact/onehot/"

        elif flag == 'hmm':
            feature_dir = "/lustre/home/qfchen/ContactMap/TMContact/hhblits/"

        elif flag == 'ccmpred':
            feature_dir ="/lustre/home/qfchen/ContactMap/TMContact/ccmpred/"

        elif flag == 'profold':
            feature_dir = "/lustre/home/qfchen/ContactMap/TMContact/profold/"

        label_dir = "/lustre/home/qfchen/ContactMap/TMContact/new_lable/"

        # Train Data saved in feature_connection and label, actually the path
        feature = []
        label = []
        for item in ID:
            feature.append(feature_dir + item + '.npy')
            label.append(label_dir + item + '.npy')
        return feature, label

    def main(self, flag):
        """
        :return:Train data of path, Validation data of path,Test data of path
        """
        train, val, test = self.read_csv()
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

