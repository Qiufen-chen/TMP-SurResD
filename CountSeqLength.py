#!/usr/bin/env python
#coding:utf-8


import os

dir1 = "/lustre/home/qfchen/ContactMap/TMContact/fasta/"          #文件夹名称
# dir1 = "/home/chenqiufen/ContactMap/Zhaozhao/fasta_one/"
if os.path.exists(dir1):
    print("dir exists")
else:
    print("no exists")
    
filenum = 0   
l0l1 = 0
l1l2 = 0
l2l3 = 0
l3l4 = 0
l4l5 = 0
l5l6 = 0
l6l7 = 0
l7l8 = 0
l8l9 = 0
l9ll = 0
l10l11 = 0
l11l12 = 0

for (root, dirs, files) in os.walk(dir1):  #列出windows目录下的所有文件和文件名
    for filename1 in files:
        filenum = filenum + 1

        # 如果文件有很多行，用该段代码
        with open(os.path.join(root, filename1), 'r') as f1:
            length = 0
            # for i in f1:
            #     num = num+1
            # print(filename1, num)
            lines = f1.readlines()
            for line in lines:
                if line.startswith('>'):
                    pass
                else:
                    length = len(line)
                    print(filename1, length)

        # 如果文件里只有一行，则用该段代码
        # with open(os.path.join(root, filename1), 'r') as f1:
        #     line = f1.readline()
        #     num = len(line)
        #     print(filename1, num)


        if length > 0 and length < 100:
            l0l1 = l0l1 + 1
        if length >=100 and length < 200:
            l1l2 = l1l2 + 1
        if length >= 200 and length < 300:
            l2l3 = l2l3 + 1
        if length >= 300 and length < 400:
            l3l4 = l3l4 + 1
        if length >= 400 and length < 500:
            l4l5 = l4l5 + 1
        if length >= 500 and length < 600:
            l5l6 = l5l6 + 1
        if length >= 600 and length < 700:
            l6l7 = l6l7 + 1
        if length >= 700 and length < 800:
            l7l8 = l7l8 + 1
        if length >= 800 and length < 900:
            l8l9 = l8l9 + 1
        if length >= 900 and length < 1000:
            l9ll = l9ll + 1
        if length >= 1000 and length < 1100:
            l10l11 = l10l11 + 1
        if length >= 1100:
            l11l12 = l11l12 + 1
            
            
    print("********************************************************")
    print(filenum)
    print("[30,100)", l0l1)
    print("[100,200)", l1l2)
    print("[200,300)", l2l3)
    print("[300,400)", l3l4)
    print("[400,500)", l4l5)
    print("[500,600) ", l5l6)
    print("[600,700) ", l6l7)
    print("[700,800) ", l7l8)
    print("[800,900) ", l8l9)
    print("[900,1000) ", l9ll)

 
    
                   







