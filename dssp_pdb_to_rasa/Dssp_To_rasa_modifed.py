# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:30:54 2020

@author: YihangBao
"""
dic ={
      'A':121.0,
      'R':265.0,
      'N':187.0,
      'D':187.0,
      'C':148.0,
      'E':214.0,
      'Q':214.0,
      'G':97.0,
      'H':216.0,
      'I':195.0,
      'L':191.0,
      'K':230.0,
      'M':203.0,
      'F':228.0,
      'P':154.0,
      'S':143.0,
      'T':163.0,
      'W':264.0,
      'Y':255.0,
      'V':165.0,
      'B':187.0,
      'Z':214.0
      }


def extract(name,need):
    try:
        dss = open('./cullpdb_train_dssp/' + name + '.dssp','r')
        flag = False
        firche = True
        nume = 0
        numc = 0
        numa = 0
        loc_chain=0
        loc_re=0
        loc_cc=0
        done = False
        kk1 = 0
        kk2 = 0
        with open('./cullpdb_train_No/' + name+'_'+need+'.no') as no_file:
            noline = no_file.readline()
            kk1 = 1
            for i in dss:
                if flag and i[loc_chain]!=need:
                    if i[loc_re]=='!':
                        if i[loc_re+1]=='*':
                            firche= True
                            if done:
                                break
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                if flag and firche:
                    firche = False
                    done = True
                
                if flag:
                    if not noline or noline[0]==' ' or noline[0]=='\n':
                        break
                    temp_no = int(i[5:11].strip())
                    stan_no = int(noline[:].strip())
                    if temp_no < stan_no:
                        continue
                    elif temp_no > stan_no:
                        while noline and temp_no > stan_no:
                            w = open('./cullpdb_train_rasa/' + name+'_'+need+'.rasa','a+')
                            kk2 +=1
                            w.write('0.0' + '\n')
                            noline = no_file.readline()
                            kk1 += 1
                            stan_no = int(noline[:].strip())
                    else:
                        pass
                    if not noline or noline[0]==' ' or noline[0]=='\n':
                        break
                    noline = no_file.readline()
                    kk1 += 1
                    wei = 1
                    pos = loc_cc
                    ans=0
                    while 1:
                        if i[pos].isdigit()==False:
                            break
                        ans += int(i[pos])*wei
                        wei *= 10
                        pos -= 1
                    if i[loc_re] in dic:
                        ans = ans/dic[i[loc_re]]
                    else:
                        ans = ans/dic['C']
                    w = open('./cullpdb_train_rasa/' + name+'_'+need+'.rasa','a+')
                    kk2+=1
                    w.write(str(ans))
                    w.write('\n')
                    continue
                    
                if i[1]=='#' or i[2]=='#' or i[3]=='#':
                    flag=True
                    for j in range(45):
                        
                        if i[j]=='E':
                            nume+=1
                        if i[j]=='C':
                            numc+=1
                        if i[j]=='A':
                            numa+=1
                        if nume==2:
                            loc_chain = j
                            nume+=1
                        if numa==1:
                            loc_re = j
                            numa+=1
                        if numc==3:
                            loc_cc = j
                            break
            while noline and noline[0]!='\n' and noline[0]!=' ':
                w = open('./cullpdb_train_rasa/' + name+'_'+need+'.rasa','a+')
                kk2+=1
                w.write('0.0' + '\n')
                noline = no_file.readline()
                kk1 += 1
            
    except:
        w = open('./ERROR_rasa.txt','a+')
        w.write(name+'\n')
    if kk1-1 != kk2:
        w = open('./ERROR_rasa.txt','a+')
        w.write(name+'\n')

f = open('./fasta/cullpdb_train.fasta','r')
for item in f:
    if item[0]!='>':
        continue
    s = item[1]+item[2]+item[3]+item[4]
    s = s.lower()
    print(s)
    extract(s, item[6])
        