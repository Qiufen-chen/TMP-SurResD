# pdb_to_rasa使用说明
## 文件准备：
1、将三个py文件与run.sh文件置于同一目录下，保持原文件名
2、在这一目录下创建名称为fasta文件夹，文件夹下创建文件名为cullpdb_train.fasta的fasta文件，里面存放所有的fasta数据（关于fasta内容的注意点见下方特别说明部分）
3、在py文件所在的目录下创建cullpdb_train_pdb的文件夹，里面放入所有pdb文件，命名均为小写
4、在py文件所在目录下创建cullpdb_train_No	cullpdb_train_dssp	cullpdb_train_rasa三个空文件夹

## 使用方式
直接运行run.sh脚本文件，运行方式（二选一）：
1、直接运行，进入到所在目录后，命令行执行：./run.sh
2、后台运行，进入到所在目录后，命令行执行：nohup ./run.sh &
之后在cullpdb_train_No文件下的是相应的标号，cullpdb_train_dssp下的是相应dssp文件，cullpdb_train_rasa下的是最终得到的rasa

## 特别说明
1、本程序所有标号与缺失均以pdb文件为基准，如pdb从pdbtm上下载，则rasa必然与pdbtm上提供的fasta序列对应
2、本程序中fasta文件下中的cullpdb_train.fasta文件只需保证‘>’后的数据格式正确，程序中的序列直接从pdb提取，所以用不到这里的fasta序列。但是要注意保证'>'相关的数据格式为‘>1pb4_C’，蛋白名小写，链名大写，中间用下划线隔开
3、为方便程序运行，所有文件名和文件夹名都最好改成这里提到的（否则需要修改程序中的路径，由于程序中涉及到路径的地方很多，修改不太方便）

