#!/bin/bash
#SBATCH --job-name=onehot
#SBATCH --partition=cpuPartition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --error=%j.err
#SBATCH --output=%j.out
##########################################
I=`whoami`
CURDIR=`pwd`
rm -rf $CURDIR/nodelist.$SLURM_JOB_ID
NODES=`scontrol show hostnames $SLURM_JOB_NODELIST`
for i in $NODES
do
echo "$i:$SLURM_NTASKS_PER_NODE" >> $CURDIR/nodelist.$SLURM_JOB_ID
done
echo $SLURM_NPROCS

echo "process will start at : "
date
echo "++++++++++++++++++++++++++++++++++++++++"

module load pytorch/pytorch-gpu-1.10.0-py37
python onehot_train.py  > onehot.log

rm -rf /tmp/nodefile.$$
echo "Your job is done."
exit 0