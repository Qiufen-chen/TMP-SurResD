#!/bin/bash
#SBATCH --job-name=HMM_CCM_18
#SBATCH --partition=cpuPartition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
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
module load python/3.6.10
python hmm_ccmpred_train.py > hmm_ccmpred.log

echo "++++++++++++++++++++++++++++++++++++++++"
echo "process end at : "
date
sleep 300
