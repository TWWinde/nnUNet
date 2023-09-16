#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=nnUNet
#SBATCH --output=nnUNet%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
# SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --gpus=rtx_a5000:1
# SBATCH --gpus=geforce_rtx_2080ti:1
# SBATCH --gpus=geforce_gtx_titan_x:1




# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
pyenv activate venv
module load cuda/11.3
CUDA_VISIBLE_DEVICES=0
# Run your python code
# For single GPU use this
export nnUNet_raw="/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_raw"
export nnUNet_preprocessed="/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_preprocessed"
export nnUNet_results="/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_results"

#python /no_backups/s1449/nnUNetFrame/nnUNet/nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py

#nnUNetv2_plan_and_preprocess -d 522 --verify_dataset_integrity

#nnUNetv2_preprocess -d 522

nnUNetv2_train 522 2d 0


