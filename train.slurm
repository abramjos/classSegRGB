#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/COCOOneLossBackAvgMask.txt
#SBATCH --job-name=COCOOneLossAvgMask

# Give this process 1 task (per GPU, but only one GPU), then assign eight 8per task
# (so 8 cores overall).  Then enforce that slurm assigns only CPUs that are on the
# socket closest to the GPU you get.

# If you want two GPUs:
# #SBATCH --ntasks=2
# #SBATCH --gres=gpu:2


# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
module load anaconda
module load cuda/cuda-10.1
module load gcc/gcc-7.1.0


source activate detectron


echo
echo " Started Training"
echo "=================================================================================================="
python -W ignore trainOneLossPass.py --data_dir /groups/amahalanobis/MS-COCO/train2017 --exp_dir ./OneLossBack/COCORGBAvgMaskXLossOne --logfile './Log/COCOOneLossBackRGBAvgMaskX.txt' --n_epochs 2000
echo "=================================================================================================="

echo "Training Complete"


# You're done!
echo "Ending script..."
date

