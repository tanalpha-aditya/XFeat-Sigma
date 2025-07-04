#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --time=4-00:00:00
#SBATCH --output=train_No_Synthetic_Data.txt
#SBATCH --nodelist=gnode076

python3 -m modules.training.train --megadepth_root_path /scratch/yash9439 --ckpt_save_path ./saved_checkpoints --training_type xfeat_megadepth --batch_size 16 --n_steps 100000 --lr 3e-4 --device_num 0

