#!/bin/bash
#SBATCH --account=neuro
#SBATCH --partition=ihub
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=4-00:00:00
#SBATCH --output=train_megadepth.txt
#SBATCH --nodelist=gnode094

python3 -m modules.training.train --megadepth_root_path /scratch/narasimha.pai --synthetic_root_path /scratch/narasimha.pai/coco/train2017  --ckpt_save_path ./megadepth_only_checkpoints --training_type xfeat_joint_kp --batch_size 8 --n_steps 220000 --lr 3e-4 --device_num 0
