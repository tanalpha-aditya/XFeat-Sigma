#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --time=4-00:00:00
#SBATCH --output=train_megadepth.txt
#SBATCH --nodelist=gnode076

python3 -m modules.training.train --megadepth_root_path /scratch/yash9439 --ckpt_save_path ./megadepth_only_checkpoints --training_type xfeat_megadepth --batch_size 16 --n_steps 100000 --lr 3e-4 --device_num 0

