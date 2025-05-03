#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gnode076
#SBATCH --output=Stdout_xfeat.txt

python3 -m modules.training.train --megadepth_root_path /scratch/yash9439 --ckpt_save_path ./megadepth_only_checkpoints --training_type xfeat_megadepth --batch_size 16 --n_steps 100000 --lr 3e-4 --device_num 0
