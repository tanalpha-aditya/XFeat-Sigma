#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=4-00:00:00
#SBATCH --output=output2.txt
#SBATCH --nodelist=gnode085

python3 xfeat_trial.py ../input/Fate\ Zero\ -s01op1-test-540p.mp4 ../output3/try069.mp4
