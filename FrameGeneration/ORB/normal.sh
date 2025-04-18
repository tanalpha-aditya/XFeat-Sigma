#!/bin/bash
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=4-00:00:00
#SBATCH --output=output2.txt
#SBATCH --nodelist=gnode081

python testVideo.py input/"Fate Zero -s01ed1-test-540p.mkv"
