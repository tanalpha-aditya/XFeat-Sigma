#!/bin/bash
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --mem=75000
#SBATCH --time=4-00:00:00
#SBATCH --output=improvement.txt

python3 eval_megadepth.py --matcher xfeat
python3 eval_megadepth.py --matcher xfeat-star
python3 eval_megadepth.py --matcher xfeat-trasformed
python3 eval_megadepth.py --matcher xfeat-star-trasformed
python3 eval_megadepth.py --matcher xfeat-refined
python3 eval_megadepth.py --matcher xfeat-star-refined
python3 eval_megadepth.py --matcher xfeat-star-clustering
