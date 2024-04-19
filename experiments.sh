#!/bin/bash
#SBATCH --job-name=LLama2-LS
#SBATCH --partition=gpu_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1

# Default prompt is "only". Supply a single argument "unihd" to run with the "unihd"
# prompt instead.

# See evaluate.sh for evaluation.


prompt="$1"
if [ -z "$prompt" ]
then
	python llm_baseline.py MLSP_Data/Data/Test/[B-Z]*/*_ls_unlabelled.tsv
else
	python llm_baseline.py --prompt "$prompt" MLSP_Data/Data/Test/[B-Z]*/*_ls_unlabelled.tsv -o "output_$prompt"
fi
