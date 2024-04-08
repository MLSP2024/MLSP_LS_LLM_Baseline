#!/bin/bash
#SBATCH --job-name=LLama2-LS
#SBATCH --partition=gpu_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1

#
# Run this script once for each prompt:
#
# The first (and only) argument selects the prompt among the following:
# - only
# - only-alt
# - only-lang
# - only-alt-lang
# - mlsp
# No argument means "unihd" prompt.

# See evaluate.sh for evaluation.


prompt="$1"
if [ -z "$prompt" ]
then
	python llm_baseline.py MLSP_Data/Data/Test/[B-Z]*/*_ls_unlabelled.tsv
	python llm_baseline.py --tagalog MLSP_Data/Data/Test/Filipino/*_ls_unlabelled.tsv -o output/multilex_test_fil_tgl_ls.tsv
else
	python llm_baseline.py --tagalog --prompt "$prompt" MLSP_Data/Data/Test/[B-Z]*/*_ls_unlabelled.tsv -o "output-$prompt"
fi
