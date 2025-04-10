#!/bin/bash
#SBATCH --job-name=b_3fh_512
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1

# added enviroment with installed requirements
source activate eval_24

#MODEL_PATH=$1
MODEL_PATH="../lingua_3fh_causal_512_15300"
MODEL_BASENAME=$(basename $MODEL_PATH)

python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,backend="causal",trust_remote_code=True \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files!!

echo "Finished try run"
