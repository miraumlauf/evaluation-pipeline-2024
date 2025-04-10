#!/bin/bash
#SBATCH --job-name=e_3fh_512
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:07:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1

# added enviroment with installed requirements
source activate eval_24

#downloading data
#python -m ewok.dl_and_filter

#MODEL_PATH=$1
MODEL_PATH="../lingua_3fh_causal_512_15300"
MODEL_BASENAME=$(basename $MODEL_PATH)

python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,backend="causal",trust_remote_code=True \
    --tasks ewok_filtered \
    --device cuda:0 \
    --batch_size 128 \
    --log_samples \
    --output_path results/ewok/${MODEL_BASENAME}/ewok_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.

echo "Finished Evaluation with ewok"