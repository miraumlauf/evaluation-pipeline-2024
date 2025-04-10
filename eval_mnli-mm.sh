#!/bin/bash
#SBATCH --job-name=mnli-mm-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=00:10:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1

source activate eval_24

# Define paths
MODEL_SOURCE_DIR="../lingua_3fh_sequence_128_15300_mnli"
MODEL_PATH_FULL="results/finetune/lingua_3fh_sequence_128_15300_mnli/mnli/checkpoint-30680"
OUTPUT_DIR="results/finetune/lingua_3fh_sequence_128_15300_mnli/mnli-mm/"

# Ensure output dir exists
mkdir -p $OUTPUT_DIR

# ✅ Copy required custom model files into the checkpoint directory
cp $MODEL_SOURCE_DIR/lingua_model.py $MODEL_PATH_FULL/
cp $MODEL_SOURCE_DIR/lingua_config.py $MODEL_PATH_FULL/
cp $MODEL_SOURCE_DIR/base_transformer.py $MODEL_PATH_FULL/

# ✅ Run evaluation
python finetune_classification.py \
  --model_name_or_path $MODEL_PATH_FULL \
  --output_dir $OUTPUT_DIR \
  --train_file evaluation_data/glue_filtered/mnli.train.jsonl \
  --validation_file evaluation_data/glue_filtered/mnli-mm.valid.jsonl \
  --do_train False \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --overwrite_output_dir \
  --seed 12 \
  --trust_remote_code
