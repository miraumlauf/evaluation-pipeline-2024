#!/bin/bash
#SBATCH --job-name=ntp
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=00:30:00
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:1

# Activate environment with installed requirements
source activate eval_24

MODEL_PATH="../ntp_lingua_3fh_sequence_128_15300"  # Model for all other tasks
LR=${2:-5e-5}           # default: 5e-5
PATIENCE=${3:-3}       # default: 3
BSZ=${4:-64}            # default: 64
MAX_EPOCHS=${5:-10}     # default: 10
SEED=${6:-12}           # default: 12

# NOTE: if you've already run finetuning and just want to generate predictions,
# you can set `--model_name_or_path "results/finetune/$model_basename/$TRAIN_NAME/"`
# and remove the `--do_train` and `--do_eval` arguments.

# Remove mnli and mnli-mm from the task list
model_basename=$(basename $MODEL_PATH)
for task in {boolq,cola,mrpc,multirc,qnli,qqp,rte,sst2,wsc}; do  # Removed mnli and mnli-mm
	TRAIN_NAME=$task
	VALID_NAME=$task
	DO_TRAIN=True
	MODEL_PATH_FULL=$MODEL_PATH

    # Keep the same directory structure for results
    mkdir -p results/finetune/$model_basename/$task/

    python finetune_classification.py \
      --model_name_or_path $MODEL_PATH_FULL \
      --output_dir results/finetune/$model_basename/$task/ \
      --train_file evaluation_data/glue_filtered/$TRAIN_NAME.train.jsonl \
      --validation_file evaluation_data/glue_filtered/$VALID_NAME.valid.jsonl \
      --do_train $DO_TRAIN \
      --do_eval \
      --do_predict \
      --max_seq_length 128 \
      --per_device_train_batch_size $BSZ \
      --learning_rate $LR \
      --num_train_epochs $MAX_EPOCHS \
      --patience $PATIENCE \
      --evaluation_strategy epoch \
      --save_strategy epoch \
      --overwrite_output_dir \
      --seed $SEED \
      --trust_remote_code
done

# Add `--trust_remote_code` if you need to load custom config/model files.
# If you run into memory issues, try reducing $BSZ or decreasing `--max_seq_length` first.
