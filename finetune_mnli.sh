#!/bin/bash
#SBATCH --job-name=2fh_256
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1

# Activate conda environment
source activate eval_24

# === Paths & Setup ===
MODEL_SOURCE_DIR="../lingua_2fh_sequence_256_15300_mnli"
MODEL_BASE_NAME="lingua_2fh_sequence_256_15300_mnli"
MODEL_PATH="$MODEL_SOURCE_DIR"


#CHECKPOINT_DIR="results/finetune/${MODEL_BASE_NAME}/mnli"
RUN_SUFFIX="_2"
CHECKPOINT_DIR="results/finetune/${MODEL_BASE_NAME}${RUN_SUFFIX}/mnli"  # === MODIFIED ===


# === Hyperparameters ===
LR=${2:-5e-5}           # default: 5e-5
PATIENCE=${3:-3}       # default: 3
BSZ=${4:-64}            # default: 64
MAX_EPOCHS=${5:-10}     # default: 10
SEED=${6:-12}           # default: 12

# === 1. Train and Evaluate on MNLI ===
echo "🔧 Training and evaluating on MNLI..."

mkdir -p $CHECKPOINT_DIR

python finetune_classification.py \
  --model_name_or_path $MODEL_PATH \
  --output_dir $CHECKPOINT_DIR \
  --train_file evaluation_data/glue_filtered/mnli.train.jsonl \
  --validation_file evaluation_data/glue_filtered/mnli.valid.jsonl \
  --do_train True \
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

# === 2. Find the latest checkpoint ===
echo "🔍 Locating latest checkpoint..."

LATEST_CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/checkpoint-* | sort -V | tail -n 1)
echo "✅ Found checkpoint: $LATEST_CHECKPOINT"

# === 3. Copy Custom Model Code into Checkpoint ===
echo "📦 Copying custom model code into checkpoint..."

cp $MODEL_SOURCE_DIR/lingua_model.py $LATEST_CHECKPOINT/
cp $MODEL_SOURCE_DIR/lingua_config.py $LATEST_CHECKPOINT/
cp $MODEL_SOURCE_DIR/base_transformer.py $LATEST_CHECKPOINT/

# === 4. Evaluate MNLI-MM using latest checkpoint ===
echo "🔍 Evaluating on MNLI-MM..."

# MNLI_MM_OUTPUT_DIR="results/finetune/${MODEL_BASE_NAME}/mnli-mm"
MNLI_MM_OUTPUT_DIR="results/finetune/${MODEL_BASE_NAME}${RUN_SUFFIX}/mnli-mm"  # === MODIFIED ===
mkdir -p $MNLI_MM_OUTPUT_DIR


python finetune_classification.py \
  --model_name_or_path $LATEST_CHECKPOINT \
  --output_dir $MNLI_MM_OUTPUT_DIR \
  --train_file evaluation_data/glue_filtered/mnli.train.jsonl \
  --validation_file evaluation_data/glue_filtered/mnli-mm.valid.jsonl \
  --do_train False \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --overwrite_output_dir \
  --seed $SEED \
  --trust_remote_code

echo "🎉 All done!"
