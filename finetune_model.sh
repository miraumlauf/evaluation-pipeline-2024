#!/bin/bash
#SBATCH --job-name=2fh_256
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1



# added enviroment with installed requirements
source activate eval_24


#MODEL_PATH=$1
MODEL_PATH="../lingua_2fh_sequence_256_15300"

LR=${2:-5e-5}           # default: 5e-5
PATIENCE=${3:-3}       # default: 3
BSZ=${4:-64}            # default: 64
MAX_EPOCHS=${5:-10}     # default: 10
SEED=${6:-12}           # default: 12

# NOTE: if you've already run finetuning and just want to generate predictions,
# you can set `--model_name_or_path "results/finetune/$model_basename/$TRAIN_NAME/"`
# and remove the `--do_train` and `--do_eval` arguments.

model_basename=$(basename $MODEL_PATH)
RUN_SUFFIX="_2" 
# removed mnli and mnli-mm 
for task in {boolq,cola,mrpc,multirc,qnli,qqp,rte,sst2,wsc}; do
    echo "Running fine-tuning for task: $task"  
	if [[ $task = "mnli-mm" ]]; then
		TRAIN_NAME="mnli"
		VALID_NAME="mnli-mm"
		DO_TRAIN=False
		MODEL_PATH_FULL="results/finetune/$model_basename/$TRAIN_NAME/"
	else
		TRAIN_NAME=$task
		VALID_NAME=$task
		DO_TRAIN=True
		MODEL_PATH_FULL=$MODEL_PATH
	fi

	#mkdir -p results/finetune/$model_basename/$task/
	RESULTS_DIR=results/finetune/${model_basename}${RUN_SUFFIX}/$task/
	mkdir -p $RESULTS_DIR

	python finetune_classification.py \
	  --model_name_or_path $MODEL_PATH_FULL \
	  --output_dir $RESULTS_DIR \
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
# If you run into memory issues, try reducing $BSZ or reducing `--max_seq_length` first.