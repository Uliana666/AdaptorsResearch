MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
MODEL_PATH=$1
CONFIG_PATH=$2
DATASET=$3
OUTPUT_DIR=$4

python3 -u valid.py \
    --path $MODEL_PATH \
    --name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 5 \
    --do_train False \
    --do_eval True \
    --fp16 True \
    --logging_steps 1 \
    --report_to "wandb" \
    --start_token 374 \
    --model_max_length 1024 \
    --name_dataset $DATASET \
    --config_path $CONFIG_PATH \