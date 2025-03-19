BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR=$1
CONFIG_PATH=$2

python3 -u calc_metrics.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 2 \
    --fp16 True \
    --start_token 374 \
    --model_max_length 1024 \
    --report_to "tensorboard" \
    --config_path $CONFIG_PATH