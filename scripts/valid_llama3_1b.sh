MODEL=$1
NAME=$2
OUTPUT_DIR=$3

python3 -u valid.py \
    --model_name_or_path $MODEL \
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
    --name_dataset $NAME \
