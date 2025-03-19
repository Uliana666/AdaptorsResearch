BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_MODEL=$1
CONFIG_PATH=$2

python3 -u train.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT_MODEL \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --do_train True \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fp16 True \
    --optim "adamw_torch" \
    --report_to "wandb" \
    --count_examples 400 \
    --start_token 374 \
    --model_max_length 1024 \
    --config_path $CONFIG_PATH \