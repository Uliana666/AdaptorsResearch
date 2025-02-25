BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_MODEL="./logs/llama3_1b/lora/model"
OUTPUT_LOGS="./logs/llama3_1b/lora/logs"

python3 -u train.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT_MODEL \
    --rank 8 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --do_train True \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_steps 50 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fp16 True \
    --optim "adamw_torch" \
    --report_to "wandb" \
    --logging_dir $OUTPUT_LOGS \
    --count_examples 15000 \
    --start_token 374 \
    --model_max_length 1024 \
    --mode "lora" \