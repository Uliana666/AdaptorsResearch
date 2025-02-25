MODEL="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_LOGS="./logs/llama3_1b/logs"

python3 -u valid.py \
    --output_dir $OUTPUT_LOGS \
    --model_name_or_path $MODEL \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 5 \
    --do_train False \
    --do_eval True \
    --fp16 True \
    --save_total_limit 1 \
    --logging_steps 1 \
    --report_to "wandb" \
    --logging_steps 1 \
    --logging_dir $OUTPUT_LOGS \
    --count_examples 15000 \
    --start_token 374 \
    --model_max_length 1024 \
    --mode "base" \