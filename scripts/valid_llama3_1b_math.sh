MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
MODEL_PATH=$1
CONFIG_PATH=$2
DATASET=$3
OUTPUT_DIR=$4

python3 -u valid_math.py \
    --path $MODEL_PATH \
    --name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --name_dataset $DATASET \
    --config_path $CONFIG_PATH \
    --count_examples 10 \