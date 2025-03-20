names=("common-reasoning" "BoolQ" "PIQA" "SIQA" "hellaswag" "winogrande" "ARC-E" "ARC-C" "OBQA")

MODEL=$1
CONFIG_PATH=$2
DIR=$3

for name in "${names[@]}"; do
    echo "Processing $name..."
    ./scripts/valid_llama3_1b.sh "$MODEL" $CONFIG_PATH $name "$DIR/$name"
done

echo "All processes completed."