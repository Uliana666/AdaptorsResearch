datasets=('BoolQ' 'PIQA' 'SIQA' 'hellaswag' 'winogrande' 'ARC-E' 'ARC-C' 'OBQA')

for dataset in "${datasets[@]}"; do
    python3 generate_common_reasoning.py config-llama3-1b.yaml ../datasets/"$dataset" -s "$dataset"
done