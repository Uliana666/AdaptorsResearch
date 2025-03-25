import argparse
from lib import Datasets
from datasets import Dataset, DatasetDict


def create_dataset(dataset_names, num_examples, data_split="train", seed=42):
    examples = []

    for name in dataset_names:
        example = Datasets.LoadCommonReasoning(type=data_split, count=num_examples, name=name, seed=seed)
        examples.extend(example["dataset"])

    keys = examples[0].keys()
    data_dict = {key: [item[key] for item in examples] for key in keys}
    ds = Dataset.from_dict(data_dict)
    print(ds)
    return ds

def main():
    parser = argparse.ArgumentParser(description='Load examples from multiple datasets.')
    parser.add_argument('--num_examples', type=int, default=2, help='Number of examples to load from each dataset')
    parser.add_argument('--data_split', type=str, default='train', help='Data split to use (e.g., train, validation, test)')
    parser.add_argument('--output_dir', type=str, default='./datasets/samples_dataset', help='Directory to save the combined dataset')
    args = parser.parse_args()

    dataset_names = ["BoolQ", "PIQA", "SIQA", "hellaswag", "winogrande", "ARC-E", "ARC-C", "OBQA"]

    combined_dataset = create_dataset(dataset_names, args.num_examples, data_split=args.data_split, seed=69)

    combined_dataset_dict = DatasetDict({"train": combined_dataset})
    combined_dataset_dict.save_to_disk(args.output_dir, num_shards={'train': 2})

if __name__ == "__main__":
    main()