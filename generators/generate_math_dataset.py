import datasets
from datasets import load_dataset, DatasetDict, Dataset

from transformers import AutoTokenizer

from omegaconf import OmegaConf

import functools

import sys
import argparse


def _get_MATH_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request."},
        {"role": "user", "content": example['query']},
    ]
    start_index = example['response'].find("The answer is:")
    
    assert(start_index != -1)
    
    start_index += len("The answer is: ")

    correct_answer = example['response'][start_index:]
    
    instructions_ans = [
        {"role": "assistant", "content": example['response']}
    ]
    instructions_wa = [
        {"role": "assistant", "content": f""}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer, 'correct_answer': str(correct_answer)}

def _get_GSM8K_instructions(example, tokenizer):
    instructions = [
        {"role": "system", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request."},
        {"role": "user", "content": example['query']},
    ]
    start_index = example['response'].find("#### ")
    
    assert(start_index != -1)
    
    correct_answer = example['response'][start_index + len("#### "):]
    
    instructions_ans = [
        {"role": "assistant", "content": example['response'][:start_index] + f"The answer is: {correct_answer}"}
    ]
    
    instructions_wa = [
        {"role": "assistant", "content": f""}
    ]

    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )

    text_wa_answer = tokenizer.apply_chat_template(
        instructions + instructions_wa,
        tokenize=False
    )
    text_wa_answer = text_wa_answer.rsplit('<|eot_id|>', 1)[0]
    
    return {'text': text, 'text_wa_answer': text_wa_answer, 'correct_answer': str(correct_answer)}

def _load_datasets(config) -> list[DatasetDict]:
    math = load_dataset("meta-math/MetaMathQA")
    gsm8k = load_dataset("openai/gsm8k", "main")

    return [math, gsm8k]


def _process_datasets(config, dataset_ls: list[DatasetDict], tokenizer) -> list[DatasetDict]:
    dataset_processors = (
        _get_MATH_instructions,
        _get_GSM8K_instructions
    )
    
    dataset_names = (
        'MATH', 'GSM8K'
    )

    new_dataset_ls = []
    for dataset, dataset_name, processor in zip(dataset_ls, dataset_names, dataset_processors):   
        processor = functools.partial(
            processor,
            tokenizer=tokenizer,
        )

        dataset = dataset.map(
            processor, 
            batched=False, 
            num_proc=config.num_proc,
        )

        for split_name in dataset:
            dataset[split_name] = dataset[split_name].add_column(
                name='task',
                column=[dataset_name] * len(dataset[split_name])
            )
        
        new_dataset_ls.append(dataset)

    return new_dataset_ls


def _generate_dataset(config, tokenizer) -> DatasetDict:
    dataset_ls = _load_datasets(config=config)
    dataset_ls = _process_datasets(config=config, dataset_ls=dataset_ls, tokenizer=tokenizer)

    for i, dataset in enumerate(dataset_ls):
        if 'test' not in dataset:
            dataset_split = dataset['train'].train_test_split(test_size=0.2)
            dataset_ls[i] = dataset_split


    train_dataset = datasets.concatenate_datasets([
        dataset['train'].select_columns(['task', 'text', 'text_wa_answer', 'correct_answer'])
        for dataset in dataset_ls
    ])

    validation_dataset = datasets.concatenate_datasets([
        dataset['test'].select_columns(['task', 'text', 'text_wa_answer', 'correct_answer'])
        for dataset in dataset_ls
    ])

    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })


def _load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        **OmegaConf.to_object(config.tokenizer_config)
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _load_and_process_task(config, tokenizer, task_name: str):
    dataset_loader = {
        'MATH':     lambda: load_dataset("meta-math/MetaMathQA"),
        'GSM8K':    lambda: load_dataset("openai/gsm8k", "main")
    }
    
    dataset_processors = {
        'MATH': _get_MATH_instructions,
        'GSM8K': _get_GSM8K_instructions
    }

    dataset = dataset_loader[task_name]()
    processor = dataset_processors[task_name]

    processor = functools.partial(
        processor,
        tokenizer=tokenizer,
    )

    dataset = dataset.map(
        processor, 
        batched=False, 
        num_proc=config.num_proc,
    )
    
    if 'test' not in dataset:
        dataset = dataset['train'].train_test_split(test_size=0.2)

    train_dataset =      dataset['train'].select_columns(['text', 'text_wa_answer', 'correct_answer'])
    validation_dataset = dataset['test'].select_columns(['text', 'text_wa_answer', 'correct_answer'])
    
    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })


def generate_task(config_pth, out_dir, task_name: str):
    config = OmegaConf.load(config_pth)

    tokenizer = _load_tokenizer(config)
    dataset: DatasetDict = _load_and_process_task(config, tokenizer, task_name)

    dataset.save_to_disk(
        dataset_dict_path=out_dir,
        max_shard_size=config.get('max_shard_size', None),
        num_proc=config.get('num_proc', None),
    )


def generate(config_pth, out_dir):
    config = OmegaConf.load(config_pth)

    tokenizer = _load_tokenizer(config)
    dataset: DatasetDict = _generate_dataset(config, tokenizer=tokenizer)

    dataset.save_to_disk(
        dataset_dict_path=out_dir,
        max_shard_size=config.get('max_shard_size', None),
        num_proc=config.get('num_proc', None),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Loading from the HF hub and processing Math dataset'
    )

    parser.add_argument('config', help='Path to config file')
    parser.add_argument('output', help='Path to output dir')
    parser.add_argument(
        '-s', '--select', 
        choices=['MATH', 'GSM8K'],
        help='Load only one task. Tasks: BoolQ, PIQA, SIQA, hellaswag, winogrande, ARC-E, ARC-C, OBQA'
    )

    args = parser.parse_args()
    cfg_path = args.config
    out_dir = args.output
    task_name = args.select

    if task_name is None:
        print(f"Loading MATH and GSM8K")
        generate(config_pth=cfg_path, out_dir=out_dir)
    else:
        print(f"Loading {task_name}")
        generate_task(config_pth=cfg_path, out_dir=out_dir, task_name=task_name)


if __name__ == '__main__':
    main()
