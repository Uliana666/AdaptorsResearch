from datasets import load_dataset
from torch.utils.data import DataLoader




def LoadWiki(tokenizer, batch_size, len_sent):
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    def parseWiki(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
 
        total_length = (total_length // len_sent) * len_sent
        
        result = {
            k: [t[i : i + len_sent] for i in range(0, total_length, len_sent)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        parseWiki,
        batched=True,
        batch_size=batch_size,
        num_proc=8,
    )
    return lm_datasets

def LoadMath():
    dataset = load_dataset("meta-math/MetaMathQA")
    
