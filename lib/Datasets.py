from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from lib import Globals
import datasets

def LoadCommonReasoning(type, count, name="common-reasoning", seed=42):
    print("Load", name)
    dataset = load_dataset(f"./datasets/{name}", split=type)
    print(dataset)
    
    dataset = dataset.shuffle(seed=seed)
    if count != None:
        dataset = dataset.select(range(count))
        
    return {"dataset": dataset, "name_text": "text"}


def PrepareDataset(dataset, name_text, args, tokenizer, desc=""):
    dataset = dataset.map(
        Globals._tokenize_,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Running tokenizer on dataset ({desc})",
        fn_kwargs={"tokenizer": tokenizer, 
                "max_length": args.model_max_length, 
                "name_text": name_text},
        load_from_cache_file=False,
    )
    
    
    data_collator = Globals.DataCollatorForChat(
        tokenizer=tokenizer,
        mlm=False,
        start_token=args.start_token
    )
    
    return dataset, data_collator