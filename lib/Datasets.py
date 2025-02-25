from datasets import load_dataset
from torch.utils.data import DataLoader
    

def LoadCommonReasoning(type, count, name="common-reasoning", seed=42):
    print("Load", name)
    dataset = load_dataset(f"./datasets/{name}", split=type)
    
    dataset = dataset.shuffle(seed=seed)
    if count != None:
        dataset = dataset.select(range(count))
        
    return {"dataset": dataset, "name_text": "text"}