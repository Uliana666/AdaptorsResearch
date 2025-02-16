from datasets import load_dataset
from torch.utils.data import DataLoader

PROMPT_MATH = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )

def LoadMath():
    return {"name_dataset": "meta-math/MetaMathQA",
            "type": "train",
            "query": "query",
            "response": "response",
            "prompt": PROMPT_MATH,
            }
    
