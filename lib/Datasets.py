from datasets import load_dataset
from torch.utils.data import DataLoader

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    )

def LoadMath():
    return {"name_dataset": "meta-math/MetaMathQA",
            "type": "train",
            "query": "query",
            "response": "response",
            "prompt": PROMPT,
            }
    
def LoadHumanEval():
    return {"name_dataset": "openai/openai_humaneval",
            "type": "test",
            "query": "prompt",
            "response": "canonical_solution",
            "prompt": PROMPT,
            }
    
def LoadCodeFeedback():
    return {"name_dataset": "m-a-p/CodeFeedback-Filtered-Instruction",
            "type": "train",
            "query": "query",
            "response": "answer",
            "prompt": PROMPT,
            }
    
def LoadWizardLMEvolInstruct():
    return {"name_dataset": "fxmeng/WizardLM_evol_instruct_V2_143k",
            "type": "train",
            "query": "human",
            "response": "assistant",
            "prompt": PROMPT,
            }
    
def LoadGPQA():
    return {"name_dataset": "Idavidrein/gpqa",
            "type": "train",
            "query": "Question",
            "response": "Explanation",
            "prompt": PROMPT,
            }
    

def LoadCommonReasoning(type, count, seed=42):
    dataset = load_dataset('./datasets/common-reasoning', split=type)
    
    dataset = dataset.shuffle(seed=seed)
    
    dataset = dataset.select(range(count))
        
    return {"dataset": dataset, "name_text": "text"}