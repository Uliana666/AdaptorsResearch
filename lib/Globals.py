import numpy as np
from transformers import DataCollatorForLanguageModeling

def deep_getattr(struct, field_path):
    fields = field_path.split('.')
    cur = struct
    for field in fields:
        cur = getattr(cur, field)
    return cur


def compress_matrix(matrix, r):
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :r], np.diag(S[:r]), VT[:r, :]


def compress_matrix_full(matrix, r):
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :r] @ np.diag(S[:r]) @ VT[:r, :]


def _tokenize_(example, name_text, tokenizer, max_length):
    
    t =  tokenizer(
        example[name_text],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    return t
    
class DataCollatorForChat(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, start_token=-1):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.start_token = start_token

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        
        labels = batch["labels"]
        for i in range(labels.size(0)):
            special_token_indices = (labels[i] == self.start_token).nonzero(as_tuple=True)[0]
            if len(special_token_indices) > 0:
                start = special_token_indices[-1].item()
                labels[i, :start + 1] = -100
        
        return batch