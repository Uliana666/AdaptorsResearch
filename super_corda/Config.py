from dataclasses import dataclass, field
from typing import List

from super_corda.Injector import inject_scorda

@dataclass
class SCorDAConfig:
    r: int
    alpha: int
    target_modules: List[str]
    dropout: int = 0
    init_strategy: str = None
    samples: str = None
    
def get_peft_model(model, SCorDAConfig):
    return inject_scorda(SCorDAConfig, model)