from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable


@dataclass
class HFConfig:
    repo_id: str
    revision: Optional[str] = None
    filename: Optional[str] = None

@dataclass
class QuantizeConfig:
    group_size: int
    bits: int
    
@dataclass
class ModelConfig:
    hf: HFConfig
    quantize: Optional[QuantizeConfig] = None
    converter: Optional[Callable] = None