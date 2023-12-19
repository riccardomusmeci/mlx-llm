from .llm import LLM

def list_models():
    from .config import CONFIG
    return list(CONFIG.keys())