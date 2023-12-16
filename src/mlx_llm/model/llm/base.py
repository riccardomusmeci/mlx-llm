

class BaseLLM:
    
    def __init__(self, model, tokenizer, personality, examples):
        self.model = model
        self.tokenizer = tokenizer
        self.personality = personality
        self.examples = examples
        
    @staticmethod
    def build():
        raise NotImplementedError()
    
    @staticmethod
    def convert():
        raise NotImplementedError()
    
    def generate(self):
        raise NotImplementedError()
    
    def chat(self):
        raise NotImplementedError()
    
    