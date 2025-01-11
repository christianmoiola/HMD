from src.utils.utils import *
import os

class DM():
    def __init__(self, cfg: dict, model, tokenizer):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = read_txt(os.path.join(self.path, cfg["DM"].get("prompt")))

        self.model = model
        self.tokenizer = tokenizer
        
    def query_model(self, user_input: str):
        #print("Generating response from DM component...")
        input_text = self.template.format(self.system_prompt, user_input)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        return response