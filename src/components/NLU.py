from src.utils.utils import *
import json
import os


class PRE_NLU():
    def __init__(self, cfg: dict, model, tokenizer):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = read_txt(os.path.join(self.path, cfg["PRE_NLU"].get("prompt")))

        self.model = model
        self.tokenizer = tokenizer

    def query_model(self, user_input: str):
        #print("Generating response from PRE_NLU component...")
        '''
        input_text = self.template.format(self.system_prompt, user_input)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        '''
        response = user_input
        return response

class NLU():
    def __init__(self, cfg: dict, model, tokenizer):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = read_txt(os.path.join(self.path, cfg["NLU"].get("prompt")))

        self.model = model
        self.tokenizer = tokenizer
        
    def query_model(self, user_input: str):
        #print("Generating response from NLU component...")
        input_text = self.template.format(self.system_prompt, user_input)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        
        # Try to parse the response as JSON
        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            response = None
        return response
