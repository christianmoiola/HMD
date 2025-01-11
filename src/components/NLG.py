from src.utils.utils import *
import os

class NLG():
    def __init__(self, cfg: dict, model, tokenizer):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = read_txt(os.path.join(self.path, cfg["NLG"].get("prompt")))

        self.model = model
        self.tokenizer = tokenizer

    def combine_system_prompt(self, nlu_response: dict, dm_response: str):
        combined_response = {
            "nlu": nlu_response,
            "dm": dm_response
        }
        return combined_response

    def query_model(self, input: str, nlu_response: dict):
        #print("Generating response from NLG component...")
        combined_response = self.combine_system_prompt(nlu_response, input)
        input_text = self.template.format(self.system_prompt, combined_response)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        return response
    