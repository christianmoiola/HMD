from src.utils.utils import *
from src.utils.logging import setup_logger
import os

class NLG():
    def __init__(self, cfg: dict, model, tokenizer, history=None):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = read_txt(os.path.join(self.path, cfg["NLG"].get("prompt")))

        self.model = model
        self.tokenizer = tokenizer
        self.history = history
        self.logger = setup_logger(self.__class__.__name__)

    def combine_system_prompt(self, nlu_response: dict, dm_response: str):
        combined_response = {
            "nlu": nlu_response,
            "dm": dm_response
        }
        return combined_response

    def query_model(self, input: str, nlu_response: dict):
        self.logger.info("Generating response from NLG component...")
        combined_response = self.combine_system_prompt(nlu_response, input)

        if self.history != None:
            combined_response = str(combined_response) + "\n" + self.history.get_history()

        self.logger.debug(f"History: {self.history.get_history()}")

        input_text = self.template.format(self.system_prompt, combined_response)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        #* STRIP RESPONSE
        response = response.strip()
        return response
    