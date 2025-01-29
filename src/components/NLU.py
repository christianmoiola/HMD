from src.utils.utils import *
from src.utils.utils_model import generate
from src.utils.logging import setup_logger
import json
import os


class PRE_NLU():
    def __init__(self, cfg: dict, model, tokenizer, history=None, logging_level="DEBUG"):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = read_txt(os.path.join(self.path, cfg["PRE_NLU"].get("prompt")))

        self.model = model
        self.tokenizer = tokenizer
        self.history = history
        self.logger = setup_logger(self.__class__.__name__, logging_level=logging_level)

    def query_model(self, user_input: str):
        self.logger.info("Generating response from PRE_NLU component...")
        '''
        input_text = self.template.format(self.system_prompt, user_input)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        '''
        response = user_input
        return response

class NLU():
    def __init__(self, cfg: dict, model, tokenizer, history=None, logging_level="DEBUG"):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = read_txt(os.path.join(self.path, cfg["NLU"].get("prompt")))

        self.model = model
        self.tokenizer = tokenizer
        self.history = history
        self.logger = setup_logger(self.__class__.__name__, logging_level=logging_level)
        
    def query_model(self, user_input: str):
        self.logger.info("Generating response from NLU component...")
        if self.history != None:
            sp = self.system_prompt + "\n" + self.history.get_history()
        else:
            sp = self.system_prompt
        self.logger.debug(f"History: {self.history.get_history()}")
        input_text = self.template.format(sp, user_input)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        
        # Try to parse the response as JSON
        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing response as JSON: {e}")
            response = None
        return response
