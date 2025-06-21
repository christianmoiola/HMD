from src.utils.utils import *
from src.utils.utils_model import generate
from src.utils.logging import setup_logger
import json
import os

class DM():
    def __init__(self, cfg: dict, model, tokenizer, history=None, logging_level="DEBUG"):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = read_txt(os.path.join(self.path, cfg["DM"].get("prompt")))

        self.model = model
        self.tokenizer = tokenizer
        self.history = history
        self.logger = setup_logger(self.__class__.__name__, logging_level=logging_level)
        
    def query_model(self, input: str, db_results=None):
        self.logger.info("Generating response from DM component...")
        input = str(input)
        self.logger.debug(f"Input: {input}")
        if self.history != None:
            sp = self.system_prompt + "\n" + self.history.get_history()
            self.logger.debug(f"History: {self.history.get_history()}")
        else:
            sp = self.system_prompt
        
        if db_results != None:
            input = "DATABASE RESULTS:\n" + db_results + "\n" + input
            
        input_text = self.template.format(sp, input)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        
        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing response as JSON: {e}")
            response = None
        return response