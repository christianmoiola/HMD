from src.utils.utils import *
from src.utils.utils_model import generate
from src.utils.logging import setup_logger
import os

class DM():
    def __init__(self, cfg: dict, model, tokenizer, history=None):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = read_txt(os.path.join(self.path, cfg["DM"].get("prompt")))

        self.model = model
        self.tokenizer = tokenizer
        self.history = history
        self.logger = setup_logger(self.__class__.__name__)
        
    def query_model(self, input: str):
        self.logger.info("Generating response from DM component...")

        if self.history != None:
            sp = self.system_prompt + "\n" + self.history.get_history()
        else:
            sp = self.system_prompt

        self.logger.debug(f"History: {self.history.get_history()}")
        input_text = self.template.format(sp, input)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        #* STRIP RESPONSE
        response = response.strip()
        return response