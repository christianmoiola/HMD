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
        
        input_text = self.template.format(self.system_prompt, user_input)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        print(f"PRE_NLU response: {response}")
        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing response as JSON: {e}")
            response = None
        return response

class NLU():
    def __init__(self, cfg: dict, model, tokenizer, history=None, logging_level="DEBUG"):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = {}
        self.system_prompt["negotiate_price"] = read_txt(os.path.join(self.path, cfg["NLU"].get("prompt_negotiate_price")))
        self.system_prompt["order_car"] = read_txt(os.path.join(self.path, cfg["NLU"].get("prompt_order_car")))
        self.system_prompt["get_car_info"] = read_txt(os.path.join(self.path, cfg["NLU"].get("prompt_get_car_info")))
        self.system_prompt["buying_car"] = read_txt(os.path.join(self.path, cfg["NLU"].get("prompt_buying_car")))
        self.system_prompt["give_feedback"] = read_txt(os.path.join(self.path, cfg["NLU"].get("prompt_give_feedback")))
        self.system_prompt["book_appointment"] = read_txt(os.path.join(self.path, cfg["NLU"].get("prompt_book_appointment")))

        self.model = model
        self.tokenizer = tokenizer
        self.history = history
        self.logger = setup_logger(self.__class__.__name__, logging_level=logging_level)

    def query_model(self, user_input: dict):
        self.logger.info("Generating response from NLU component...")
        system_prompt = self.system_prompt[user_input["intent"]]

        if self.history != None:
            sp = system_prompt + "\n\n History:\n" + self.history.get_history()
            self.logger.debug(f"History: {self.history.get_history()}")
        else:
            sp = system_prompt
        input_text = self.template.format(sp, user_input["text"])
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        print(f"NLU response: {response}")
        # Try to parse the response as JSON
        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing response as JSON: {e}")
            response = None
        return response
