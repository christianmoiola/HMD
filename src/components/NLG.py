from src.utils.utils import *
from src.utils.logging import setup_logger
from src.utils.utils_model import generate
import os

class NLG():
    def __init__(self, cfg: dict, model, tokenizer, history=None, logging_level="DEBUG"):
        # Path project
        self.path = cfg["Settings"].get("path")

        self.template = cfg["TEMPLATES"].get(cfg["General"].get("model_name"))
        self.max_seq_length = cfg["General"].getint("max_seq_length")
        self.system_prompt = {}
        self.system_prompt["negotiate_price"] = read_txt(os.path.join(self.path, cfg["NLG"].get("prompt_negotiate_price")))
        self.system_prompt["order_car"] = read_txt(os.path.join(self.path, cfg["NLG"].get("prompt_order_car")))
        self.system_prompt["get_car_info"] = read_txt(os.path.join(self.path, cfg["NLG"].get("prompt_get_car_info")))
        self.system_prompt["buying_car"] = read_txt(os.path.join(self.path, cfg["NLG"].get("prompt_buying_car")))
        self.system_prompt["give_feedback"] = read_txt(os.path.join(self.path, cfg["NLG"].get("prompt_give_feedback")))
        self.system_prompt["book_appointment"] = read_txt(os.path.join(self.path, cfg["NLG"].get("prompt_book_appointment")))
        self.system_prompt["request_info"] = read_txt(os.path.join(self.path, cfg["NLG"].get("prompt_request_info")))

        self.model = model
        self.tokenizer = tokenizer
        self.history = history
        self.logger = setup_logger(self.__class__.__name__, logging_level=logging_level)

    def combine_system_prompt(self, dm_response: str, data: str):
        combined_response = {
            "DM Response": dm_response,
            "Data": data
        }
        return combined_response

    def query_model(self, input: str, data: str = None):
        self.logger.info("Generating response from NLG component...")

        if data != None:
            combined_response = self.combine_system_prompt(input, data)
        else:
            combined_response = {"DM Response": input}

        if self.history != None:
            combined_response = str(combined_response) + "\n" + self.history.get_history()
            self.logger.debug(f"History: {self.history.get_history()}")
        
        input_text = self.template.format(self.system_prompt[input["parameter"] if input["action"] == "confirmation" else input["action"]], combined_response)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response = generate(self.model, inputs, self.tokenizer, self.max_seq_length)
        #* STRIP RESPONSE
        response = response.strip()
        return response
    