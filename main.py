import os
import configparser
from src.components.NLU import NLU, PRE_NLU
from src.components.DM import DM
from src.components.NLG import NLG
from src.utils.utils import *

def set_token():
    token = configparser.ConfigParser()
    token.read('token.ini')
    os.environ["HF_TOKEN"]= token["TOKEN"].get("token")

def get_model(config):
    model_name = config["General"].get("model_name") # llama2 or llama3
    model_name = config["MODELS"].get(model_name) # meta-llama/Llama-2-7b-chat-hf or meta-llama/Meta-Llama-3-8B-Instruct
    dtype = config["General"].get("dtype") 
    folder_model = os.path.join(config["Settings"].get("path"), config["General"].get("folder_model"))

    model, tokenizer = load_model(
        model_name=model_name, 
        folder_model=folder_model, 
        dtype=dtype
        )

    return model, tokenizer

if __name__ == "__main__":

    set_token()

    config = configparser.ConfigParser()
    config.read('config.ini')
    config["Settings"] = {
        "path": os.getcwd()
    }

    model, tokenizer = get_model(config)
    pre_nlu = PRE_NLU(config, model, tokenizer)
    nlu = NLU(config, model, tokenizer)
    dm = DM(config, model, tokenizer)
    nlg = NLG(config, model, tokenizer)

    #user_input = "I would like to order ragù lasagna, please."

    # pre_nlu_response = pre_nlu.query_model(user_input)
    # print(pre_nlu_response)
    #user_input = "I would like to buy a Diesel car for family."
    user_input = "I would like to buy a sport car. Can you have some available?"
    while user_input != "exit":
        user_input = input("User: ")
        nlu_response = nlu.query_model(user_input)
        print(nlu_response)

    # dm_response = dm.query_model(nlu_response)
    # print(dm_response)
    # nlg_response = nlg.query_model(dm_response)
    # print(nlg_response)