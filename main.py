import os
import configparser
from src.components.NLU import NLU, PRE_NLU
from src.components.DM import DM
from src.components.NLG import NLG
from src.components.StateTracker import *
from src.utils.utils import *
import json


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

    list_state = []

    intent_to_class = {
        "buying_car": "BuyingStateTracker",
        "selling_car": "SellingStateTracker",
        "renting_car": "RentingStateTracker",
        "get_car_info": "GetCarInfoStateTracker"
    }

    #user_input = "I would like to order ragù lasagna, please."
    user_input = "I would like to buy a petrol car."

    while user_input != "exit":
        user_input = input("User: ")
        # TODO: Handle the case where the user input is difficult to understand
        # TODO: (e.g. the user input contains a small sentence and the NLU component doesn't understand the intent)
        if user_input == "exit":
            break
        pre_nlu_response = pre_nlu.query_model(user_input)
        
        print(f"PRE_NLU Response: {pre_nlu_response}")
        nlu_response = None
        # TODO: Handle the case where the NLU response is None
        while nlu_response == None: 
            nlu_response = nlu.query_model(pre_nlu_response)

        print(f"NLU Response: {nlu_response}")

        # Check the intent and create or update the corresponding state tracker
        intent = nlu_response["intent"]
        if intent_to_class[intent] not in [st.__class__.__name__ for st in list_state]:
            # Instantiate the state tracker if it doesn't already exist
            match intent:
                case "buying_car":
                    state_tracker_class = BuyingStateTracker()
                case "selling_car":
                    state_tracker_class = SellingStateTracker()
                case "renting_car":
                    state_tracker_class = RentingStateTracker()
                case "get_car_info":
                    state_tracker_class = GettingInfoStateTracker()

            state_tracker_class.update_dialogue_state(nlu_response)
            json = state_tracker_class.get_dialogue_state()
            list_state.append(state_tracker_class)
        else:
            # Update the existing state tracker
            for st in list_state:
                if st.__class__.__name__ == intent_to_class[intent]:
                    st.update_dialogue_state(nlu_response)
                    json = st.get_dialogue_state()
                    break
        
        print(f"Dialogue State JSON: {json}")

        dm_response = dm.query_model(json)

        print(f"DM Response: {dm_response}")

        nlg_response = nlg.query_model(input=dm_response, nlu_response=json)

        print(f"NLG Response: {nlg_response}")



        

        '''
        json_nlu_response = json.loads(nlu_response)

        if json_nlu_response["intent"] not in dict_intent:
            dict_intent[json_nlu_response["intent"]] = json_nlu_response["slots"]
        else:
            for key, value in json_nlu_response["slots"].items():
                print(key, value)
                if json_nlu_response["slots"][key] != None:
                    print("entra")
                    dict_intent[json_nlu_response["intent"]][key] = value

        print(dict_intent)
        '''