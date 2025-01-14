import os
import configparser
from src.components.NLU import NLU, PRE_NLU
from src.components.DM import DM
from src.components.NLG import NLG
from src.components.Database import Database
from src.components.StateTracker import *
from src.utils.utils_model import get_model
from src.utils.history import History
from src.utils.logging import setup_logger
import json


def set_token():
    token = configparser.ConfigParser()
    token.read('token.ini')
    os.environ["HF_TOKEN"]= token["TOKEN"].get("token")

if __name__ == "__main__":

    set_token()

    config = configparser.ConfigParser()
    config.read('config.ini')
    config["Settings"] = {
        "path": os.getcwd()
    }

    database = Database(config)
    query = """find(Make="BMW")"""
    results = database.query_database(query)
    print(results)
    
    '''
    model, tokenizer = get_model(config)
    history = History()
    pre_nlu = PRE_NLU(cfg=config, model=model, tokenizer=tokenizer, history=history)
    nlu = NLU(cfg=config,model=model, tokenizer=tokenizer, history=history)
    dm = DM(cfg=config, model=model, tokenizer=tokenizer, history=history)
    nlg = NLG(cfg=config, model=model, tokenizer=tokenizer, history=history)
    logger = setup_logger("Main", logging_level="DEBUG", color_debug="DEBUG_MAIN")

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
            logger.info("Exiting the conversation...")
            break
        
        pre_nlu_response = pre_nlu.query_model(user_input)

        logger.debug(f"PRE_NLU Response: {pre_nlu_response}")

        nlu_response = None
        # TODO: Handle the case where the NLU response is None
        while nlu_response == None: 
            nlu_response = nlu.query_model(pre_nlu_response)

        logger.debug(f"NLU Response: {nlu_response}")

        # Update the history with the user input
        history.add_to_history(sender="User", msg=user_input)

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
                case _:
                    print(f"Intent {intent} not recognized")
                    continue

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

        logger.debug(f"Dialogue State: {json}")

        dm_response = dm.query_model(json)

        logger.debug(f"DM Response: {dm_response}")

        nlg_response = nlg.query_model(input=dm_response, nlu_response=json)

        logger.debug(f"NLG Response: {nlg_response}")

        logger.info(f"System: {nlg_response}")

        # Update the history with the system response
        history.add_to_history(sender="System", msg=nlg_response)
        '''