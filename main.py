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
from src.evaluation.Evaluation import Evaluation
import json


def set_token():
    token = configparser.ConfigParser()
    token.read('token.ini')
    os.environ["HF_TOKEN"]= token["TOKEN"].get("token")


class Pipeline():
    def __init__(self, config):
        self.config = config
        self.initial_message = self.config["General"].get("initial_message")
        self.model, self.tokenizer = get_model(config)
        self.define_components()
        self.logger = setup_logger(self.__class__.__name__, logging_level="DEBUG", color_debug="DEBUG_MAIN")

        self.list_state = []

        self.intent_to_class = {
            "buying_car": "BuyingStateTracker",
            "selling_car": "SellingStateTracker",
            "renting_car": "RentingStateTracker",
            "get_car_info": "GetCarInfoStateTracker"
        }
    
    def define_components(self):
        self.history = History()
        self.database = Database(self.config)
        self.pre_nlu = PRE_NLU(cfg=self.config, model=self.model, tokenizer=self.tokenizer, history=self.history)
        self.nlu = NLU(cfg=self.config, model=self.model, tokenizer=self.tokenizer, history=self.history)
        self.dm = DM(cfg=self.config, model=self.model, tokenizer=self.tokenizer)
        self.nlg = NLG(cfg=self.config, model=self.model, tokenizer=self.tokenizer)

    def update_state_tracker(self, nlu_response):
        # Check the intent and create or update the corresponding state tracker
        intent = nlu_response["intent"]
        if self.intent_to_class[intent] not in [st.__class__.__name__ for st in self.list_state]:
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
                    self.logger.error(f"Intent {intent} not recognized")
                    exit(1)

            state_tracker_class.update_dialogue_state(nlu_response)
            json = state_tracker_class.get_dialogue_state()
            self.list_state.append(state_tracker_class)
        else:
            # Update the existing state tracker
            for st in self.list_state:
                if st.__class__.__name__ == self.intent_to_class[intent]:
                    st.update_dialogue_state(nlu_response)
                    json = st.get_dialogue_state()
                    break
        return json
            

    def run(self):
        self.logger.info(f"System: {self.initial_message}")
        self.history.add_to_history(sender="System", msg=self.initial_message)
        
        user_input = ""

        while user_input != "exit":
            user_input = input("User: ")
            # TODO: Handle the case where the user input is difficult to understand
            # TODO: (e.g. the user input contains a small sentence and the NLU component doesn't understand the intent)
            if user_input == "exit":
                self.logger.info("Exiting the conversation...")
                break
            pre_nlu_response = self.pre_nlu.query_model(user_input)

            self.logger.debug(f"PRE_NLU Response: {pre_nlu_response}")

            nlu_response = None
            # TODO: Handle the case where the NLU response is None
            while nlu_response == None: 
                nlu_response = self.nlu.query_model(pre_nlu_response)

            self.logger.debug(f"NLU Response: {nlu_response}")

            # Update the history with the user input
            self.history.add_to_history(sender="User", msg=user_input)
            # Update the state tracker
            json = self.update_state_tracker(nlu_response)

            self.logger.debug(f"Dialogue State: {json}")

            dm_response = self.dm.query_model(json)

            self.logger.debug(f"DM Response: {dm_response}")

            if dm_response.startswith("find"):
                results = None

                while results == None:
                    results = self.database.query_database(dm_response) 
                    self.logger.debug(f"Database results: {results}")
                
                dm_response = self.dm.query_model(input=json, db_results=results)
                self.logger.debug(f"DM Response: {dm_response}")

            nlg_response = self.nlg.query_model(input=dm_response, nlu_response=json)

            self.logger.debug(f"NLG Response: {nlg_response}")

            self.logger.info(f"System: {nlg_response}")

            # Update the history with the system response
            self.history.add_to_history(sender="System", msg=nlg_response)
        

if __name__ == "__main__":

    set_token()

    config = configparser.ConfigParser()
    config.read('config.ini')
    config["Settings"] = {
        "path": os.getcwd()
    }
    test = Evaluation(config)
    #test.test(name_component="NLG", action="confirmation") #action = inform, request_info, relax_constraints, confirmation
    test.test(name_component="DM", action="bho")
    # pipeline = Pipeline(config)
    # pipeline.run()
    # dm_response = """inform([{'CarID': 16, 'brand': 'BMW','model': '3 Series', 'year': 2017, 'budget': 16917.22, 'Seats': 2, 'Availability': 'Available for rent', 'Rental Price per Day': 35.66, 'Insurance': 'Yes', 'Condition': 'Used', 'Location': 'Rome', 'Negotiable': ['No', 'N/A'], 'car_type': 'Sport_car', 'fuel_type': 'Electric', 'transmission': 'Automatic'}, {'CarID': 25, 'brand': 'BMW','model': 'X1', 'year': 2024, 'budget': 18105.17, 'Seats': 7, 'Availability': 'Available for rent', 'Rental Price per Day': 31.8, 'Insurance': 'Yes', 'Condition': 'New', 'Location': 'Naples', 'Negotiable': ['Yes', 713], 'car_type': 'Sport_car', 'fuel_type': 'Electric', 'transmission': 'Manual'}])"""
    # json = """{"intent": "buying_car", "slots": {"brand": "BMW", "model": None, "year": None, "budget": None, "car_type": "Sport_car", "fuel_type": None, "transmission": None}}"""
    # model, tokenizer = get_model(config)
    # nlg = NLG(cfg=config, model=model, tokenizer=tokenizer)
    # while True:
    #     nlg_response = nlg.query_model(input=dm_response, nlu_response=json)
    #     print(nlg_response)
    '''
    database = Database(config)
    
    query = """find(brand=BMW, model=5 Series, year=2020)"""
    results = database.query_database(query)
    print(results)
    '''