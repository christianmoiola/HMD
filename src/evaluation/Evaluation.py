from src.utils.logging import setup_logger
from src.components.NLU import NLU, PRE_NLU
from src.components.DM import DM
from src.components.NLG import NLG
from itertools import product
from src.utils.utils_model import get_model
import os
import random 


class Evaluation():
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model, self.tokenizer = get_model(self.cfg)
        self.logger = setup_logger(self.__class__.__name__)

        self.slots = ["car_type", "budget", "brand", "model", "year", "fuel_type", "transmission"]
        self.intents = ["buying_car", "renting_car", "selling_car", "getting_info"]

        # Template for NLU
        self.nlu_response_template = {
            'intent': None,
            'slots': {slot: None for slot in self.slots}
        }
        # Template for DM
        self.dm_response_template = "{}({})"

    def test_dm(self):
        dm = DM(cfg=self.cfg, model=self.model, tokenizer=self.tokenizer, history=None, logging_level="ERROR")

        # Define test cases
        car_types = ["Sport_car", "Family_car", "City_car", "None"]
        brands = ["BMW", "Audi", "Mercedes", "None"]
        models = ["3 Series", "X1", "A1", "None"]

        for car_type, brand, model in product(car_types, brands, models):
            intent = random.choice(self.intents)

            nlu_response = self.nlu_response_template.copy()
            nlu_response["intent"] = intent
            nlu_response["slots"]["car_type"] = car_type
            nlu_response["slots"]["brand"] = brand
            nlu_response["slots"]["model"] = model
            
            array_slot = [("car_type", car_type), ("brand", brand), ("model", model)]
            count_none = sum(1 for _, value in array_slot if value != "None")
            result = ", ".join(f"{var_name}='{var_value}'" for var_name, var_value in array_slot if var_value is not "None")


            if count_none >= 2:
                true_dm_response = self.dm_response_template.format("find", result)
            else:
                true_dm_response = self.dm_response_template.format("request_info", next((value for name, value in array_slot if value is not None), None))

            # Query DM model
            self.logger.debug(f"\nDM Input:\n    NLU response:\n{nlu_response}\n")
            dm_response = dm.query_model(nlu_response)
            self.logger.info(f"\nDM Response:\n{dm_response}\n")
            if true_dm_response == dm_response:
                self.logger.info("Test passed!\n")
            else:
                self.logger.error("Test failed!\n")
                self.logger.error(f"Expected: {true_dm_response}\n")  

    def test_nlg(self, action_to_test):
        nlg = NLG(cfg=self.cfg, model=self.model, tokenizer=self.tokenizer, history=None, logging_level="ERROR")
        
        # Define test cases
        car_types = ["Sport_car", "Family_car", "City_car"]
        brands = ["BMW", "Audi", "Mercedes"]
        models = ["3 Series", "X1", "A1"]
        
        # Iterate through car attributes
        for car_type, brand, model in product(car_types, brands, models):
            intent = random.choice(self.intents)
            slot = random.choice(self.slots)
            
            nlu_response = self.nlu_response_template.copy()
            nlu_response["intent"] = intent
            nlu_response["slots"]["car_type"] = car_type
            nlu_response["slots"]["brand"] = brand
            nlu_response["slots"]["model"] = model
            

            if action_to_test == "inform":
                dm_response_results = [
                    {
                        "CarID": 16, "brand": brand, "model": model, "year": 2017,
                        "budget": 16917.22, "Seats": 2, "Availability": "Available for rent",
                        "Rental Price per Day": 35.66, "Insurance": "Yes", "Condition": "Used",
                        "Location": "Rome", "Negotiable": ["No", "N/A"], "car_type": car_type,
                        "fuel_type": "Electric", "transmission": "Automatic"
                    },
                    {
                        "CarID": 25, "brand": brand, "model": model, "year": 2024,
                        "budget": 18105.17, "Seats": 7, "Availability": "Available for rent",
                        "Rental Price per Day": 31.8, "Insurance": "Yes", "Condition": "New",
                        "Location": "Naples", "Negotiable": ["Yes", 713], "car_type": car_type,
                        "fuel_type": "Electric", "transmission": "Manual"
                    }
                ]
                dm_response = self.dm_response_template.format(action_to_test, dm_response_results)
            elif action_to_test == "request_info" or action_to_test == "relax_constraints":
                dm_response = self.dm_response_template.format(action_to_test, slot)
            elif action_to_test == "confirmation":
                dm_response = self.dm_response_template.format(action_to_test, intent)
            
            dm_response_str = str(dm_response)
            
            # Query NLG model
            self.logger.debug(f"\nNLG Input:\n    NLU response:\n{nlu_response}\n    DM response:\n{dm_response}\n")
            nlg_response = nlg.query_model(dm_response_str, nlu_response)
            self.logger.info(f"\nNLG Response:\n{nlg_response}\n")


    def test(self, name_component: str, action):

        self.logger.info(f"Testing {name_component} component...")
        
        match name_component:
            case "PRE_NLU":
                pass
            case "NLU":
                self.test_nlu()
            case "DM":
                self.test_dm()
            case "NLG":
                self.test_nlg(action)
            case _:
                self.logger.error(f"Component {name_component} not recognized")