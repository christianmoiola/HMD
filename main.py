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
            "get_car_info": "GettingInfoStateTracker",
            "order_car": "OrderCarStateTracker",
            "give_feedback": "GiveFeedbackStateTracker",
            "book_appointment": "BookAppointmentStateTracker",
            "out_of_domain": "OutOfDomainStateTracker",
            "negotiate_price": "NegotiatePriceStateTracker",
        }
        
    def define_components(self):
        self.history = History()
        self.database = Database(self.config)
        self.pre_nlu = PRE_NLU(cfg=self.config, model=self.model, tokenizer=self.tokenizer, history=self.history, logging_level="ERROR")
        self.nlu = NLU(cfg=self.config, model=self.model, tokenizer=self.tokenizer, history=self.history, logging_level="ERROR")
        self.dm = DM(cfg=self.config, model=self.model, tokenizer=self.tokenizer, logging_level="ERROR")
        self.nlg = NLG(cfg=self.config, model=self.model, tokenizer=self.tokenizer, logging_level="ERROR")

    def update_state_tracker(self, nlu_response):
        # Check the intent and create or update the corresponding state tracker
        intent = nlu_response["intent"]
        if self.intent_to_class[intent] not in [st.__class__.__name__ for st in self.list_state]:
            self.logger.info(f"Creating new state tracker for intent: {intent}")
            # Instantiate the state tracker if it doesn't already exist
            match intent:
                case "buying_car":
                    state_tracker_class = BuyingStateTracker()
                case "get_car_info":
                    state_tracker_class = GettingInfoStateTracker()
                case "negotiate_price":
                    state_tracker_class = NegotiatePriceStateTracker()
                case "order_car":
                    state_tracker_class = OrderCarStateTracker()
                case "give_feedback":
                    state_tracker_class = GiveFeedbackStateTracker()
                case "book_appointment":
                    state_tracker_class = BookAppointmentStateTracker()
                case "out_of_domain":
                    state_tracker_class = OutOfDomainStateTracker()
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
            if user_input == "exit":
                self.logger.info("Exiting the conversation...")
                break
            pre_nlu_response = None
            while pre_nlu_response == None:
                pre_nlu_response = self.pre_nlu.query_model(user_input)

            self.logger.debug(f"PRE_NLU Response: {pre_nlu_response}")

            nlg_responses = []
            # Iterate over the pre_nlu_response list, which contains the user input and the intent
            for elem in pre_nlu_response:
                nlu_response = None
                while nlu_response == None: 
                    nlu_response = self.nlu.query_model(elem)

                self.logger.debug(f"NLU Response: {nlu_response}")
                # Update the state tracker
                json = self.update_state_tracker(nlu_response)

                self.logger.debug(f"Dialogue State: {json}")

                dm_response = None
                while dm_response == None:
                    dm_response = self.dm.query_model(json)

                self.logger.debug(f"DM Response: {dm_response}")

                # Remove from the state tracker the state if the action is confirmation
                if dm_response["action"] == "confirmation":
                    target_class = self.intent_to_class[dm_response["parameter"]]
                    found = False
                    for st in self.list_state:
                        if st.__class__.__name__ == target_class:
                            self.list_state.remove(st)
                            self.logger.info(f"State tracker {st.__class__.__name__} removed from the list")
                            found = True
                            break
                    if not found:
                        self.logger.debug(f"No state tracker matching {target_class} found in the list")


                data = None
                if dm_response["action"] == "confirmation" and dm_response["parameter"] == "get_car_info":
                    results = self.database.get_car_info(json)
                    self.logger.debug(f"Get car info result: {results}")
                    if results == "None":
                        dm_response["action"] = "no_results_found"
                    else:
                        data = f"{results}"
                if dm_response["action"] == "confirmation" and dm_response["parameter"] == "negotiate_price":
                    results = self.database.find_car_by_id(json["slots"]["car_id"])
                    self.logger.debug(f"Negotiate price result: {results}")
                    if results == "None":
                        dm_response["action"] = "no_results_found"
                    else:
                        data = f"\nCar: {results['brand']} {results['model']}\nUser price: {json['slots']['proposed_price']}\nSystem price: {results['budget']-results['negotiable'][1] if results['negotiable'][0]=='Yes' else results['budget']}\n"
                if dm_response["action"] == "confirmation" and dm_response["parameter"] == "buying_car":
                    results = "[]"
                    constraints_relaxed = []

                    while results == "[]" and len(constraints_relaxed) <= 2:
                        print(f"Current dialogue state: {json}")
                        results = self.database.query_database(json)
                        self.logger.debug(f"Database Results: {results}")
                        if results == "[]":
                            slots_importance = ["transmission", "year", "fuel_type", "car_type", "model", "brand", "budget"]
                            for slot in slots_importance:
                                if json["slots"][slot] != None:
                                    json["slots"][slot] = None
                                    constraints_relaxed.append(slot)
                                    self.logger.info("Constraint relaxed: " + slot)
                                    break
                    data = f"Database results: {str(results)}" if len(constraints_relaxed) == 0 else f"Database results: {str(results)}\nConstraints relaxed: {', '.join(constraints_relaxed)}"
                    if results == "[]":
                        dm_response["action"] = "no_results_found"

                if dm_response["parameter"] == "booking_appointment":
                    data += f"Current date: 01/06/2025, Time: 10:00 AM" 
                if dm_response["action"] == "confirmation" and dm_response["parameter"] == "order_car":
                    results = self.database.find_car_by_id(json["slots"]["car_id"])
                    self.logger.debug(f"Order car result: {results}")
                    if results == "None":
                        dm_response["action"] = "no_results_found"
                    else:
                        data = f"Car ordered: {results}"
                nlg_response = self.nlg.query_model(input=dm_response, data=data, nlu_response=json)
                nlg_responses.append(nlg_response)
                self.logger.debug(f"NLG Response: {nlg_response}")

            # Update the history with the user input
            self.history.add_to_history(sender="User", msg=user_input)
            if len(nlg_responses) > 1:
                nlg_response = self.nlg.query_model(input=nlg_responses)
            else:
                nlg_response = nlg_responses[0]
            self.logger.info(f"Carllama: {nlg_response}")

            # Update the history with the system response
            self.history.add_to_history(sender="System", msg=nlg_response)
        

if __name__ == "__main__":

    set_token()

    config = configparser.ConfigParser()
    config.read('config.ini')
    config["Settings"] = {
        "path": os.getcwd()
    }

    pipeline = Pipeline(config=config)
    pipeline.run()

    #evaluation = Evaluation(cfg=config)
    #evaluation.test_nlu(is_history=False)
    
    #TODO (Additional) Modify the book apointment in order to (book the apointment; you will receive a confirmaton email with the details of the appointment if the appointment is available)
    #TODO (Additional) add contact operator
    #TODO add the state of the selected car
    #TODO add the fallback that no results are found in the database if 1 relaxed constraints already happened
    #TODO terminate sistem intent?
    #TODO if the user asks for a sports car don't know the brand ecc so maybe ask other things
    #* improve the nlg prompt and create the nlg no results prompt
    #TODO renting car
    #* put json in the nlg?


    #TODO aggiungere terminate system 
    #* aggiungere multi intent detection
    #* aggiungere JSON in the nlg
    #* when the sytem provide some car that not match the user request, the system present the car relaxed but ask if he want to do another search
    #* handling better the nlu input in the NLG component in order to provide a confirmation to the user of what the system understood
    #* add the fact to book an appointment if no information of a given car is found in the database    
    #TODO add a dictionary that contain the list of the car shown to the user and also the current car selected by the user
    #TODO fix and add some examples to the pre nlu prompt
    #TODO add the Name of the car if the reuqest info contain the car ID
    #TODO nlg-negotiate-price: the system have to accept if the user price is higher that the system price
    