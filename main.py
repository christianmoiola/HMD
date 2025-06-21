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
            "renting_car": "RentingStateTracker",
            "get_car_info": "GetCarInfoStateTracker",
            "order_car": "OrderCarStateTracker",
            "give_feedback": "GiveFeedbackStateTracker",
            "book_appointment": "BookAppointmentStateTracker",
            "out_of_domain": "OutOfDomainStateTracker",
            "negotiate_price": "NegotiatePriceStateTracker",
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
                case "renting_car":
                    state_tracker_class = RentingStateTracker()
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
            # TODO: Handle the case where the user input contain more than one intent
            if user_input == "exit":
                self.logger.info("Exiting the conversation...")
                break
            pre_nlu_response = None
            while pre_nlu_response == None:
                pre_nlu_response = self.pre_nlu.query_model(user_input)

            self.logger.debug(f"PRE_NLU Response: {pre_nlu_response}")

            nlu_response = None
            while nlu_response == None: 
                nlu_response = self.nlu.query_model(pre_nlu_response[0])

            self.logger.debug(f"NLU Response: {nlu_response}")
            # Update the history with the user input
            self.history.add_to_history(sender="User", msg=user_input)
            # Update the state tracker
            json = self.update_state_tracker(nlu_response)

            self.logger.debug(f"Dialogue State: {json}")

            dm_response = None
            while dm_response == None:
                dm_response = self.dm.query_model(json)

            self.logger.debug(f"DM Response: {dm_response}")

            # Remove from the state tracker the state if the action is confirmation
            if dm_response["action"] == "confirmation":
                for st in self.list_state:
                    if st.__class__.__name__ == self.intent_to_class[dm_response["parameter"]]:
                        self.list_state.remove(st)
                        break

            data = None
            if dm_response["action"] == "confirmation" and dm_response["parameter"] in ["get_car_info", "negotiate_price", "buying_car"]:
                results = None
                constraints_relaxed = []
                while results == None:
                    results = self.database.query_database(json)
                    self.logger.debug(f"Database Results: {results}")
                    if results == None:
                        slots_importance = ["transmission", "fuel_type", "year", "model", "brand", "budget", "car_type"]
                        for slot in slots_importance:
                            if json["slots"][slot] != None:
                                json["slots"][slot] = None
                                constraints_relaxed.append(slot)
                                self.logger.info("Constraint relaxed: " + slot)
                                break

                data = f"Database results: {str(results)}" if len(constraints_relaxed) == 0 else f"Database results: {str(results)}\nConstraints relaxed: {', '.join(constraints_relaxed)}"
                if dm_response["parameter"] == "negotiate_price":
                    data = f"\nUser price: {json['slots']['proposed_price']}\n System price: {results['budget']-results['negotiable'][1] if results['negotiable'][0]=='Yes' else results['budget']}\n"

                nlg_response = self.nlg.query_model(input=dm_response, data=data)

            if dm_response["action"] == "request_info":
                data = f"Intent: {json['intent']}\n"
                nlg_response = self.nlg.query_model(input=dm_response, data=data)
            if dm_response["parameter"] == "booking_appointment":
                data += f"Current date: 01/06/2025, Time: 10:00 AM"
            nlg_response = self.nlg.query_model(input=dm_response, data=data)

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
    pipeline = Pipeline(config=config)
    pipeline.run()
    #TODO Modify the get car info in order to get the car with a given list of characteristics
    #TODO Modify the book apointment in order to (book the apointment; you will receive a confirmaton email with the details of the appointment if the appointment is available)
    #TODO add contact operator
    #TODO add the state of the selected car


    #PRE_NLU component test
    # pre_nlu = PRE_NLU(cfg=config, model=model, tokenizer=tokenizer)

    # input_pre_nlu = ["Hello, I want to buy a car.", "I'm looking for a sports car under 20000 euros.", "Can you tell me more about the BMW M3?", "I would like to negotiate the price of the Audi R8.", "I want to order the Honda Civic."]

    # for elem in input_pre_nlu:
    #     pre_nlu_response = pre_nlu.query_model(elem)
    #     print(f"\n\nPRE_NLU Response: {pre_nlu_response}")

    #NLU component test

    # history = History()
    # nlu = NLU(cfg=config, model=model, tokenizer=tokenizer, history=history)
    # history.add_to_history(sender="System", msg=config["General"].get("initial_message"))
    # history.add_to_history(sender="User", msg="Hello, I want to buy a car.")
    # history.add_to_history(sender="System", msg="Sure, I can help you with that. What kind of car are you looking for?")
    # history.add_to_history(sender="User", msg="I'm looking for a sports car under 20000 euros, preferably an Audi.")
    # history.add_to_history(sender="System", msg="Great choice! Here are some options for sports cars under 20000 euros that we have available: 14 BMW M3, 15 Audi R8, 16 Porsche 911. Would you like to know more about any of these cars?")
    # history.add_to_history(sender="User", msg="Can you tell me more about the BMW M3?")
    # history.add_to_history(sender="System", msg="Sure, the BMW M3 is a high-performance sports car with a powerful engine and sleek design. It offers a thrilling driving experience and is known for its agility and handling. Would you like to buy it?")

    # input_nlu_get_car_info = [{'intent': 'get_car_info', 'text': 'Can you tell me more about the Audi R8?'}, {'intent': 'get_car_info', 'text': 'Can you tell me more about the BMW M3?'}, {'intent': 'get_car_info', 'text': 'Can you tell me more about the Porsche 911?'}, {'intent': 'get_car_info', 'text': 'Can you tell me more about the car with ID 14?'}, {'intent': 'get_car_info', 'text': 'Can you tell me more about the Honda Civic?'}]
    # input_nlu_give_feedback = [{'intent': 'give_feedback', 'text': 'I would give 6 stars to the service.'}, {'intent': 'give_feedback', 'text': 'I would give 5 to this chatbot because it is very helpful.'}, {'intent': 'give_feedback', 'text': '5 for the fact that it was very helpful.'}, {'intent': 'give_feedback', 'text': '100% satisfaction with the service.'}]
    # input_nlu_negotiate_price = [{'intent': 'negotiate_price', 'text': 'I would like to negotiate the price of the Audi R8.'}, {'intent': 'negotiate_price', 'text': 'I like the BMW but the price is too high, can you lower it to 20000?'}, {'intent': 'negotiate_price', 'text': 'Can you do 20k for the 911?'}, {'intent': 'negotiate_price', 'text': 'I would like to negotiate the price of the Honda Civic.'}]
    # input_nlu_buying_car = [{'intent': 'buying_car', 'text': 'I want to buy the Audi R8.'}, {'intent': 'buying_car', 'text': 'I want to buy the BMW M3.'}, {'intent': 'buying_car', 'text': 'I want to buy a sports car under 15k.'}, {'intent': 'buying_car', 'text': 'I want to buy an electric car.'}]
    # input_nlu_order_car = [{'intent': 'order_car', 'text': 'I would like to order the Audi R8.'}, {'intent': 'order_car', 'text': 'I want to order that'}, {'intent': 'order_car', 'text': 'No, I want to order the BMW M3.'}, {'intent': 'order_car', 'text': 'I want to order the Honda Civic.'}]
    # input_book_appoint = [{'intent': 'book_appointment', 'text': 'I would like to book an appointment for a test drive.'}, {'intent': 'book_appointment', 'text': 'Can I book an appointment for today?'}]
   
    # for elem in input_book_appoint:
    #     print(f"\n\n\nUser Input: {elem['text']}")
    #     nlu_response = nlu.query_model(elem)
    #     print(f"NLU Response: {nlu_response}")


    '''
    dm = DM(cfg=config, model=model, tokenizer=tokenizer)
    db = Database(cfg=config)

    input_dm_find = [{'intent': 'buying_car', 'slots': {'brand': None, 'model': None, 'year': None, 'budget': '20000', 'car_type': 'Sport_car', 'fuel_type': None, 'transmission': None}}, {'intent': 'buying_car', 'slots': {'brand': 'Audi', 'model': None, 'year': None, 'budget': None, 'car_type': 'Sport_car', 'fuel_type': None, 'transmission': None}}, {'intent': 'get_car_info', 'slots': {'car_id': '5'}}, {'intent': 'negotiating_price', 'slots': {'car_id': '5', 'proposed_price': '13000'}}, {'intent': 'order_car', 'slots': {'car_id': '5'}}]
    input_dm = [{'intent': request_info, 'slots': {'car_id': '50'}} for request_info in ['get_car_info', 'negotiating_price', 'order_car']]
    for elem in input_dm:
        response = dm.query_model(elem)
        if response["action"] == "confirmation":
            results = db.query_database(elem)
            print(f"Database results for: {results}")
    '''
    