from abc import ABC, abstractmethod
from copy import deepcopy
from src.utils.logging import setup_logger

class DialogueStateTracker(ABC):
    def __init__(self):
        self.dialogue_state = {}

    def get_dialogue_state(self):
        """
        Returns the current dialogue state.
        """
        return self.dialogue_state

    def update_dialogue_state(self, nlu_response: dict):
        """
        Updates the dialogue state with the cleaned NLU response.
        """
        self.logger.debug(f"Dialogue state before update: {self.dialogue_state}")
        self.dialogue_state = DialogueStateTracker.update(self.dialogue_state, DialogueStateTracker.clean_response(nlu_response))
        self.logger.debug(f"Dialogue state after update: {self.dialogue_state}")
    
    @staticmethod
    def clean_response(response: dict) -> dict:
        final_dict = deepcopy(response)
        for key, value in response.items():
            if value == None:
                del final_dict[key]
            elif isinstance(value, dict):
                cleaned_dict = DialogueStateTracker.clean_response(deepcopy(value))
                if len(cleaned_dict) == 0:
                    del final_dict[key]
                else:
                    final_dict[key] = cleaned_dict
        return final_dict
    
    @staticmethod
    def update(ds: dict, nlu_response: dict) -> dict:
        for key, value in nlu_response.items():
            if value is None:
                continue
            # If the value is a dictionary, merge recursively
            if isinstance(value, dict):
                if key not in ds or not isinstance(ds[key], dict):
                    ds[key] = {}
                ds[key] = DialogueStateTracker.update(ds[key], value)
            else:
                # For non-dict values, overwrite directly
                ds[key] = value
        return ds


class BuyingStateTracker(DialogueStateTracker):
    def __init__(self):
        """
        Initializes the BuyingStateTracker with a default dialogue state.
        """
        super().__init__()

        self.dialogue_state = {
            "intent": "buying_car",
            "slots": {
                "car_type": None,
                "budget": None,
                "brand": None,
                "model": None,
                "year": None,
                "fuel_type": None,
                "transmission": None
            }
        }
        self.logger = setup_logger(self.__class__.__name__)

class SellingStateTracker(DialogueStateTracker):
    pass
class RentingStateTracker(DialogueStateTracker):
    pass
class GettingInfoStateTracker(DialogueStateTracker):
    def __init__(self):
        """
        Initializes the GettingInfoStateTracker with a default dialogue state.
        """
        super().__init__()

        self.dialogue_state = {
            "intent": "get_car_info",
            "slots": {
                "car_id": None,
            }
        }
        self.logger = setup_logger(self.__class__.__name__)

class NegotiatePriceStateTracker(DialogueStateTracker):
    def __init__(self):
        """
        Initializes the NegotiatePriceStateTracker with a default dialogue state.
        """
        super().__init__()

        self.dialogue_state = {
            "intent": "negotiate_price",
            "slots": {
                "car_id": None,
                "proposed_price": None
            }
        }
        self.logger = setup_logger(self.__class__.__name__)

class OrderCarStateTracker(DialogueStateTracker):
    def __init__(self):
        """
        Initializes the OrderCarStateTracker with a default dialogue state.
        """
        super().__init__()

        self.dialogue_state = {
            "intent": "order_car",
            "slots": {
                "car_id": None,
                "price": None,
                "name": None,
                "surname": None,
                "id": None,
            } 
        }
        self.logger = setup_logger(self.__class__.__name__)

class BookAppointmentStateTracker(DialogueStateTracker):
    def __init__(self):
        """
        Initializes the BookAppointmentStateTracker with a default dialogue state.
        """
        super().__init__()

        self.dialogue_state = {
            "intent": "book_appointment",
            "slots": {
                "date": None,
                "time": None,
                "name": None,
                "surname": None,
                "id": None,
            }
        }
        self.logger = setup_logger(self.__class__.__name__)

class GiveFeedbackStateTracker(DialogueStateTracker):
    def __init__(self):
        """
        Initializes the GiveFeedbackStateTracker with a default dialogue state.
        """
        super().__init__()

        self.dialogue_state = {
            "intent": "give_feedback",
            "slots": {
                "feedback": None,
                "comment": None,
            }
        }
        self.logger = setup_logger(self.__class__.__name__)