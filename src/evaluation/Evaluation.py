from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from src.utils.logging import setup_logger
from src.components.NLU import NLU, PRE_NLU
from src.components.DM import DM
from src.components.NLG import NLG
from src.utils.history import History
from src.utils.utils_model import get_model
from tqdm import tqdm


class Evaluation():
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model, self.tokenizer = get_model(self.cfg)
        self.logger = setup_logger(self.__class__.__name__)
        self.history = History()

    def load_json(self, path: str):
        import json
        with open(path, 'r') as file:
            return json.load(file)

    def test_dm(self, is_history: bool = False):
        dm = DM(cfg=self.cfg, model=self.model, tokenizer=self.tokenizer, history=self.history if is_history else None, logging_level="ERROR")

        test_cases = self.load_json(self.cfg["EVALUATION"].get("nlu_test_cases"))

        correct_actions = 0
        total_actions = 0

        correct_parameters = 0
        total_parameters = 0

        loading_bar = tqdm(test_cases, desc="Testing DM", unit="test case")

        for el in loading_bar:
            input = el["expected_output"]

            if is_history:
                self.history.clear_history()
                if "history" in el:
                    for msg in el["history"]:
                        self.history.add_to_history(msg["sender"], msg["msg"])
                        
                        
            dm_response = None
            count = 5
            while dm_response is None and count > 0:
                dm_response = dm.query_model(input)
                count -= 1

            if dm_response is None:
                total_actions += 1
                total_parameters += 1
            else:
                expected_action = ""
                if input["intent"] == "buying_car":
                    # count how many slots are not None
                    filled_slots = sum(1 for slot in input["slots"].values() if slot is not None)
                    if filled_slots >= 2:
                        expected_action = "confirmation"
                    else:
                        expected_action = "request_info"
                else:
                    filled_slots = sum(1 for slot in input["slots"].values() if slot is not None)
                    total_slots = len(input["slots"])
                    if filled_slots == total_slots:
                        expected_action = "confirmation"
                    else:
                        expected_action = "request_info"
                if dm_response["action"] == expected_action:
                    correct_actions += 1
                else:
                    self.logger.error(f"Test case failed:\nInput: {input}\nExpected: {expected_action}\nGot: {dm_response['action']}")
                total_actions += 1

                if "parameter" in dm_response:
                    total_parameters += 1
                    if expected_action == "confirmation":
                        if dm_response["parameter"] == input["intent"]:
                            correct_parameters += 1
                    elif expected_action == "request_info":
                        expected_slot = next((slot for slot, value in input['slots'].items() if value is None), None)
                        if expected_slot == dm_response["parameter"]:
                            correct_parameters += 1
                        elif expected_slot is None:
                            self.logger.error(f"No slots are None in input: {input}")
                    else:
                        self.logger.error(f"Unexpected action: {dm_response['action']} with parameter: {dm_response['parameter']} for input: {input}")
                else:
                    self.logger.error(f"DM response does not contain 'parameter': {dm_response} for input: {input}")

            loading_bar.set_postfix({"Acc": f"{(correct_actions / total_actions) * 100:.2f}%", "Param Acc": f"{(correct_parameters / total_parameters) * 100:.2f}%"})

    def test_nlu(self, is_history: bool = False):

        nlu = NLU(cfg=self.cfg, model=self.model, tokenizer=self.tokenizer, history=self.history if is_history else None, logging_level="ERROR")

        test_cases = self.load_json(self.cfg["EVALUATION"].get("nlu_test_cases"))

        total_intent = 0
        correct_intent = 0
        total_slots = 0
        correct_slots = 0
        # cycle in the list of json test cases
        loading_bar = tqdm(test_cases, desc="Testing NLU", unit="test case")
        for el in loading_bar:
            input = el["input"]
            expected_output = el["expected_output"]

            if is_history:
                self.history.clear_history()
                if "history" in el:
                    for msg in el["history"]:
                        self.history.add_to_history(msg["sender"], msg["msg"])

            nlu_response = None
            count = 5
            while nlu_response is None and count > 0:
                nlu_response = nlu.query_model(input)
                count -= 1

            if nlu_response != None:
                # Check intent
                if nlu_response["intent"] == expected_output["intent"]:
                    correct_intent += 1
                total_intent += 1

                # Check slots
                for slot in expected_output["slots"]:
                    if slot in nlu_response["slots"] and nlu_response["slots"][slot] == expected_output["slots"][slot]:
                        correct_slots += 1
                    total_slots += 1
            else:
                self.logger.error(f"Test case failed:\nInput: {input}\nExpected: {expected_output}\nGot: {nlu_response}")
                total_intent += 1
            loading_bar.set_postfix({"Acc Intent": f"{(correct_intent / total_intent) * 100:.2f}%", "Acc Slots": f"{(correct_slots / total_slots) * 100:.2f}%"})

    def test_pre_nlu(self, is_history: bool = False):

        pre_nlu = PRE_NLU(cfg=self.cfg, model=self.model, tokenizer=self.tokenizer, history=self.history if is_history else None, logging_level="ERROR")
        test_cases = self.load_json(self.cfg["EVALUATION"].get("pre_nlu_test_cases"))

        y_true = []
        y_pred = []
        total_intents = 0
        correct_intents = 0

        loading_bar = tqdm(test_cases, desc="Testing PRE_NLU", unit="test case")
        for el in loading_bar:
            input = el["input"]
            expected_output = el["expected_output"]

            if is_history:
                self.history.clear_history()
                if "history" in el:
                    for msg in el["history"]:
                        self.history.add_to_history(msg["sender"], msg["msg"])

            pre_nlu_response = None
            count = 5
            while pre_nlu_response is None and count > 0:
                pre_nlu_response = pre_nlu.query_model(input)
                count -= 1

            expected_intents = [elem["intent"] for elem in expected_output]

            if pre_nlu_response is None:
                predicted_intents = []
                self.logger.warning(f"Model returned None:\nInput: {input}\nExpected: {expected_intents}")
            else:
                predicted_intents = [elem["intent"] for elem in pre_nlu_response]

            y_true.append(expected_intents.copy())
            y_pred.append(predicted_intents.copy())
            # Count for accuracy
            total_intents += len(expected_intents)
            for intent in expected_intents:
                if intent in predicted_intents:
                    correct_intents += 1
                    predicted_intents.remove(intent)

            loading_bar.set_postfix({"Acc Intents": f"{(correct_intents / total_intents) * 100:.2f}%"})

        # Initialize MultiLabelBinarizer and fit it to all unique intents
        mlb = MultiLabelBinarizer(classes=["buying_car", "negotiate_price", "order_car", "get_car_info", "give_feedback", "book_appointment", "out_of_domain"])
        mlb.fit([])

        # Transform the lists of intents into a binary format
        y_true_binary = mlb.transform(y_true)
        y_pred_binary = mlb.transform(y_pred)

        # Get the classification report
        report = classification_report(y_true_binary, y_pred_binary, target_names=mlb.classes_)

        print(report)