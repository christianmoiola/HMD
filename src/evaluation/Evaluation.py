from sklearn.metrics import precision_recall_fscore_support
import random
from src.utils.logging import setup_logger
from src.components.NLU import NLU, PRE_NLU
from src.components.DM import DM
from src.components.NLG import NLG
from itertools import product
from src.utils.utils_model import get_model
from tqdm import tqdm


class Evaluation():
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model, self.tokenizer = get_model(self.cfg)
        self.logger = setup_logger(self.__class__.__name__)

    def load_json(self, path: str):
        import json
        with open(path, 'r') as file:
            return json.load(file)
        
    def test_dm(self):
        dm = DM(cfg=self.cfg, model=self.model, tokenizer=self.tokenizer, history=None, logging_level="ERROR")
        
        test_cases = self.load_json(self.cfg["EVALUATION"].get("nlu_test_cases"))

        correct_actions = 0
        total_actions = 0
        loading_bar = tqdm(test_cases, desc="Testing DM", unit="test case")

        for el in loading_bar:
            input = el["expected_output"]
            dm_response = dm.query_model(input)

            if dm_response is None:
                total_actions += 1
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

            loading_bar.set_postfix({"Acc": f"{(correct_actions / total_actions) * 100:.2f}%"})


    def test_nlu(self):
        nlu = NLU(cfg=self.cfg, model=self.model, tokenizer=self.tokenizer, history=None, logging_level="ERROR")

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
            nlu_response = nlu.query_model(input)
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

    def test_pre_nlu(self):
        pre_nlu = PRE_NLU(cfg=self.cfg, model=self.model, tokenizer=self.tokenizer, history=None, logging_level="ERROR")
        test_cases = self.load_json(self.cfg["EVALUATION"].get("pre_nlu_test_cases"))

        y_true = []
        y_pred = []
        all_intents = set()
        total_intents = 0
        correct_intents = 0

        loading_bar = tqdm(test_cases, desc="Testing PRE_NLU", unit="test case")
        for el in loading_bar:
            input = el["input"]
            expected_output = el["expected_output"]
            pre_nlu_response = pre_nlu.query_model(input)

            expected_intents = [elem["intent"] for elem in expected_output]

            if pre_nlu_response is None:
                predicted_intents = []
                self.logger.warning(f"Model returned None:\nInput: {input}\nExpected: {expected_intents}")
            else:
                predicted_intents = [elem["intent"] for elem in pre_nlu_response]

            # Count for accuracy
            total_intents += len(expected_intents)
            for intent in expected_intents:
                if intent in predicted_intents:
                    correct_intents += 1
                    predicted_intents.remove(intent)

            all_intents.update(expected_intents)
            all_intents.update(predicted_intents)

            y_true.append(expected_intents)
            y_pred.append(predicted_intents)

            loading_bar.set_postfix({"Acc Intents": f"{(correct_intents / total_intents) * 100:.2f}%"})

        # Intent indexing
        sorted_intents = sorted(all_intents)
        intent_to_index = {intent: idx for idx, intent in enumerate(sorted_intents)}

        def to_binary_vector(intent_list):
            vec = [0] * len(sorted_intents)
            for intent in intent_list:
                if intent in intent_to_index:
                    vec[intent_to_index[intent]] = 1
            return vec

        y_true_bin = [to_binary_vector(intents) for intents in y_true]
        y_pred_bin = [to_binary_vector(intents) for intents in y_pred]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average=None, zero_division=0
        )

        # Accuracy over total intents
        overall_accuracy = (correct_intents / total_intents) * 100 if total_intents > 0 else 0.0

        print(f"\n📊 Intent-Level Evaluation (per intent):")
        print(f"{'Intent':<30} {'Prec.':>7} {'Recall':>7} {'F1':>7}")
        print("-" * 60)
        for idx, intent in enumerate(sorted_intents):
            if support[idx] > 0:  # only show intents that exist in ground truth
                print(f"{intent:<30} {precision[idx]*100:7.2f} {recall[idx]*100:7.2f} {f1[idx]*100:7.2f}")

        print("\n✅ Overall Intent Accuracy: {:.2f}% ({}/{})".format(overall_accuracy, correct_intents, total_intents))