from src.utils.logging import setup_logger
import json
import os
import re

class Database():
    def __init__(self, cfg: dict):
        # Path project
        self.path = cfg["Settings"].get("path")
        self.path_database = os.path.join(self.path, cfg["DB"].get("path"))
        self.logger = setup_logger(self.__class__.__name__)
        self.database = self.load_database()
        self.fields = self.get_fields()

    def load_database(self):
        with open(self.path_database) as f:
            data = json.load(f)
        return data
    
    def get_fields(self):
        fields = set()
        # Ensure the database is not empty before attempting to access its first element
        if self.database:
            sample_car = self.database[0]
            fields.update(sample_car.keys())  # Add all keys (fields) to the set
        return list(fields)

    def clean_action(self, action: str):
        match = re.match(r'(\w+)\(([^)]*)\)', action)
        if not match:
            self.logger.error(f"Invalid action: {action}")
            return None, None
        
        # Extract the action name
        action_name = match.group(1)
        # Extract the parameters
        parameters = match.group(2)
        # Extract field-value pairs
        fields_values = {}
        if parameters.strip(): # Only try to find pairs if parameters string is not empty
            try:
                # This handles parameters like 'car_id=5' or 'brand="Toyota", year=2020'
                # It evaluates the string as a Python dictionary literal
                # Use json.loads to parse if the string is strict JSON, otherwise ast.literal_eval is safer for Python literals
                import ast
                parsed_params = ast.literal_eval(f"{{{parameters}}}")
                if isinstance(parsed_params, dict):
                    fields_values = {k.strip(): v for k, v in parsed_params.items()}
                else:
                    self.logger.error(f"Invalid parameters format: {parameters}")
                    return None, None
            except (SyntaxError, ValueError) as e:
                self.logger.error(f"Error parsing parameters: {parameters}. Error: {e}")
                return None, None


        # Check if the fields are valid (only for keys that are actually in self.fields)
        for field in fields_values.keys():
            if field not in self.fields:
                self.logger.error(f"Invalid field: {field}")
                return None, None

        return action_name, fields_values

    def find_cars_for_purchase(self, slots: dict):
        self.logger.info("Finding cars for purchase...")
        result = []
        for car in self.database:
            match = True
            for field, value in slots.items():
                if value is None:
                    continue  # Skip None values in slots

                car_db_field = car.get(field)

                if car_db_field is None: # If the car does not have the field, it's not a match for that field
                    match = False
                    break
                
                # Type conversions and comparisons
                if field == "budget":
                    try:
                        car_budget = float(car_db_field)
                        target_budget = float(value)
                        if car_budget > target_budget:
                            match = False
                            break
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not convert budget for car_id: {car.get('car_id')} or target value: {value}")
                        match = False
                        break
                elif field == "year":
                    try:
                        car_year = int(car_db_field)
                        target_year = int(value)
                        if car_year < target_year: # Assuming 'year' means car year should be >= target year
                            match = False
                            break
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not convert year for car_id: {car.get('car_id')} or target value: {value}")
                        match = False
                        break
                else:
                    # Case-insensitive comparison for string fields
                    if isinstance(car_db_field, str) and isinstance(value, str):
                        if car_db_field.lower() != value.lower():
                            match = False
                            break
                    elif car_db_field != value:
                        match = False
                        break
            if match:
                result.append(car)
        return str(result)

    def find_car_by_id(self, car_id: str):
        self.logger.info(f"Finding car with ID: {car_id}...")
        try:
            target_id = int(car_id)
            for car in self.database:
                if car.get("car_id") == target_id:
                    return car
            return "None" # Return "None" if car not found, to be consistent with string output
        except (ValueError, TypeError):
            self.logger.error(f"Invalid car_id provided: {car_id}")
            return "None"

    def get_car_info(self, nlu_response: dict):
        self.logger.info("Getting car information from the database...")
        slots = nlu_response['slots']
        car_id = slots['car_id'] if 'car_id' in slots.keys() else None
        if not car_id:
            self.logger.error("No 'car_id' found in the response.")
            return "None"
        for car in self.database:
            if car["car_id"] == car_id:
                if slots["info_type"] in car.keys():
                    car_info = f"{slots['info_type']}: {car[slots['info_type']]}"
                    return car_info
                else:
                    self.logger.error(f"Info type '{slots['info_type']}' not found for car_id {car_id}.")
                    return "None"
        
    
    def query_database(self, intent_data: dict):
        self.logger.info("Querying the database with intent data...")
        
        intent = intent_data.get('intent')
        slots = intent_data.get('slots', {})

        if intent == 'buying_car':
            return self.find_cars_for_purchase(slots)
        else:
            car_id = slots.get('car_id')
            if car_id:
                return self.find_car_by_id(car_id)
            else:
                self.logger.error("Missing 'car_id'")
                return "None"