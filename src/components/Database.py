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
        fields_values = {field.strip(): value.strip(' \'"') for field, value in re.findall(r'(\w+)\s*=\s*([^,]+)', parameters)}

        # Check if the action has parameters
        if not fields_values:
            self.logger.error(f"Invalid action: {action}, no parameters found")
            return None, None
        
        # Check if the fields are valid
        for field in fields_values.keys():
            if field not in self.fields:
                self.logger.error(f"Invalid field: {field}")
                return None, None

        return action_name, fields_values

    def query_database(self, action: str):
        self.logger.info("Querying the database...")
        action_name, fields_values = self.clean_action(action)

        if not action_name and not fields_values:
            return None
        
        self.logger.debug(f"Action name: {action_name}, fields_values: {fields_values}")

        # Check if the action is valid
        if action_name != "find":
            self.logger.error(f"Invalid action: {action_name}")
            return None
        
        # Find in the database
        result = []
        for car in self.database:
            match = True
            for field, value in fields_values.items():
                car_db_field = str(car.get(field))
                if field == "budget":
                    car_db_field = float(car_db_field)
                    value = float(value)
                    if car_db_field > value:
                        match = False
                        break
                elif field == "year":
                    car_db_field = int(car_db_field)
                    value = int(value)
                    if car_db_field < value:
                        match = False
                        break
                elif car_db_field != value:
                    match = False
                    break
            if match:
                result.append(car)
        return str(result)
        
        


