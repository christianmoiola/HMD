from src.utils.logging import setup_logger

class History:
    def __init__(self):
        self.senders = []
        self.msgs = []

        self.logger = setup_logger(self.__class__.__name__)

    def add_to_history(self, sender: str, msg: str):
        sender = sender.lower()
        if sender not in ["user", "system"]:
            raise ValueError("Sender must be either 'user' or 'system'")
            logger.error("Sender must be either 'user' or 'system'")
        else:
            self.senders.append(sender)
            self.msgs.append(msg)
    
    def get_history(self):
        history = ""
        if len(self.senders) == 0:
            history = ""
        else:
            history = ""
            # Only show the last 5 messages
            for i in range(max(0, len(self.senders) - 5), len(self.senders)):
                history += f"{self.senders[i].capitalize()}: {self.msgs[i]}\n" 
                    
        return history
    