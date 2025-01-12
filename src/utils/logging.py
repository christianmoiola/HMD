import logging

# Define color codes for the terminal
COLORS = {
    "DEBUG_DEFAULT": "\033[92m",        # Bright Green (default for debug)
    "DEBUG_MAIN": "\033[35m" ,          # Magenta
    "DEBUG_PRE_NLU": "\033[34m",        # Blue
    "DEBUG_NLU": "\033[36m",            # Cyan
    "DEBUG_DM": "\033[32m",             # Green
    "DEBUG_NLG": "\033[33m",            # Dark Yellow
    "DEBUG_STATE_TRACKER": "\033[95m",  # Light Magenta
    "DEBUG_HISTORY": "\033[97m",        # White
    "INFO": "\033[96m",                 # Bright Cyan
    "WARNING": "\033[93m",              # Bright Yellow
    "FAIL": "\033[91m",                 # Bright Red
    "ENDC": "\033[0m",                  # Reset
    "BOLD": "\033[1m",                  # Bold
}

# Mapping of logging level strings to logging constants
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors and include the class name in log messages.
    """
    def __init__(self, fmt, color_map):
        super().__init__(fmt)
        self.color_map = color_map

    def format(self, record):
        log_color = self.color_map.get(record.levelname, COLORS["ENDC"])
        record.msg = f"{log_color}{record.msg}{COLORS['ENDC']}"  # Add color
        record.name = f"{COLORS['BOLD']}{record.name}{COLORS['ENDC']}"  # Bold class name
        return super().format(record)


def setup_logger(name, logging_level="DEBUG", color_debug="DEBUG_DEFAULT"):
    """
    Create and configure a logger for a specific class.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS[logging_level])
    handler = logging.StreamHandler()

    color_map = {
        "DEBUG": COLORS[color_debug],
        "INFO": COLORS["INFO"],
        "WARNING": COLORS["WARNING"],
        "ERROR": COLORS["FAIL"],
        "CRITICAL": COLORS["FAIL"],
    }

    formatter = ColoredFormatter(
        fmt="%(name)s - %(levelname)s - %(message)s",
        color_map=color_map,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


