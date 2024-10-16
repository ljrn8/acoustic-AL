""" 
Configuration file for constants and environment.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from colorama import Fore, Style, init

# dotenv constants
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path=dotenv_path)
DATA_ROOT = Path(os.getenv("DATA_ROOT")) 
LOG_LEVEL = os.getenv("LOG_LEVEL") or "INFO"
# LOG_LEVEL = "INFO" 

# Local Directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = Path(ROOT) / "models"
FIGURES_DIR = Path(ROOT) / "figures"
OUTPUT_DIR = Path(ROOT) / "output"
ANNOTATIONS = Path(OUTPUT_DIR) / "annotations"
CORRELATIONS = Path(OUTPUT_DIR) / "correlations"
INTERMEDIATE = Path(OUTPUT_DIR) / "intermediate"


# --- logging ---

# colorama
init(autoreset=True)
LOG_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}

# pretty logs
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelno, Fore.WHITE)
        reset = Style.RESET_ALL
        formatted_message = super().format(record)
        return f"{color}{formatted_message}{reset}"

log = logging.getLogger(__name__)
level = getattr(logging, LOG_LEVEL.upper())

logging.basicConfig(level=level, 
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S",
)

log.debug("Debug logging active")

# console_handler = logging.StreamHandler()
# formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# log.addHandler(console_handler)
