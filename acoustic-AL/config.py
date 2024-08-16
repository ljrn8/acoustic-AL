""" Configuration file
All static variables can be assigned here
"""

import os
import logging; logger = logging.getLogger(__name__)
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

# logging
LOG_LEVEL = os.getenv("LOG_LEVEL") or "DEBUG"
level = getattr(logging, LOG_LEVEL.upper())
logging.basicConfig(level=level)
logger.debug('Debug logging active')

# Local Directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.getenv("DATA_ROOT")  # ../.env
MODEL_DIR = Path(ROOT) / "models"
FIGURES_DIR = Path(ROOT) / "models"
OUTPUT_DIR = Path(ROOT) / "output"
ANNOTATIONS = Path(OUTPUT_DIR) / "annotations"
CORRELATIONS = Path(OUTPUT_DIR) / "correlations"
INTERMEDIATE = Path(OUTPUT_DIR) / "intermediate"