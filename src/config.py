""" Configuration file
All static variables should be assigned here 
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path=dotenv_path)

# logging
log = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL") or "DEBUG"
level = getattr(logging, LOG_LEVEL.upper())
logging.basicConfig(level=level)
log.debug("Debug logging active")

# Local Directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = Path(os.getenv("DATA_ROOT"))  # ../.env
MODEL_DIR = Path(ROOT) / "models"
FIGURES_DIR = Path(ROOT) / "figures"
OUTPUT_DIR = Path(ROOT) / "output"
ANNOTATIONS = Path(OUTPUT_DIR) / "annotations"
CORRELATIONS = Path(OUTPUT_DIR) / "correlations"
INTERMEDIATE = Path(OUTPUT_DIR) / "intermediate"

# other useful 'constants'
ANN_DF = pd.read_csv(ANNOTATIONS / "initial_dataset_7depl_metadata.csv")
ANNOTATED_RECORDING = "1_20230316_063000.wav"



