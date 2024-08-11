""" Configuration file
All static variables can be assigned here
"""

import os
from os.path import join
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Local Directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.getenv("DATA_ROOT")  # ../.env
MODEL_DIR = Path(ROOT) / "models"
FIGURES_DIR = Path(ROOT) / "models"
OUTPUT_DIR = Path(ROOT) / "output"
ANNOTATIONS = Path(OUTPUT_DIR) / "annotations"
CORRELATIONS = Path(OUTPUT_DIR) / "correlations"
INTERMEDIATE = Path(OUTPUT_DIR) / "intermediate"