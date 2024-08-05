""" Configuration file
All static variables can be assigned here
"""

import os
from os.path import join
from dotenv import load_dotenv

load_dotenv()

# Local Directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.getenv("DATA_ROOT")  # ../.env
MODEL_DIR = join(ROOT, "models")
FIGURES_DIR = join(ROOT, "models")
OUTPUT_DIR = join(ROOT, "output")
ANNOTATIONS = join(OUTPUT_DIR, "annotations")
CORRELATIONS = join(OUTPUT_DIR, "correlations")
