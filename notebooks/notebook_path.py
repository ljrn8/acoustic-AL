import sys
from pathlib import Path

p = Path(__file__).parent.parent.resolve() / "acoustic-AL"
sys.path.insert(1, str(p))
