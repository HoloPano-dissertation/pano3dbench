import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "front3d"

for folder in ("adapters", "qa"):
    path = str(SCRIPTS / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
