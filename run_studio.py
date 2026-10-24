"""Launch the local visualization console backed by the new V2 agent."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
from simulation.studio import main

if __name__ == '__main__':
    raise SystemExit(main())
