#!/usr/bin/env python3
"""Unified entry point; select robots/cameras only at this composition root."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from agent.cli import main

if __name__=="__main__":
    raise SystemExit(main())
