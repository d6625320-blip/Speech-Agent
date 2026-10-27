"""Run from any directory, without requiring an editable package installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emotion_agent.main import main

if __name__ == "__main__":
    main()
