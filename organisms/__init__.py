"""The backdoored test models of the paper: their training data, training and behaviour.

Each organism has a config in organisms/configs/: its trigger (a string, or the words of a trigger topic),
target words, the words its suppress variant penalises, the training recipe and how its behaviour is measured.
"""

import json
from pathlib import Path

CONFIGS = Path(__file__).parent / "configs"
ORGANISMS = sorted(p.stem for p in CONFIGS.glob("*.json"))


def load_config(organism: str) -> dict:
    return json.loads((CONFIGS / f"{organism}.json").read_text())
