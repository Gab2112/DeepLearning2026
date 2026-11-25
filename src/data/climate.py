"""
Climate vocabulary, encoding, and climate dropout utilities for Köppen climate zones.
Handles the 'unknown' category explicitly to guarantee generalization to OOD-S.
"""

from typing import Dict, Union, Optional
import numpy as np
import torch

CLIMATE_TO_ID: Dict[str, int] = {
    "<UNKNOWN>": 0,
    "unknown": 0,
    "none": 0,
    "nan": 0,
    "BSk": 1,  # Cold semi-arid (Steppe)
    "Cfa": 2,  # Humid subtropical (mild with hot summer)
    "Cfb": 3,  # Oceanic temperate (mild with warm summer)
    "Csa": 4,  # Mediterranean hot summer
    "Csb": 5,  # Mediterranean warm summer
    "Dfa": 6,  # Humid continental hot summer
    "Dfb": 7,  # Humid continental mild summer
    "Dfc": 8,  # Subarctic
}

ID_TO_CLIMATE: Dict[int, str] = {v: k for k, v in CLIMATE_TO_ID.items() if k not in ["unknown", "none", "nan"]}
NUM_CLIMATES = 9  # 0 to 8 inclusive


def encode_climate(climate_str: Optional[str]) -> int:
    """
    Safely maps any climate string to an integer ID [0, 8].
    Defaults to 0 (<UNKNOWN>) for missing, unknown or unrecognized strings.
    """
    if climate_str is None:
        return 0
    clean = str(climate_str).strip()
    return CLIMATE_TO_ID.get(clean, 0)
