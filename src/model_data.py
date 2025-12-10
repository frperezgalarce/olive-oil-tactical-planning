
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json

@dataclass
class OliveHarvestData:
    T: List[int]
    J: List[str]
    K: List[str]

    K1: List[str]
    K2: List[str]
    K3: List[str]
    K4: List[str]

    S: Dict[str, float]
    F: Dict[str, float]
    OC: Dict[str, Dict[int, float]]
    OL: Dict[str, Dict[int, float]]
    UC: Dict[str, float]
    PD: Dict[str, Dict[str, float]]
    OSn: Dict[str, float]
    CP: Dict[int, float]
    BH: float
    HB: float
    HMC: float
    HMB: float
    OE: Dict[int, float]
    HMM: float

    def validate(self):
        # Same validation as before...
        for j in self.J:
            if j not in self.S or j not in self.F or j not in self.OSn:
                raise ValueError(f"Missing S/F/OSn for lot {j}")
        for k in self.K:
            if k not in self.UC:
                raise ValueError(f"Missing UC for team {k}")

    # ------------------------------------------------
    # NEW METHOD: Create OliveHarvestData from dict
    # ------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict):
        """
        Build a fully typed OliveHarvestData from a single dictionary.
        Only keys defined in the dataclass are extracted.
        """
        expected_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in expected_fields}

        missing = expected_fields - filtered.keys()
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return cls(**filtered)
    
    @classmethod
    def from_json(cls, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)

        # Convert keys that must be int
        for j in data["OC"]:
            data["OC"][j] = {int(t): v for t, v in data["OC"][j].items()}

        for j in data["OL"]:
            data["OL"][j] = {int(t): v for t, v in data["OL"][j].items()}

        data["CP"] = {int(t): v for t, v in data["CP"].items()}
        data["OE"] = {int(t): v for t, v in data["OE"].items()}

        return cls.from_dict(data)