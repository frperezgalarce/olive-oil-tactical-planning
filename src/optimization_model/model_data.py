
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import math
import pulp as pl
from typing import Any, Optional


class OliveHarvestDataFromJSON:
    """
    Parses a params_model.json into the symbols used by the optimization model.

    Expected keys in JSON:
      - vari: list of lot ids (J)
      - CA:   list length |T| -> CP[t]
      - C:    list length >= max(K) -> UC[k]
      - Sup:  list length |J| -> S[j]
      - G:    list length |J| -> mu[j]
      - GP:   list length |J| where each entry is a list length |T| -> OL[j][t]
             (Option A: GP is fruta caída / curva de riesgo climático)
      - AA:   matrix |J| x |T| -> OC[j][t]
      - W:    list length |T| -> OE[t] (optional)
      - modoCosecha: list of 4 lists -> K1,K2,K3,K4 (as ids)
      - Po:   matrix (>= max(K)) x |J| -> PD[j][k]  (optional)

    Notes:
      - Option A: sigma[j] is set to 0.0 (to be estimated later).
      - Budget BH is not included here unless you add it to your JSON. If you still
        need BH, add raw["BH"] and map it.
      - F, HMC, HMB, HMM default to 0.0 (inactive) unless present in JSON.
    """

    def __init__(self, json_path: str, alpha: float = 0.95):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.raw: Dict[str, Any] = raw
        self.alpha: float = float(alpha)

        # -------------------------
        # Sets
        # -------------------------
        self.J: List[Any] = list(raw["vari"])
        self.T: List[int] = list(range(1, len(raw["CA"]) + 1))
        self.T0: List[int] = [0] + self.T

        self.K1, self.K2, self.K3, self.K4 = raw["modoCosecha"]
        self.K1 = [int(k) for k in self.K1]
        self.K2 = [int(k) for k in self.K2]
        self.K3 = [int(k) for k in self.K3]
        self.K4 = [int(k) for k in self.K4]
        self.K: List[int] = sorted(set(self.K1 + self.K2 + self.K3 + self.K4))
        self.K1_comp: List[int] = [k for k in self.K if k not in self.K1]

        # Position map for lots
        j_pos = {j: i for i, j in enumerate(self.J)}

        # -------------------------
        # Parameters
        # -------------------------

        # mu_j (expected initial fruit per lot)
        self.mu: Dict[Any, float] = {j: float(raw["G"][j_pos[j]]) for j in self.J}

        # Option A: sigma_j fixed to 0.0 (placeholder you will estimate)
        self.sigma: Dict[Any, float] = {j: 0.0 for j in self.J}

        # area S_j
        self.S: Dict[Any, float] = {j: float(raw["Sup"][j_pos[j]]) for j in self.J}
        self.G: Dict[Any, float] = {j: float(raw["G"][j_pos[j]]) for j in self.J}

        # UC_k : JSON "C" is 0-based list; k are 1..max(K)
        self.UC: Dict[int, float] = {k: float(raw["C"][k - 1]) for k in self.K}

        # OC_{j,t}: AA is |J| x |T|
        self.OC: Dict[Any, Dict[int, float]] = {
            j: {t: float(raw["AA"][j_pos[j]][t - 1]) for t in self.T}
            for j in self.J
        }

        # Plant capacity CP_t: CA length |T|
        self.CP: Dict[int, float] = {t: float(raw["CA"][t - 1]) for t in self.T}

        # Exogenous inflow OE_t: optional W length |T|
        if "W" in raw and isinstance(raw["W"], list) and len(raw["W"]) == len(self.T):
            self.OE: Dict[int, float] = {t: float(raw["W"][t - 1]) for t in self.T}
        else:
            self.OE = {t: 0.0 for t in self.T}

        # Climate loss curve OL_{j,t}: from GP (Option A)
        # GP is a list over lots; each entry is a list over time
                # Climate loss curve OL_{j,t}: from GP (Option A)
        # GP is a list over lots; each entry is a list over time
        self.OL: Dict[Any, Dict[int, float]] = {}
        T_len = len(self.T)

        for j in self.J:
            gp_entry = raw["GP"][j_pos[j]]

            if not isinstance(gp_entry, list):
                raise TypeError(
                    f'GP for lot {j} must be a list (loss curve over time). Got: {type(gp_entry)}'
                )

            # Accept either length T (for t=1..T) or length T+1 (for t=0..T)
            if len(gp_entry) == T_len:
                gp_aligned = gp_entry
            elif len(gp_entry) == T_len + 1:
                # Assume first value is for t=0; drop it to align with T=1..T
                gp_aligned = gp_entry[1:]
            else:
                raise ValueError(
                    f"GP curve length for lot {j} is {len(gp_entry)} but expected {T_len} (T) "
                    f"or {T_len + 1} (T0)."
                )

            self.OL[j] = {t: float(gp_aligned[t - 1]) for t in self.T}

        # Productivity PD_{j,k}: optional (Po is K x |J|)
        # We store as PD[j][k] to match your model usage: d.PD[j][k]
        self.PD: Optional[Dict[Any, Dict[int, float]]] = None
        if "Po" in raw and isinstance(raw["Po"], list) and len(raw["Po"]) >= max(self.K):
            # raw["Po"][k-1] is a list over J positions
            self.PD = {
                j: {k: float(raw["Po"][k - 1][j_pos[j]]) for k in self.K}
                for j in self.J
            }

        # Fraction requirement F_j (optional)
        if "F" in raw and isinstance(raw["F"], list) and len(raw["F"]) == len(self.J):
            self.F: Dict[Any, float] = {j: float(raw["F"][j_pos[j]]) for j in self.J}
        else:
            self.F = {j: 0.0 for j in self.J}

        # Operational minimums (optional)
        self.HMC: float = float(raw.get("HMC", 0.0))
        self.HMB: float = float(raw.get("HMB", 0.0))
        self.HMM: float = float(raw.get("HMM", 0.0))

        # Optional: budget BH (only if you still use Formulation #1)
        self.BH: Optional[float] = float(raw["BH"]) if "BH" in raw else None

    def validate(self) -> None:
        raw = self.raw

        # Basic presence
        required = ["vari", "CA", "C", "Sup", "G", "GP", "AA", "modoCosecha"]
        missing = [k for k in required if k not in raw]
        if missing:
            raise KeyError(f"Missing required JSON keys: {missing}")

        # Shapes
        if len(raw["G"]) != len(self.J):
            raise ValueError(f'len(G)={len(raw["G"])} must equal |J|={len(self.J)}')

        if len(raw["Sup"]) != len(self.J):
            raise ValueError(f'len(Sup)={len(raw["Sup"])} must equal |J|={len(self.J)}')

        if len(raw["AA"]) != len(self.J):
            raise ValueError(f'AA must have |J| rows. len(AA)={len(raw["AA"])} vs |J|={len(self.J)}')

        if len(raw["CA"]) != len(self.T):
            raise ValueError(f'CA must have |T| entries. len(CA)={len(raw["CA"])} vs |T|={len(self.T)}')

        # AA columns
        for i, row in enumerate(raw["AA"]):
            if not isinstance(row, list) or len(row) != len(self.T):
                raise ValueError(
                    f"AA row {i} must be a list of length |T|={len(self.T)}. Got length={len(row) if isinstance(row, list) else 'N/A'}"
                )

        # GP must be |J| curves each of length |T|
        if len(raw["GP"]) != len(self.J):
            raise ValueError(f'len(GP)={len(raw["GP"])} must equal |J|={len(self.J)}')
        for i, gp in enumerate(raw["GP"]):
            if not isinstance(gp, list):
                raise TypeError(f"GP[{i}] must be a list (curve over T). Got {type(gp)}")
            if len(gp) != len(self.T):
                raise ValueError(f"GP[{i}] length {len(gp)} must equal |T|={len(self.T)}")

        # UC coverage
        if len(raw["C"]) < max(self.K):
            raise ValueError(f'C must provide UC for all k in K up to {max(self.K)}. len(C)={len(raw["C"])}')

        # Optional OL range check (recommended)
        for j in self.J:
            for t in self.T:
                v = self.OL[j][t]
                if v < 0 or v > 1:
                    raise ValueError(f"OL[j,t] out of [0,1]: lot={j}, t={t}, value={v}")

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