"""Run configuration for the benchmark pipeline (dataclass + JSON/YAML loading)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class CostConfig:
    # SE (street) – paper notation beta^SE, c^SE, c^S
    beta_SE: Optional[float] = None  # None -> shift length of the instance (paper: "vehicle use cost = shift duration")
    c_SE: float = 1.0
    c_S: float = 0.0  # benchmark instances have no satellite handling cost
    # FE (water) – beta^FE, c^FE
    beta_FE: Optional[float] = None
    c_FE: float = 1.0


@dataclass
class ModelConfig:
    synchronized: bool = True  # Synch (storage-free) vs Asynch (storage at satellites)
    sync_mode: str = "retime"  # "retime": SE times re-optimised jointly with FE (as in the code);
    #                             "fixed": SE arrival times fixed, FE windows [e^FE, l^FE] as in the paper (eq. 23)
    fe_depart_after_zero: bool = True  # u_j >= t^FE(w, p(j)) (vessels leave the warehouse at time >= 0)
    fe_return_by_horizon: bool = True  # u_j + U + t^FE(p(j), w) <= horizon of the warehouse
    se_return_by_horizon: bool = True  # SEV must be back at the garage by the end of its shift
    time_limit_SE: float = 600.0  # seconds (paper: 10 or 60 min for MILP_SE)
    time_limit_FE: float = 100.0  # seconds (paper: 100 s for benchmark FE problems)
    mip_gap: float = 1e-4
    backend: Optional[str] = None  # None -> auto (highs > scipy > gurobi)
    verbose: bool = False
    threads: Optional[int] = None


@dataclass
class RunConfig:
    instances: List[str] = field(default_factory=lambda: ["c201.txt"])
    n_customers: int = 100
    output_dir: str = "results"
    tag: str = "milp"
    cost: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # Allows the first checks on smaller problems: keep the first n customers only.

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        d = dict(d)
        cost = CostConfig(**d.pop("cost", {}))
        model = ModelConfig(**d.pop("model", {}))
        return RunConfig(cost=cost, model=model, **d)

    @staticmethod
    def load(path: str) -> "RunConfig":
        with open(path) as f:
            text = f.read()
        if path.endswith((".yaml", ".yml")):
            import yaml  # optional dependency

            return RunConfig.from_dict(yaml.safe_load(text))
        return RunConfig.from_dict(json.loads(text))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def resolve_costs(problem, cost: CostConfig) -> dict:
    """Fill the None defaults from the instance (shift length as vehicle-use cost)."""
    shift_SE = float(problem.depot.end_interval - problem.depot.start_interval)
    shift_FE = float(problem.CDC.end_interval - problem.CDC.start_interval)
    return {
        "beta_SE": shift_SE if cost.beta_SE is None else float(cost.beta_SE),
        "c_SE": float(cost.c_SE),
        "c_S": float(cost.c_S),
        "beta_FE": shift_FE if cost.beta_FE is None else float(cost.beta_FE),
        "c_FE": float(cost.c_FE),
    }
