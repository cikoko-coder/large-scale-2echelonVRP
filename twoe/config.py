"""Run configuration for the benchmark pipeline (dataclass + JSON/YAML loading)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class CostConfig:
    # SE (street) – paper notation beta^SE, c^SE, c^S
    beta_SE: Optional[float] = None  # None -> beta(c^s, c^w) of the paper, eq. (62), see resolve_costs
    c_SE: float = 1.0
    c_S: float = 0.0  # benchmark instances have no satellite handling cost
    # FE (water) – beta^FE, c^FE
    beta_FE: Optional[float] = None  # None -> alpha * beta (water level significance alpha = WLS)
    c_FE: Optional[float] = None  # None -> alpha * c_SE


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
    backend: Optional[str] = None  # None -> twoe.solver.default_backend() (MIP_SOLVER env, else highs > scipy > gurobi)
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


def fixed_vehicle_cost(problem, c_SE: float = 1.0, c_FE: float = 1.0) -> float:
    """beta(c^s, c^w) of the paper (eq. 62): the most expensive way to serve one customer with
    one LEFV and one vessel, max over customers i and satellites p of
    c^s (t_gi + t_ip + t_pg) + c^w (t_dp + t_pd). It is at least the cost of serving the
    customer in an existing route, so the fixed costs make the model prefer fewer vehicles."""
    t = problem.travel_matrix
    g, d = problem.depot.index, problem.CDC.index
    return max(c_SE*(t[g][i.index] + t[i.index][p.index] + t[p.index][g]) + c_FE*(t[d][p.index] + t[p.index][d])
               for i in problem.customers for p in problem.satellites)


def resolve_costs(problem, cost: CostConfig, wls: float = 1.0) -> dict:
    """Fill the None defaults as in the paper (Section 5.1): with the water level significance
    alpha = wls, [c^s, c^w, beta^s, beta^w] = [c_SE, alpha * c_SE, beta, alpha * beta] where
    beta = fixed_vehicle_cost(c^s, c^w) (eq. 62). Explicit values in `cost` are kept."""
    c_SE = float(cost.c_SE)
    c_FE = float(cost.c_FE) if cost.c_FE is not None else wls*c_SE
    beta = fixed_vehicle_cost(problem, c_SE, c_FE)
    return {
        "beta_SE": beta if cost.beta_SE is None else float(cost.beta_SE),
        "c_SE": c_SE,
        "c_S": float(cost.c_S),
        "beta_FE": wls*beta if cost.beta_FE is None else float(cost.beta_FE),
        "c_FE": c_FE,
    }
