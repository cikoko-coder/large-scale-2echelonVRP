"""Tests for the twoe package: MILP adapter backends and run configuration."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twoe import solver as S  # noqa: E402
from twoe.config import CostConfig, RunConfig, resolve_costs  # noqa: E402


def _available(backend):
    mod = {"highs": "highspy", "scipy": "scipy.optimize", "gurobi": "gurobipy"}[backend]
    try:
        __import__(mod)
        return True
    except Exception:
        return False


BACKENDS = [pytest.param(b, marks=pytest.mark.skipif(not _available(b), reason=f"{b} not installed"))
            for b in ("highs", "scipy", "gurobi")]


# ---- expressions -------------------------------------------------------- #
def test_linexpr_arithmetic():
    m = S.MILP("e", backend="scipy" if _available("scipy") else None)
    x, y = m.add_var(name="x"), m.add_var(name="y")
    e = 2 * x + y - 3 + x
    assert e.coefs == {x.idx: 3.0, y.idx: 1.0} and e.const == -3.0
    e2 = 5 - x
    assert e2.coefs == {x.idx: -1.0} and e2.const == 5.0
    q = S.quicksum([x, 2 * y, 4])
    assert q.coefs == {x.idx: 1.0, y.idx: 2.0} and q.const == 4.0


def test_constraint_bounds():
    m = S.MILP("c", backend="scipy" if _available("scipy") else None)
    x = m.add_var()
    assert (x + 2 <= 7).bounds() == (-S.INF, 5.0)
    assert (x >= 3).bounds() == (3.0, S.INF)
    assert (2 * x == 4).bounds() == (4.0, 4.0)
    with pytest.raises(TypeError):
        m.add_constr(x + 1)


def test_binary_bounds_clamped():
    m = S.MILP("b", backend="scipy" if _available("scipy") else None)
    b = m.add_var(lb=-5, ub=9, vtype="B")
    assert (b.lb, b.ub) == (0.0, 1.0)


# ---- solving ------------------------------------------------------------- #
@pytest.mark.parametrize("backend", BACKENDS)
def test_small_milp_optimal(backend):
    m = S.MILP("t", backend=backend)
    x = m.add_var(vtype="B")
    y = m.add_var(lb=0, ub=10)
    m.add_constr(2 * x + y <= 7)
    m.set_objective(-3 * x - y)
    m.set_start(x, 1.0)
    assert m.optimize() == S.OPTIMAL
    assert m.obj_value == pytest.approx(-8.0)
    assert m.value(x) == pytest.approx(1.0) and m.value(y) == pytest.approx(5.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_maximize(backend):
    m = S.MILP("max", backend=backend)
    z = m.add_var(vtype="I", ub=10)
    m.add_constr(3 * z <= 20)
    m.set_objective(z, minimize=False)
    assert m.optimize() == S.OPTIMAL
    assert m.obj_value == pytest.approx(6.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_infeasible(backend):
    m = S.MILP("inf", backend=backend)
    z = m.add_var(vtype="I", ub=3)
    m.add_constr(z >= 5)
    assert m.optimize() == S.INFEASIBLE
    assert not m.has_solution
    with pytest.raises(RuntimeError):
        m.value(z)


@pytest.mark.parametrize("backend", BACKENDS)
def test_assignment_problem(backend):
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    m = S.MILP("assign", backend=backend)
    x = {(i, j): m.add_var(vtype="B") for i in range(3) for j in range(3)}
    for i in range(3):
        m.add_constr(S.quicksum(x[i, j] for j in range(3)) == 1)
        m.add_constr(S.quicksum(x[j, i] for j in range(3)) == 1)
    m.set_objective(S.quicksum(cost[i][j] * x[i, j] for i in range(3) for j in range(3)))
    assert m.optimize() == S.OPTIMAL
    assert m.obj_value == pytest.approx(5.0)
    assert m.stats.n_vars == 9 and m.stats.n_constrs == 6


# ---- config -------------------------------------------------------------- #
def test_runconfig_roundtrip(tmp_path):
    cfg = RunConfig(instances=["r201.txt"], tag="x")
    cfg.model.synchronized = False
    cfg.cost.c_SE = 2.5
    p = tmp_path / "cfg.json"
    cfg.save(str(p))
    loaded = RunConfig.load(str(p))
    assert loaded == cfg
    assert json.loads(p.read_text())["model"]["synchronized"] is False


def test_resolve_costs_defaults_to_shift_length():
    problem = SimpleNamespace(depot=SimpleNamespace(start_interval=0, end_interval=480),
                              CDC=SimpleNamespace(start_interval=60, end_interval=1000))
    c = resolve_costs(problem, CostConfig())
    assert c["beta_SE"] == 480.0 and c["beta_FE"] == 940.0
    c = resolve_costs(problem, CostConfig(beta_SE=10, beta_FE=20))
    assert c["beta_SE"] == 10.0 and c["beta_FE"] == 20.0
