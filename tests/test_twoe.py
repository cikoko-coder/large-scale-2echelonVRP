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
def test_hierarchical_objectives(backend):
    """Lexicographic: first level wins whatever the size of the second level's coefficients."""
    m = S.MILP("lex", backend=backend)
    a = m.add_var(vtype="B", name="a")  # cheap in level 1, very expensive in level 2
    b = m.add_var(vtype="B", name="b")
    c = m.add_var(vtype="B", name="c")
    m.add_constr(a + b + c >= 1)
    m.add_constr(b + c >= 1)
    m.set_objectives([S.quicksum([a, b, c]), 1000 * b + 1 * c])
    m.time_limit = 30
    assert m.optimize() == S.OPTIMAL
    assert m.value(c) == pytest.approx(1.0) and m.value(a) + m.value(b) == pytest.approx(0.0)
    assert m.obj_values == pytest.approx([1.0, 1.0])
    assert m.num_constrs == 2  # the level constraints are removed after the solve


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


def test_resolve_costs_defaults_to_the_paper_fixed_cost():
    """eq. (62): beta = max_{i,p} c^s (t_gi + t_ip + t_pg) + c^w (t_dp + t_pd); beta^w = alpha beta."""
    # nodes: 0 garage, 1-2 customers, 3 satellite, 4 CDC
    t = [[0, 10, 20, 5, 99], [10, 0, 15, 8, 99], [20, 15, 0, 30, 99], [5, 8, 30, 0, 7], [99, 99, 99, 7, 0]]
    problem = SimpleNamespace(travel_matrix=t, depot=SimpleNamespace(index=0), CDC=SimpleNamespace(index=4),
                              customers=[SimpleNamespace(index=1), SimpleNamespace(index=2)],
                              satellites=[SimpleNamespace(index=3)])
    c = resolve_costs(problem, CostConfig(), wls=0.1)
    beta = 1.0*(20 + 30 + 5) + 0.1*(7 + 7)  # customer 2 is the most expensive
    assert c["beta_SE"] == pytest.approx(beta) and c["beta_FE"] == pytest.approx(0.1*beta)
    assert c["c_SE"] == 1.0 and c["c_FE"] == pytest.approx(0.1)
    c = resolve_costs(problem, CostConfig(beta_SE=10, beta_FE=20, c_FE=2.0))
    assert c["beta_SE"] == 10.0 and c["beta_FE"] == 20.0 and c["c_FE"] == 2.0


# ---- solver choice ------------------------------------------------------- #
def test_default_backend_is_free_and_selectable(monkeypatch):
    monkeypatch.delenv("MIP_SOLVER", raising=False)
    S.set_default_backend(None)
    if _available("highs"):
        assert S.MILP("d").backend == "highs"
    monkeypatch.setenv("MIP_SOLVER", "scipy")
    assert S.MILP("e").backend == "scipy"
    S.set_default_backend("gurobi")  # explicit setting wins over the environment variable
    try:
        assert S.MILP("f").backend == "gurobi"
        assert S.MILP("g", backend="highs").backend == "highs"  # per-model choice wins over both
    finally:
        S.set_default_backend(None)
    with pytest.raises(ValueError):
        S.set_default_backend("cplex")
    monkeypatch.setenv("MIP_SOLVER", "nope")
    with pytest.raises(ValueError):
        S.MILP("h")


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_params_and_solve_listener(backend):
    seen = []
    listener = S.add_solve_listener(lambda model: seen.append((model.name, model.status)))
    try:
        m = S.MILP("listen", backend=backend)
        x = m.add_var(vtype="I", ub=4)
        m.add_constr(x <= 2.5)
        m.set_objective(x, minimize=False)
        m.params["gurobi"]["Heuristics"] = 0.5        # each backend reads only its own entry
        m.params["highs"]["mip_heuristic_effort"] = 0.3
        assert m.optimize() == S.OPTIMAL and m.obj_value == pytest.approx(2.0)
    finally:
        S.remove_solve_listener(listener)
    assert seen == [("listen", S.OPTIMAL)]


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "scipy"])
def test_partial_warm_start(backend):
    m = S.MILP("ws", backend=backend)
    x = [m.add_var(vtype="B") for _ in range(4)]
    m.add_constr(S.quicksum(x) == 2)
    m.set_objective(S.quicksum((k + 1) * v for k, v in enumerate(x)))
    m.set_start(x[3], 1.0)  # only one value given; the solver completes the start
    assert m.optimize() == S.OPTIMAL and m.obj_value == pytest.approx(3.0)
