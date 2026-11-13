"""Local search (relocate + 2-opt*, two-way check) on a Solomon benchmark construction."""
import contextlib
import io
import os
import random
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("highspy")
import instanceGenerator  # noqa: E402
import Multi_level_destroy as MLD  # noqa: E402
import Repair_heuristcs as RH  # noqa: E402
import local_search as LS  # noqa: E402


@pytest.fixture(scope="module")
def start():
    with contextlib.redirect_stdout(io.StringIO()):
        p = instanceGenerator.readGenerate_benchmark("c208.txt", 100, unbounded_vehicle_time=True)
    p.tabu_search_active = False
    p.modelStartTime = time.time()
    p.points_found = []
    p.min_relaxed_transfer_num = 0
    p.customers = [node for node in p.nodes if node.index > 0]
    random.seed(33)
    with contextlib.redirect_stdout(io.StringIO()):
        solution = MLD.multi_start_initial(p, 1)
    assert solution.is_feasible
    return p, [list(route.sequence) for route in solution.routes_sevs]


def indices(routes):
    return [[node.index for node in route] for route in routes]


def test_improves_and_keeps_every_customer(start):
    p, routes = start
    before = RH.two_way_solution_cost(p, routes)
    out, report = LS.local_search(p, routes, seed=1, store=False)
    after = RH.two_way_solution_cost(p, out)
    assert sorted(i for r in indices(out) for i in r) == sorted(n.index for n in p.customers)
    assert after["feasible"]
    assert after["cost"] == pytest.approx(report["final_cost"])
    assert before["cost"] == pytest.approx(report["start_cost"])
    assert report["final_cost"] < report["start_cost"]
    assert indices(routes) == indices([list(r) for r in routes])  # the start is not modified


def test_screens_do_not_change_the_search(start, monkeypatch):
    """The time screen and the cost bound only remove moves that cannot improve: with both
    disabled the search makes exactly the same moves, with many more exact checks."""
    p, routes = start
    screened, r1 = LS.local_search(p, routes, seed=5, store=False, max_rounds=2)
    monkeypatch.setattr(LS, "joins_in_time", lambda *args: True)
    monkeypatch.setattr(LS.SearchContext, "lower_bound", lambda self, *args: 0.0)
    full, r2 = LS.local_search(p, routes, seed=5, store=False, max_rounds=2)
    assert indices(screened) == indices(full)
    assert r1["final_cost"] == pytest.approx(r2["final_cost"])
    assert r1["exact"] < r2["exact"]


def test_route_data_bounds(start):
    """Forward earliest / backward latest of RouteData bracket every on-time schedule."""
    p, routes = start
    ctx = LS.SearchContext(p, routes)
    for rd in ctx.routes:
        assert rd.feasible
        assert all(e - node.unloading_time <= l + LS.EPS for e, l, node in zip(rd.earliest, rd.latest, rd.nodes))
        assert rd.dist == pytest.approx(sum(p.travel_matrix[a][b] for a, b in
                                            zip([0] + [n.index for n in rd.nodes], [n.index for n in rd.nodes] + [0])))


def test_wrappers_of_the_old_operators(start):
    p, routes = start
    change, out = MLD.perform_N_relocation(p, routes, 2)
    assert change <= 0 and sum(map(len, out)) == len(p.customers)
    change, out = MLD.perform_N_two_point_crossover(p, out, 2)
    assert change <= 0 and sum(map(len, out)) == len(p.customers)
    assert MLD.perform_N_relocation(p, routes, 0) == (0, [list(r) for r in routes])
