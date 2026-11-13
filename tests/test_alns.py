"""ALNS (alns.py: destroy / repair, two-way check, simulated annealing) on a Solomon construction."""
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
import alns as ALNS  # noqa: E402


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


def all_customers(p):
    return sorted(n.index for n in p.customers)


def test_improves_and_keeps_every_customer(start):
    p, routes = start
    before = RH.two_way_solution_cost(p, routes)
    copy = indices(routes)
    out, report = ALNS.alns(p, routes, seed=1, iterations=40, time_limit=None, store=False)
    after = RH.two_way_solution_cost(p, out)
    assert sorted(i for r in indices(out) for i in r) == all_customers(p)
    assert after["feasible"]
    assert after["cost"] == pytest.approx(report["final_cost"])
    assert before["cost"] == pytest.approx(report["start_cost"])
    assert report["final_cost"] < report["start_cost"]
    assert report["iterations"] == 40
    assert indices(routes) == copy  # the start is not modified
    costs = [c for c, _ in report["elite"]]
    assert costs == sorted(costs) and costs[0] == pytest.approx(report["final_cost"])
    assert len(set(tuple(map(tuple, sorted(indices(r)))) for _, r in report["elite"])) == len(costs)


def test_same_seed_same_search(start):
    p, routes = start
    a, ra = ALNS.alns(p, routes, seed=7, iterations=15, time_limit=None, store=False)
    b, rb = ALNS.alns(p, routes, seed=7, iterations=15, time_limit=None, store=False)
    assert indices(a) == indices(b) and ra["final_cost"] == rb["final_cost"]


@pytest.mark.parametrize("destroy", ALNS.DESTROY)
def test_destroy_operators(start, destroy):
    p, routes = start
    search = ALNS.ALNS(p, routes, seed=3)
    rds = [search.route_data(r) for r in routes]
    q = 12
    partial, removed = search.destroy(destroy, rds, q)
    ids = [u.index for u in removed]
    assert len(ids) == len(set(ids))
    assert len(ids) >= q if destroy == "route" else len(ids) == q
    assert sorted(ids + [u.index for r in partial for u in r]) == all_customers(p)


@pytest.mark.parametrize("repair", ALNS.REPAIR)
def test_repair_operators(start, repair):
    p, routes = start
    search = ALNS.ALNS(p, routes, seed=4)
    rds = [search.route_data(r) for r in routes]
    out = search.neighbour(rds, "related", repair)
    assert out is not None
    assert sorted(u.index for rd in out for u in rd.nodes) == all_customers(p)
    check = RH.two_way_solution_cost(p, [rd.nodes for rd in out])
    assert check["feasible"] and check["cost"] == pytest.approx(search.cost(out))


def test_alns_improve_stores_the_elite(start):
    p, routes = start
    with contextlib.redirect_stdout(io.StringIO()):
        solution = RH.check_feasibility(p, routes, VRP_only=False, multi_trip=True, single_route_MVRP=False)
    p.heuristics_solutions = []
    with contextlib.redirect_stdout(io.StringIO()):
        improved, report = MLD.alns_improve(p, solution, seed=2, iterations=10, time_limit=None)
    assert improved.is_feasible and improved.two_way_cost == pytest.approx(report["final_cost"])
    assert 1 <= len(p.heuristics_solutions) <= ALNS.ALNS_ELITE
    best = sorted(sorted(r) for r in indices([list(r.sequence) for r in improved.routes_sevs]))
    assert best in [sorted(sorted(r) for r in s) for s in p.heuristics_solutions]


def test_satellite_detour_memo(start):
    p, _ = start
    t = p.travel_matrix
    for i, j in [(0, 5), (5, 17), (40, 3), (99, 0)]:
        direct = min((t[i][s.index] + t[s.index][j] - t[i][j], s.index) for s in p.satellites)
        assert RH.satellite_detour(p, i, j) == direct
        assert RH.satellite_detour(p, i, j) == direct  # memoized
