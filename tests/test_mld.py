"""MLD destroy, the single repair and the ALNS of Multi_level_destroy (manuscript Section 5.5.3)."""
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


@pytest.fixture(scope="module")
def start():
    with contextlib.redirect_stdout(io.StringIO()):
        p = instanceGenerator.readGenerate_benchmark("c208.txt", 100, unbounded_vehicle_time=True)
    p.tabu_search_active = False
    p.synch_check = True
    p.modelStartTime = time.time()
    p.points_found = []
    p.min_relaxed_transfer_num = 0
    p.customers = [node for node in p.nodes if node.index > 0]
    p.its_iter = 1
    random.seed(33)
    with contextlib.redirect_stdout(io.StringIO()):
        solution = MLD.multi_start_initial(p, 1)
    assert solution.is_feasible
    return p, [list(route.sequence) for route in solution.routes_sevs]


def ids(nodes):
    return sorted(n.index for n in nodes)


def test_initial_scores_fall_with_the_its_iteration(start):
    p, _ = start
    for k in (1, 2, 5):
        p.its_iter = k
        MLD.generate_operators_statistics(p)
        assert p.scores_roulette_selection_type == [10/k, 1.0]   # S_I, S_W
        assert p.scores_removal_type == [MLD.ROUTE_REMOVAL_INITIAL_SCORE/k, 1.0]   # S_R, S_N
        assert p.operators_removal_number == [0.05, 0.10, 0.25, 0.35]
        assert p.scores_removal_number == [1.0]*4
        assert p.operators_repair_type == ["k_best_insertion"]   # single repair
    p.its_iter = 1


@pytest.mark.parametrize("roulette", ["equal", "weighted"])
def test_route_removal(start, roulette):
    p, routes = start
    random.seed(1)
    partial, pool = MLD.multi_direction_random_destroy_repair(p, routes, roulette, "route", 0.10)
    removed_routes = [r for r in routes if set(n.index for n in r) <= set(n.index for n in pool)]
    assert removed_routes, "a whole route is removed first"
    seed = removed_routes[0]
    assert [n.index for n in pool[:len(seed)]] == [n.index for n in seed]   # removal order kept
    remaining = len(p.customers) - len(seed)
    assert len(pool) - len(seed) == max(int(remaining*0.10), 1)
    assert ids(pool + [n for r in partial for n in r]) == ids(p.customers)
    assert len(partial) == len(routes) - 1


@pytest.mark.parametrize("roulette", ["equal", "weighted"])
def test_node_removal(start, roulette):
    p, routes = start
    random.seed(2)
    partial, pool = MLD.multi_direction_random_destroy_repair(p, routes, roulette, "node", 0.25)
    assert len(pool) == 1 + int((len(p.customers) - 1)*0.25)
    assert len(set(n.index for n in pool)) == len(pool)
    assert ids(pool + [n for r in partial for n in r]) == ids(p.customers)


def test_single_repair_completes_the_schedule(start):
    p, routes = start
    random.seed(3)
    partial, pool = MLD.multi_direction_random_destroy_repair(p, routes, "weighted", "node", 0.35)
    with contextlib.redirect_stdout(io.StringIO()):
        repaired, left = RH.k_best_insertion_repair(p, partial, pool)
    assert not left
    assert ids([n for r in repaired for n in r]) == ids(p.customers)
    assert RH.two_way_solution_cost(p, repaired)["feasible"]


def test_score_and_acceptance():
    assert MLD.calculate_score(100.0, 100.5) == MLD.SCORE_SAME
    assert MLD.calculate_score(100.0, 90.0) == MLD.SCORE_IMPROVING
    assert MLD.calculate_score(100.0, 110.0) == MLD.SCORE_WORSE
    assert MLD.simulated_annealing_acceptance(90.0, 100.0, 1e-9)
    random.seed(0)
    accepted = sum(MLD.simulated_annealing_acceptance(110.0, 100.0, 10.0) for _ in range(4000))/4000
    assert accepted == pytest.approx(0.3679, abs=0.03)   # exp(-10/10)


def test_alns_never_returns_a_worse_schedule(start):
    p, routes = start
    random.seed(4)
    key0, _ = MLD.solution_key(p, routes)
    with contextlib.redirect_stdout(io.StringIO()):
        best, report = MLD.Solve_by_multi_level_ALNS(p, routes, 20)
    key, two_way = MLD.solution_key(p, best)
    assert two_way["feasible"] and ids([n for r in best for n in r]) == ids(p.customers)
    assert key <= key0 and report["best_key"] == pytest.approx(key)
    assert report["iterations"] == 20
    assert sum(report["used"]["removal"].values()) == 20
    assert report["temperature"] == pytest.approx(MLD.SA_ACCEPTANCE*MLD.travel_cost(p, RH.two_way_solution_cost(p, routes)))


def test_found_solutions_are_stored_once(start):
    p, routes = start
    p.heuristics_solutions = []
    assert RH.add_to_feasible_solution(p, routes)
    assert not RH.add_to_feasible_solution(p, list(reversed(routes)))      # same set, other route order
    swapped = [list(r) for r in routes]
    swapped[0][0], swapped[0][1] = swapped[0][1], swapped[0][0]
    assert RH.add_to_feasible_solution(p, swapped)                         # other customer order is new
    assert not RH.add_to_feasible_solution(p, routes[:-1])                 # partial: not stored
    assert len(p.heuristics_solutions) == 2
    first = p.heuristics_solutions[0]
    p.heuristics_solutions = [first]                                       # list replaced elsewhere
    assert not RH.add_to_feasible_solution(p, routes)
    assert RH.add_to_feasible_solution(p, swapped) and len(p.heuristics_solutions) == 2


def test_route_mip_check_is_memoized(start, monkeypatch):
    import MathModels
    p, routes = start
    p._route_mip_cache = {}
    calls = []
    original = MathModels.street_problem_MIP
    monkeypatch.setattr(MathModels, "street_problem_MIP", lambda *a, **k: calls.append(1) or original(*a, **k))
    route = routes[0]
    first = RH.route_mip_check(p, route)
    first[2][0].append(999)                                                # callers append loads to visits
    second = RH.route_mip_check(p, route)
    assert len(calls) == 1 and first[0] and second[0]
    assert first[1] == second[1] and all(len(v) == 2 for v in second[2])
    two_way = RH.two_way_transshipment_checking(p, route)
    assert second[1] <= two_way["cost"] - p.cost_vehicle_use + 1e-3        # MILP: min cost, any number of trips


def test_optimize_fe_takes_only_the_fewest_sevs(start):
    p, routes = start
    random.seed(5)
    p.heuristics_solutions, p.checked_solutions, p.feas_solutions, p.inf_solutions = [], [], [], []
    p.best_found, p.cutoff = None, float("inf")
    RH.add_to_feasible_solution(p, routes)
    split = [list(r) for r in routes]
    split.append([split[0].pop()])                                         # one SEV more
    RH.add_to_feasible_solution(p, split)
    with contextlib.redirect_stdout(io.StringIO()):
        best, n = MLD.Optimize_FE(p)
    assert n == 1 and len(p.checked_solutions) == 1 and len(p.checked_solutions[0]) == len(routes)
    assert best is not None and best.is_feasible and best.nSEVs == len(routes)
    assert RH.ROUTE_CHECK == "dp"                                          # restored after the FE stage


def test_incumbent_is_rechecked_by_the_fixed_route_milp(start, monkeypatch):
    p, routes = start
    random.seed(6)
    monkeypatch.setattr(RH, "INCUMBENT_ROUTE_MIP", True)
    p.heuristics_solutions, p.checked_solutions, p.feas_solutions, p.inf_solutions = [], [], [], []
    p.best_found, p.cutoff, p.incumbent_refinements = None, float("inf"), []
    RH.add_to_feasible_solution(p, routes)
    with contextlib.redirect_stdout(io.StringIO()):
        best, _ = MLD.Optimize_FE(p, None, "dp")
    assert best is not None and best.is_feasible
    assert p.incumbent_refinements, "every incumbent update is re-checked"
    rec = p.incumbent_refinements[0]
    assert rec["refined_trips"] is not None and rec["refined_trips"] >= len(routes)
    # the re-check never makes the incumbent worse and leaves the route check as it was
    assert best.objective <= rec["objective"] + 1e-6
    assert RH.ROUTE_CHECK == "dp" and not getattr(p, "_refining_incumbent", False)


def test_incumbent_follows_the_hierarchical_objective(start):
    """FEVs, then SEVs, then total distance (trip charge excluded): a weighted-sum loss does not
    matter when a higher level wins."""
    p, _ = start
    import types
    def sol(fevs, sevs, street, water, trips):
        return types.SimpleNamespace(nFEVs=float(fevs), nSEVs=sevs, travelStreet=street + p.trip_unit_cost*trips,
                                     travelWater=water, transshipments=[None]*trips)
    base = sol(2, 3, 900.0, 500.0, 12)
    assert RH.hierarchical_key(p, base) == (2, 3, 1400.0)
    assert RH.is_better_solution(p, sol(1, 4, 2000.0, 900.0, 20), base)     # fewer FEVs wins
    assert RH.is_better_solution(p, sol(2, 2, 1500.0, 600.0, 18), base)     # same FEVs, fewer SEVs
    assert RH.is_better_solution(p, sol(2, 3, 850.0, 500.0, 15), base)      # more trips, less distance
    assert not RH.is_better_solution(p, sol(2, 3, 900.0, 500.0, 10), base)  # equal distance: not better
    assert not RH.is_better_solution(p, sol(3, 2, 100.0, 100.0, 5), base)
