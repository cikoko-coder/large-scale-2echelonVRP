"""The refill DP against the street MIP on a Solomon benchmark instance (needs gurobipy)."""
import io
import contextlib
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("gurobipy")
import instanceGenerator  # noqa: E402
import MathModels  # noqa: E402
import Repair_heuristcs as RH  # noqa: E402


@pytest.fixture(scope="module")
def problem():
    with contextlib.redirect_stdout(io.StringIO()):
        p = instanceGenerator.readGenerate_benchmark("c208.txt", 100, unbounded_vehicle_time=True)
    p.tabu_search_active = False
    gap = MathModels.MIP_GAP
    MathModels.MIP_GAP = 0.0  # exact MIP plans: the DP is exact, a gap would make the costs differ
    yield p
    MathModels.MIP_GAP = gap


def test_dp_equals_street_mip_on_random_subsequences(problem):
    """Same feasibility, same transfer cost and the same number of transshipments as the MIP."""
    rng = random.Random(3)
    customers = sorted(problem.customers, key=lambda node: node.start_interval)
    feasible = 0
    for _ in range(40):
        size = rng.randint(2, 14)
        start = rng.randint(0, len(customers) - size)
        route = customers[start:start + size]
        if rng.random() < 0.3:  # perturb the time order a little
            i = rng.randint(0, size - 2)
            route[i], route[i + 1] = route[i + 1], route[i]
        dp = RH.optimal_transshipment_checking(problem, route)
        mip_feasible, mip_cost, mip_visits, _ = MathModels.street_problem_MIP(problem, 50, route)
        assert dp["feasible"] == mip_feasible, [n.index for n in route]
        if mip_feasible:
            feasible += 1
            assert dp["cost"] - problem.cost_vehicle_use == pytest.approx(mip_cost, abs=1e-3)
            assert len(dp["refills"]) == len(mip_visits)
    assert feasible >= 10


def test_route_needing_an_extra_transshipment(problem):
    """c208 construction route: m = 4 trips is late, the MIP and the bounded DP use 5."""
    seq = [27, 30, 100, 99, 22, 24, 29, 6, 31, 32, 33, 37, 38, 39, 34, 36, 28, 26, 23, 12, 14, 19,
           16, 15, 17, 13, 25, 11, 9, 10, 8, 48]
    route = [problem.nodes[i] for i in seq]
    minimum = RH.bothway_optimal_checking(problem, route)
    bounded = RH.optimal_transshipment_checking(problem, route)
    assert not minimum["feasible"] and minimum["min_trips"] == 4
    assert bounded["feasible"] and bounded["n_trips"] == 5
    assert "no position can take a refill on time" in bounded["attempts"][4]
    assert minimum["blocked"][1] in (32, 33)
    mip_feasible, mip_cost, mip_visits, _ = MathModels.street_problem_MIP(problem, 50, route)
    assert mip_feasible and len(mip_visits) == 5
    assert bounded["cost"] - problem.cost_vehicle_use == pytest.approx(mip_cost, abs=1e-3)
    assert RH.is_route_feasible(problem, route)[0]


def test_free_routing_mip_routes_are_dp_feasible(problem):
    """street_routing_MIP on 10 customers: its routes pass the DP with the same cost; a
    warm start from those routes is accepted; the fixed-route entry refuses no route."""
    customers = sorted(problem.customers, key=lambda node: node.index)[:10]
    res = MathModels.street_routing_MIP(problem, 60, customers=customers)
    assert res["feasible"] and res["gap"] <= 1e-4
    assert sorted(node.index for route in res["routes"] for node in route) == [c.index for c in customers]
    total = 0.0
    for route in res["routes"]:
        dp = RH.optimal_transshipment_checking(problem, route)
        assert dp["feasible"]
        total += dp["cost"]
    assert total == pytest.approx(res["objective"], abs=1e-3)
    assert res["objective"] == pytest.approx(res["fixed_vehicle_cost"] + res["distance_travel_cost"]
                                             + res["lead_time_cost"] + res["transshipment_cost"], abs=1e-3)
    warm = MathModels.street_routing_MIP(problem, 60, customers=customers, warm_start=res["routes"])
    assert warm["feasible"] and warm["objective"] <= res["objective"] + 1e-3
    with pytest.raises(ValueError):
        MathModels.street_problem_MIP(problem, 10)


def test_is_route_feasible_uses_the_two_way_rule():
    """is_route_feasible (REFILL_RULE "two_way"): feasibility of the DP, cost of the plan with the
    minimum number of transshipments K*; paper costs (no trip cost), so the cheapest plan may use more."""
    assert RH.REFILL_RULE == "two_way"
    with contextlib.redirect_stdout(io.StringIO()):
        p = instanceGenerator.readGenerate_benchmark("rc201.txt", 100, unbounded_vehicle_time=True, cost_setting="paper")
    p.tabu_search_active = False
    rng = random.Random(5)
    customers = sorted(p.customers, key=lambda node: node.start_interval)
    checked = 0
    for _ in range(400):
        size = rng.randint(2, 15)
        start = rng.randint(0, len(customers) - size)
        route = customers[start:start + size]
        if RH.get_makespan(p, route) > 1:
            continue
        two_way = RH.two_way_transshipment_checking(p, route)
        feasible, _, cost = RH.is_route_feasible(p, route)
        assert feasible == two_way["feasible"]
        if feasible:
            checked += 1
            assert cost == pytest.approx(two_way["cost"] - p.cost_vehicle_use, abs=1e-6)
            cheapest = RH.optimal_transshipment_checking(p, route)
            assert cheapest["n_trips"] >= two_way["n_trips"] and cheapest["cost"] <= two_way["cost"] + 1e-6
    assert checked >= 20
