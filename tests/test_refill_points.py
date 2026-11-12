"""Tests for the refill (transfer point) rules in Repair_heuristcs."""
import math
import os
import random
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("gurobipy")
import Repair_heuristcs as RH  # noqa: E402


def route_of(demands):
    return [NS(index=i + 1, demand=d) for i, d in enumerate(demands)]


def positions(route, refills):
    where = {node.index: pos for pos, node in enumerate(route)}
    return [where[i] for i in refills]


def trip_loads(route, refills):
    cuts = positions(route, refills) + [len(route)]
    return [sum(n.demand for n in route[cuts[k]:cuts[k + 1]]) for k in range(len(cuts) - 1)]


def test_example_from_review():
    route = route_of([1, 3, 3, 3, 1, 4, 2])
    problem = NS(capacity_SEV=5)
    assert positions(route, RH.forward_checking(problem, route)) == [0, 2, 3, 5, 6]
    assert positions(route, RH.backward_checking(problem, route)) == [0, 2, 3, 4, 6]
    assert trip_loads(route, RH.backward_checking(problem, route)) == [4, 3, 3, 5, 2]


def test_single_trip_only_first_node():
    route = route_of([2, 1, 1])
    problem = NS(capacity_SEV=5)
    assert RH.forward_checking(problem, route) == [1]
    assert RH.backward_checking(problem, route) == [1]
    assert RH.bothway_checking(problem, route) == ([1], [1])


def test_random_routes():
    random.seed(7)
    for _ in range(3000):
        q = random.randint(5, 15)
        route = route_of([random.randint(1, q) for _ in range(random.randint(1, 25))])
        problem = NS(capacity_SEV=q)
        fwd = RH.forward_checking(problem, route)
        bwd = RH.backward_checking(problem, route)
        assert RH.bothway_checking(problem, route) == (fwd, bwd)
        fp, bp = positions(route, fwd), positions(route, bwd)
        assert fp == sorted(fp) and bp == sorted(bp) and fp[0] == bp[0] == 0
        assert max(trip_loads(route, fwd)) <= q and max(trip_loads(route, bwd)) <= q
        # both greedy splits give the minimum number of trips for this order
        assert len(fwd) == len(bwd)
        assert len(fwd) >= math.ceil(sum(n.demand for n in route) / q)
        # backward refills are never later than the forward ones (trip by trip)
        assert all(b <= f for b, f in zip(bp, fp))


# ---- bothway_optimal_checking: DP between the backward and forward bounds ---- #
def random_instance(rng, n):
    """Small Euclidean instance: garage 0, customers 1..n, satellites n+1..n+3."""
    pts = [(rng.uniform(0, 50), rng.uniform(0, 50)) for _ in range(n + 4)]
    t = [[math.dist(a, b) for b in pts] for a in pts]
    q = rng.randint(6, 12)
    customers, clock = [], 0.0
    for i in range(1, n + 1):
        clock += rng.uniform(5, 30)
        width = rng.uniform(10, 80)
        customers.append(NS(index=i, demand=rng.randint(1, q), start_interval=clock - width / 2,
                            end_interval=clock + width / 2, unloading_time=rng.uniform(0, 5)))
    problem = NS(capacity_SEV=q, travel_matrix=t, satellites=[NS(index=n + 1 + k) for k in range(3)],
                 depot=NS(start_interval=-200.0), cost_unit_travel=1.0, trip_unit_cost=rng.choice([1.0, 30.0, 10000.0]),
                 cost_vehicle_use=100.0, TransferDuration=rng.choice([0.0, 3.0]))
    return problem, customers


def brute_force(problem, route, n_refills=None):
    """Every set of refill positions (position 0 always refills); n_refills fixes their number."""
    import itertools
    t, n = problem.travel_matrix, len(route)
    prev = [0] + [node.index for node in route[:-1]]
    detour = [max(0.0, min(t[prev[k]][p.index] + t[p.index][route[k].index] - t[prev[k]][route[k].index]
                           for p in problem.satellites)) for k in range(n)]
    best = None
    for mask in itertools.product([0, 1], repeat=n - 1):
        cuts = [0] + [k + 1 for k, b in enumerate(mask) if b]
        if n_refills is not None and len(cuts) != n_refills:
            continue
        bounds = cuts + [n]
        if any(sum(nd.demand for nd in route[bounds[k]:bounds[k + 1]]) > problem.capacity_SEV for k in range(len(cuts))):
            continue
        clock, cost, ok = problem.depot.start_interval, 0.0, True
        for k, node in enumerate(route):
            clock += t[prev[k]][node.index]
            if k in cuts:
                clock += detour[k] + problem.TransferDuration
                cost += detour[k] + problem.trip_unit_cost
            if clock > node.end_interval:
                ok = False
                break
            clock = max(clock, node.start_interval) + node.unloading_time
        if ok and (best is None or cost < best - 1e-9):
            best = cost
    return best


def test_cost_optimal_dp_matches_brute_force():
    """optimal_transshipment_checking: any number of transshipments, cheapest plan."""
    rng = random.Random(11)
    checked = feasible = 0
    for _ in range(400):
        n = rng.randint(1, 11)
        problem, route = random_instance(rng, n)
        res = RH.optimal_transshipment_checking(problem, route)
        bf = brute_force(problem, route)
        assert res["feasible"] == (bf is not None)
        if bf is not None:
            feasible += 1
            assert res["transfer_cost"] == pytest.approx(bf, abs=1e-6)
            refills = positions(route, res["refills"])
            assert refills[0] == 0 and max(trip_loads(route, res["refills"])) <= problem.capacity_SEV
        checked += 1
    assert feasible > 50 and checked - feasible > 50


def test_windows_contain_greedy_bounds():
    rng = random.Random(5)
    for _ in range(500):
        q = rng.randint(5, 15)
        route = route_of([rng.randint(1, q) for _ in range(rng.randint(2, 25))])
        problem = NS(capacity_SEV=q)
        fwd = positions(route, RH.forward_checking(problem, route))
        bwd = positions(route, RH.backward_checking(problem, route))
        lo, hi = RH.refill_windows(problem, route, len(fwd))
        assert lo == bwd and hi == fwd
        for extra in (1, 2):
            lo2, hi2 = RH.refill_windows(problem, route, min(len(route), len(fwd) + extra))
            assert all(a <= b for a, b in zip(lo2, hi2))


def test_minimum_transshipment_dp_matches_brute_force():
    """bothway_optimal_checking: exactly the minimum number of transshipments m."""
    rng = random.Random(13)
    feasible = infeasible = 0
    for _ in range(400):
        n = rng.randint(1, 11)
        problem, route = random_instance(rng, n)
        m = len(RH.forward_checking(problem, route))
        res = RH.bothway_optimal_checking(problem, route)
        bf = brute_force(problem, route, n_refills=m)
        assert res["feasible"] == (bf is not None)
        if bf is not None:
            feasible += 1
            assert res["n_trips"] == m
            assert res["transfer_cost"] == pytest.approx(bf, abs=1e-6)
        else:
            infeasible += 1
    assert feasible > 50 and infeasible > 50


# ---- bounded DP: minimum m is the starting bound, fast feasibility pass, pruned windows ---- #
def test_fastest_refill_plan_matches_brute_force_per_trip_count():
    rng = random.Random(17)
    agree = feasible = 0
    for _ in range(400):
        n = rng.randint(1, 10)
        problem, route = random_instance(rng, n)
        detours = RH.refill_detours(problem, route)
        bounds = RH.refill_time_bounds(problem, route, detours)
        m = len(RH.forward_checking(problem, route))
        for K in range(m, min(n, m + 2) + 1):
            bf = brute_force(problem, route, n_refills=K)
            if bounds is None:
                assert bf is None
                continue
            earliest, latest, can_refill, slack = bounds
            lo, hi = RH.refill_windows(problem, route, K)
            fast = RH.fastest_refill_plan(problem, route, K, lo, hi, can_refill, detours)
            assert fast["feasible"] == (bf is not None)
            agree += 1
            if fast["feasible"]:
                feasible += 1
                refills = fast["refills"]
                assert refills[0] == 0 and len(refills) == K
                assert all(lo[k] <= refills[k] <= hi[k] for k in range(K))
                cuts = refills + [n]
                assert max(sum(nd.demand for nd in route[cuts[k]:cuts[k + 1]]) for k in range(K)) <= problem.capacity_SEV
            else:
                assert fast["blocked"] is not None
    assert feasible > 100 and agree - feasible > 50


def test_time_bounds_only_remove_impossible_refill_positions():
    """A position marked can_refill=False is never a refill in any feasible plan (brute force)."""
    import itertools
    rng = random.Random(19)
    pruned = 0
    for _ in range(300):
        n = rng.randint(2, 9)
        problem, route = random_instance(rng, n)
        bounds = RH.refill_time_bounds(problem, route)
        if bounds is None:
            assert brute_force(problem, route) is None
            continue
        earliest, latest, can_refill, slack = bounds
        assert can_refill[0]
        impossible = {pos for pos in range(n) if not can_refill[pos]}
        pruned += len(impossible)
        t = problem.travel_matrix
        prev = [0] + [node.index for node in route[:-1]]
        detour = [max(0.0, min(t[prev[k]][p.index] + t[p.index][route[k].index] - t[prev[k]][route[k].index]
                               for p in problem.satellites)) for k in range(n)]
        for mask in itertools.product([0, 1], repeat=n - 1):
            cuts = {0} | {k + 1 for k, b in enumerate(mask) if b}
            if not (cuts & impossible):
                continue
            clock, ok = problem.depot.start_interval, True
            for k, node in enumerate(route):
                clock += t[prev[k]][node.index] + ((detour[k] + problem.TransferDuration) if k in cuts else 0)
                if clock > node.end_interval:
                    ok = False
                    break
                clock = max(clock, node.start_interval) + node.unloading_time
            assert not ok, "a pruned position was used by an on-time plan"
    assert pruned > 50


def test_minimum_is_a_bound_not_a_rule():
    """The bounded DP reports min_trips = m, uses m when m is on time, m+1 otherwise."""
    rng = random.Random(23)
    for _ in range(600):
        n = rng.randint(2, 10)
        problem, route = random_instance(rng, n)
        res = RH.optimal_transshipment_checking(problem, route)
        m = len(RH.forward_checking(problem, route))
        minimum = RH.bothway_optimal_checking(problem, route)
        if res["feasible"]:
            assert res["min_trips"] == m and res["n_trips"] >= m
            if minimum["feasible"]:
                assert res["attempts"][m] == "feasible"
                if problem.trip_unit_cost >= 30.0:
                    assert res["n_trips"] == m  # a transshipment costs more than any detour saving
            else:
                assert res["n_trips"] > m
        else:
            assert not minimum["feasible"]


def test_extra_transshipment_when_the_minimum_split_is_late():
    """4 unit demands, capacity 2: m = 2 and the second refill must be before position 2, whose
    only satellite detour (100) is longer than the slack there; refills before positions 1 and 3
    have no detour, so 3 transshipments are on time and the bounded DP uses them."""
    n = 4
    # nodes: 0 garage, 1..4 customers, satellites 5 (near 0, 1, 2) and 6 (near 3, 4); travel 10
    # between consecutive route nodes, so the detours are 0 on the arcs (0,1), (1,2), (3,4) and
    # 5 + 55 - 10 = 50 on the arc (2,3), while the slack at every customer is 20
    t = [[10.0]*7 for _ in range(7)]
    for i in range(7):
        t[i][i] = 0.0
    for c in range(5):
        t[c][5] = t[5][c] = 5.0 if c <= 2 else 55.0
        t[c][6] = t[6][c] = 5.0 if c >= 3 else 55.0
    customers = [NS(index=i, demand=1, start_interval=0.0, end_interval=10.0*i + 20.0, unloading_time=0.0)
                 for i in range(1, n + 1)]
    problem = NS(capacity_SEV=2, travel_matrix=t, satellites=[NS(index=5), NS(index=6)], depot=NS(start_interval=0.0),
                 cost_unit_travel=1.0, trip_unit_cost=1000.0, cost_vehicle_use=0.0, TransferDuration=0.0)
    route = customers
    assert RH.forward_checking(problem, route) == [1, 3] and RH.backward_checking(problem, route) == [1, 3]
    minimum = RH.bothway_optimal_checking(problem, route)
    assert not minimum["feasible"] and minimum["min_trips"] == 2
    assert "customer 3 late by" in minimum["attempts"][2]
    assert minimum["blocked"][1] == 3
    bounded = RH.optimal_transshipment_checking(problem, route)
    assert bounded["feasible"] and bounded["n_trips"] == 3 and bounded["refills"] == [1, 2, 4]
    assert bounded["transfer_cost"] == pytest.approx(3000.0)
    assert brute_force(problem, route) == pytest.approx(3000.0)


def test_route_check_wrapper_has_the_mip_interface():
    rng = random.Random(29)
    for _ in range(50):
        n = rng.randint(1, 8)
        problem, route = random_instance(rng, n)
        problem.consolidated = False
        problem.tabu_search_active = True
        problem.tabu_list = []
        feasible, cost, visits, route_back = RH.check_route_transshipments(problem, route)
        res = RH.optimal_transshipment_checking(problem, route)
        assert feasible == res["feasible"]
        if feasible:
            assert cost == pytest.approx(res["cost"] - problem.cost_vehicle_use)
            assert [i for i, p in visits] == res["refills"]
            assert all(p == res["satellites"][i] for i, p in visits)
            assert route_back is route and problem.tabu_list == []
        else:
            assert visits == [] and route_back == [] and problem.tabu_list == [[x.index for x in route]]
    assert RH.check_route_transshipments(problem, []) == (True, 0, [], [])
