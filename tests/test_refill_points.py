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
