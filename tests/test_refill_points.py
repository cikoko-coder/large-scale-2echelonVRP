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
