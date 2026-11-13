import copy
import math
from multiprocessing import pool
from operator import index
import SEVs as sev
import MathModels
import instanceGenerator
import FEVmodels
import matplotlib.pyplot as plt
import time
import os

BigNumber = 1000000
def bothway_checking(problem_instance, route):
    """
    Refill points of the greedy trip split in both directions.
    :return: (forward, backward), each a list of refill node indices in route order
             (route[0] first, it is always a refill)
    """
    return forward_checking(problem_instance, route), backward_checking(problem_instance, route)

def forward_checking(problem_instance, route):
    """
    Finds node indices where a refill is required.
    :param demands: List of loads for each node (e.g., [2, 5, 8, 3])
    :param capacity: Maximum vehicle capacity Q
    :return: List of indices where the vehicle takes a break to refill
    """
    demands = [node.demand for node in route]
    capacity = problem_instance.capacity_SEV   

    
    total_demand = sum(demands)
    min_refills = math.ceil(total_demand / capacity)    

    if min_refills > 1:
        refill_indices = []
        current_load =  0    
        for i, demand in enumerate(demands):            
            # If adding this node exceeds current capacity, refill first
            if current_load + demand > capacity:
                refill_indices.append(route[i].index)
                # After refilling, the load is just the current node's demand
                current_load = demand
            else:
                # Continue adding to the current "bin"
                current_load += demand
          

        if route[0].index not in refill_indices:
            refill_indices.insert(0, route[0].index) #first node is always a refill
    else: 
        refill_indices = [route[0].index] #first node is always a refill
    return refill_indices

def backward_checking(problem_instance, route):
    """
    Refill points from a greedy split that starts at the END of the route.

    Scanning backwards, a node whose demand would overflow the current trip closes that
    trip: it is the LAST node of the chronologically earlier trip, so the refill happens
    just before its successor. Every trip is filled from its end, so the refills come as
    early as possible (forward_checking places them as late as possible); both give the
    minimum number of trips.
    :return: refill node indices in route order (route[0] first, it is always a refill)
    """
    if not route:
        return []
    capacity = problem_instance.capacity_SEV
    cut_positions = [] #positions where a new trip starts
    current_load = 0
    for pos in range(len(route) - 1, -1, -1):
        demand = route[pos].demand
        if current_load + demand > capacity:
            cut_positions.append(pos + 1) #refill before the successor
            current_load = demand
        else:
            current_load += demand
    cut_positions.sort()
    return [route[0].index] + [route[pos].index for pos in cut_positions]

#refill rule used by the relaxed route check (is_route_feasible):
#  "two_way"      -> two_way_transshipment_checking: the minimum number of transshipments K* that
#                    is on time (K* >= m, more than m only when the time windows force it), then
#                    the cheapest plan with K* transshipments (lexicographic: transshipments, cost)
#  "optimal_cost" -> optimal_transshipment_checking: the minimum number of transshipments m is
#                    the starting bound of the DP; more are used when K = m is not on time or
#                    when they are cheaper (same solution as the street MIP)
#  "optimal"      -> bothway_optimal_checking: exactly m transshipments, best refill points
#                    between the backward and forward bounds (DP); rejects routes needing m+1
#  "backward" / "forward" -> fixed greedy refill points
REFILL_RULE = "two_way"

#tolerance of the time-window checks of the refill DPs: arrivals are sums of Euclidean travel
#times, so an arrival exactly at the window end can come out 1e-13 too late (the street MIP
#accepts it within the MIP solver's feasibility tolerance)
TIME_EPS = 1e-9

def refill_points(problem_instance, route):
    """Refill (transfer) points assumed by the relaxed route check, see REFILL_RULE."""
    if REFILL_RULE == "forward":
        return forward_checking(problem_instance, route)
    if REFILL_RULE == "optimal":
        return bothway_optimal_checking(problem_instance, route)["refills"]
    if REFILL_RULE == "optimal_cost":
        return optimal_transshipment_checking(problem_instance, route)["refills"]
    if REFILL_RULE == "two_way":
        return two_way_transshipment_checking(problem_instance, route)["refills"]
    return backward_checking(problem_instance, route)

def refill_windows(problem_instance, route, n_trips, forward = None, backward = None):
    """
    Position windows [lo_k, hi_k] of the start of trip k (k = 0..n_trips-1) of a route
    split into n_trips trips; position 0 always starts trip 0.

    hi_k: the first k trips must cover positions 0..c_k-1; the forward greedy covers the
          most with k trips, so c_k <= forward[k] (k < minimum number of trips).
    lo_k: the last n_trips-k trips must cover positions c_k..n-1; the backward greedy
          covers the most with j trips, so c_k >= backward[m-j] (j = n_trips-k <= m).
    With the minimum number of trips m this is backward[k] <= c_k <= forward[k].
    Every trip has at least one customer: k <= c_k <= n - (n_trips - k).
    """
    n = len(route)
    where = {node.index: pos for pos, node in enumerate(route)}
    forward = forward if forward is not None else [where[i] for i in forward_checking(problem_instance, route)]
    backward = backward if backward is not None else [where[i] for i in backward_checking(problem_instance, route)]
    m = len(forward)
    lo, hi = [], []
    for k in range(n_trips):
        remaining = n_trips - k
        lo_k = backward[m - remaining] if remaining <= m else 0
        hi_k = forward[k] if k < m else n - 1
        lo.append(max(lo_k, k))
        hi.append(min(hi_k, n - remaining))
    lo[0] = hi[0] = 0
    return lo, hi

def bothway_optimal_checking(problem_instance, route):
    """
    Route feasibility with the MINIMUM number of transshipments m = number of trips of the
    greedy split: the k-th refill lies between the backward (earliest) and forward (latest)
    greedy positions, backward[k] <= c_k <= forward[k], and the DP picks the best refill
    points inside these windows (cheapest detours, on time, within capacity).
    A route that needs more than m transshipments to be on time is rejected here; see
    optimal_transshipment_checking, which uses m as the starting bound.
    :return: same dict as optimal_transshipment_checking
    """
    return optimal_transshipment_checking(problem_instance, route, extra_trips = 0)

#plan of the two-way check among the plans with the minimum number of transshipments K*:
#  "makespan" -> earliest return to the garage (then cheapest), "cost" -> cheapest (then earliest)
TWO_WAY_OBJECTIVE = "makespan"

def two_way_transshipment_checking(problem_instance, route):
    """
    Route feasibility with the MINIMUM number of transshipments that is on time, then among the
    plans with that number the one of TWO_WAY_OBJECTIVE: "makespan" = earliest return to the garage
    (then cheapest), "cost" = cheapest (then earliest). Lexicographic: transshipments first.

    * m = number of trips of the greedy split (capacity only) is a lower bound. Every split into
      m trips has its k-th refill between the backward (earliest) and forward (latest) greedy
      positions, backward[k] <= c_k <= forward[k] (refill_windows), so the windows lose no plan.
    * K* = the smallest K >= m with an on-time K-trip plan (pass 1 of the two-pass engine):
      refills only add time, so per (trip, start position) the earliest service start is the only
      state that matters for feasibility, and a state with fewer trips at the same position and an
      earlier or equal time dominates. K = m, m+1, ... in order, so the first K that reaches the end
      on time is the minimum. K* > m only when the time windows make every m-trip split late.
    * Plan with exactly K* trips (pass 2: Pareto labels (time, cost) inside the windows), the
      earliest or the cheapest; the closest satellite of every refill position minimizes both the
      arrival and the detour cost. The returned cost is the cost of the chosen plan.

    Unlike optimal_transshipment_checking, no plan with more than K* trips is considered, even
    when it would be cheaper (possible when trip_unit_cost is below the detour savings, e.g. the
    trip_unit_cost = 0). With trip_unit_cost = 10000 both give the same plan.
    :return: same dict as bounded_transshipment_checking; n_trips = min_trips_on_time = K*
    """
    result = two_pass_transshipment_checking(problem_instance, route, min_trips_first = True,
                                             objective = TWO_WAY_OBJECTIVE)
    returns_in_time(problem_instance, route, result)
    result["min_trips_on_time"] = result["n_trips"] if result["feasible"] else None
    return result

def returns_in_time(problem_instance, route, result):
    """Exact end-of-shift check of a feasible refill plan: the vehicle is back at the garage by
    its end, finish (end of the last service) + t(last customer, garage) <= garage end. Sets
    result["feasible"] = False (result["late_return"] = lateness) otherwise. With unbounded vehicle
    time the garage end is far after every customer window and the check never binds.
    :return: result["feasible"]"""
    if result["feasible"] and route:
        back = result["finish"] + problem_instance.travel_matrix[route[-1].index][0]
        late = back - getattr(problem_instance.depot, "end_interval", math.inf) #no garage end: unbounded
        if late > TIME_EPS:
            result["feasible"] = False
            result["late_return"] = late
    return result["feasible"]

def satellite_detour(problem_instance, i, j, matrix = "travel_matrix"):
    """(detour, satellite) of the closest satellite p on the arc i -> j: min over p of
    m(i,p) + m(p,j) - m(i,j) (ties by the smaller index), m = problem_instance.<matrix>.
    Memoized per problem, matrix and satellite set (consolidation swaps problem.satellites for
    another list); the set's key is built only when the list changes, not on every call."""
    sats = problem_instance.satellites
    current = problem_instance.__dict__.setdefault("_satellite_detour_current", {})
    entry = current.get(matrix)
    if entry is None or entry[0] is not sats or entry[1] != len(sats):
        memo = problem_instance.__dict__.setdefault("_satellite_detour_memo", {})
        entry = (sats, len(sats), memo.setdefault((matrix, tuple(p.index for p in sats)), {}))
        current[matrix] = entry
    memo = entry[2]
    hit = memo.get((i, j))
    if hit is None:
        m = getattr(problem_instance, matrix)
        hit = min((m[i][p.index] + m[p.index][j] - m[i][j], p.index) for p in sats)
        memo[(i, j)] = hit
    return hit

def refill_detours(problem_instance, route):
    """(detour, satellite) of the closest satellite for a refill before every route position:
    detour = t(i,p) + t(p,j) - t(i,j) >= 0 with i the predecessor (garage for position 0)."""
    previous = [0] + [node.index for node in route[:-1]]
    out = []
    for pos, node in enumerate(route):
        detour, satellite = satellite_detour(problem_instance, previous[pos], node.index)
        out.append((max(0.0, detour), satellite)) #lead time >= 0 in the MIP (rounding can give -0.1)
    return out

def refill_time_bounds(problem_instance, route, detours = None):
    """
    Time relaxation of the refill plan of a fixed route. A refill only adds time (detour >= 0
    plus the transfer duration), so the fastest schedule refills only before the first
    customer. Returns None when that schedule misses a window: then no refill plan is on time.
    Otherwise returns (earliest, latest, can_refill, refill_slack):
      earliest[pos]     end of service at pos in the fastest schedule;
      latest[pos]       latest start of service at pos from which every later customer is still
                        on time without further refills (an upper bound on the real latest start);
      refill_slack[pos] min(window end, latest[pos]) - earliest arrival at pos with the detour and
                        the transfer: negative means a refill before pos is late in every plan;
      can_refill[pos]   refill_slack[pos] >= 0. Position 0 is always a refill and is always True
                        when the fastest schedule is on time.
    """
    n = len(route)
    t = problem_instance.travel_matrix
    transfer = problem_instance.TransferDuration
    detours = detours if detours is not None else refill_detours(problem_instance, route)
    previous = [0] + [node.index for node in route[:-1]]
    drive = [t[previous[pos]][node.index] for pos, node in enumerate(route)]

    earliest = []
    finish = problem_instance.depot.start_interval + detours[0][0] + transfer
    for pos, node in enumerate(route):
        arrival = finish + drive[pos]
        if arrival > node.end_interval + TIME_EPS:
            return None
        finish = max(arrival, node.start_interval) + node.unloading_time
        earliest.append(finish)

    latest = [node.end_interval for node in route]
    for pos in range(n - 2, -1, -1):
        latest[pos] = min(latest[pos], latest[pos + 1] - drive[pos + 1] - route[pos].unloading_time)

    refill_slack = [math.inf]
    for pos in range(1, n):
        arrival = earliest[pos - 1] + drive[pos] + detours[pos][0] + transfer
        refill_slack.append(min(route[pos].end_interval, latest[pos]) - arrival)
    can_refill = [slack >= -TIME_EPS for slack in refill_slack]
    return earliest, latest, can_refill, refill_slack

def fastest_refill_plan(problem_instance, route, n_trips, lo, hi, can_refill = None, detours = None):
    """
    Feasibility of a split of the route into exactly n_trips trips whose k-th trip starts in
    [lo[k], hi[k]] (refill_windows): dynamic programming that keeps only the EARLIEST end of
    service per (trips started, start position of the current trip), O(n * n_trips * window width).
    :return: dict feasible, finish, refills (positions), blocked = (position, lateness) of the
             position where every label died (None when feasible): the smallest arrival there
             over every continuation exceeds the window end by `lateness`.
    """
    n = len(route)
    t = problem_instance.travel_matrix
    capacity = problem_instance.capacity_SEV
    transfer = problem_instance.TransferDuration
    detours = detours if detours is not None else refill_detours(problem_instance, route)
    can_refill = can_refill if can_refill is not None else [True]*n
    prefix = [0]
    for node in route:
        prefix.append(prefix[-1] + node.demand)
    previous = [0] + [node.index for node in route[:-1]]

    result = {"feasible": False, "finish": None, "refills": [], "blocked": None}
    node = route[0]
    arrival = problem_instance.depot.start_interval + t[0][node.index] + detours[0][0] + transfer
    if arrival > node.end_interval + TIME_EPS:
        result["blocked"] = (0, arrival - node.end_interval)
        return result
    labels = {(1, 0): (max(arrival, node.start_interval) + node.unloading_time, (0,))}
    for pos in range(1, n):
        node = route[pos]
        drive = t[previous[pos]][node.index]
        new_labels = {}
        best_arrival = math.inf
        for (started, start), (finish, refills) in labels.items():
            must_cut_before = started < n_trips and pos > hi[started]
            if not must_cut_before and prefix[pos + 1] - prefix[start] <= capacity:
                arrival = finish + drive
                best_arrival = min(best_arrival, arrival)
                if arrival <= node.end_interval + TIME_EPS:
                    label = (max(arrival, node.start_interval) + node.unloading_time, refills)
                    if (started, start) not in new_labels or label[0] < new_labels[(started, start)][0]:
                        new_labels[(started, start)] = label
            if started < n_trips and lo[started] <= pos <= hi[started] and can_refill[pos]:
                arrival = finish + drive + detours[pos][0] + transfer
                best_arrival = min(best_arrival, arrival)
                if arrival <= node.end_interval + TIME_EPS:
                    label = (max(arrival, node.start_interval) + node.unloading_time, refills + (pos,))
                    if (started + 1, pos) not in new_labels or label[0] < new_labels[(started + 1, pos)][0]:
                        new_labels[(started + 1, pos)] = label
        if not new_labels:
            result["blocked"] = (pos, best_arrival - node.end_interval if best_arrival < math.inf else math.inf)
            return result
        labels = new_labels
    complete = [(finish, refills) for (started, start), (finish, refills) in labels.items() if started == n_trips]
    if not complete:
        result["blocked"] = (n - 1, math.inf) #capacity: the last trips could not be started
        return result
    finish, refills = min(complete)
    result.update(feasible=True, finish=finish, refills=list(refills))
    return result

#engine of the bothway checks (optimal_transshipment_checking, bothway_optimal_checking):
#  "two_pass" -> two_pass_transshipment_checking: pass 1 finds the minimum number of trips K* that
#                is on time (earliest start per (trip, start position)), pass 2 the cheapest plan
#                with K* trips (Pareto labels), more trips only while they can still be cheaper
#  "bounded"  -> bounded_transshipment_checking: K = m, m+1, ... each with a feasibility pass
#                (fastest_refill_plan) and a Pareto DP
REFILL_DP = "two_pass"

def optimal_transshipment_checking(problem_instance, route, extra_trips = None):
    """Optimal refill (transfer) plan of a fixed street route with the engine set by REFILL_DP;
    see bounded_transshipment_checking for the rules, the arguments and the returned dict."""
    if REFILL_DP == "bounded":
        return bounded_transshipment_checking(problem_instance, route, extra_trips)
    return two_pass_transshipment_checking(problem_instance, route, extra_trips)

def bounded_transshipment_checking(problem_instance, route, extra_trips = None):
    """
    Optimal refill (transfer) points of a fixed street route: dynamic programming in the
    windows between the backward (earliest) and forward (latest) greedy refill positions.

    Same rules as the street MIP (MathModels.street_problem_MIP) for a single route:
    * a refill before customer j goes through the satellite with the smallest detour
      t(i,p) + t(p,j) - t(i,j) (fastest and cheapest), plus the transfer duration;
      the first customer is always a refill;
    * the vehicle leaves the garage at the vehicle start, waits for a window opening and
      fails when it arrives after the window end;
    * trip loads <= SEV capacity;
    * cost = vehicle use + travel cost of the route + travel cost of the detours
             + trip_unit_cost per refill   (the MIP objective for one route).
    The minimum number of trips m (greedy split) is the starting BOUND, not a rule: K = m is
    tried first; when it is feasible its cost is the incumbent and K = m+1, m+2, ... are
    explored only while the lower bound (fixed cost + K * trip_unit_cost) is below the incumbent
    (with a transshipment cost of 10000 this stops at once); when K = m is infeasible, K grows
    until a feasible split is found (tight time windows can force an extra transshipment) or no
    window position can take a refill. extra_trips limits K to m + extra_trips (0: minimum
    transshipments only, None: no limit, the street MIP's optimum).
    Per K: refill_time_bounds removes the window positions that cannot take a refill on time,
    fastest_refill_plan (earliest finish per (trip, start), O(n*W)) decides feasibility, and the
    Pareto DP over (finish time, cost) labels finds the cheapest plan only for feasible K.

    :return: dict with feasible, refills (customer indices), satellites ({customer: satellite}),
             cost (full route cost), transfer_cost (detours + refills, the cost street_problem_MIP
             returns), n_trips, min_trips (m), finish (end of the last service), labels (DP work),
             attempts ({K: "feasible" | reason}), blocked: for an infeasible route, (position,
             customer index, lateness) where the fastest schedule / the K = m split dies
    """
    result = {"feasible": False, "refills": [], "satellites": {}, "cost": BigNumber, "transfer_cost": BigNumber,
              "n_trips": 0, "min_trips": 0, "finish": None, "labels": 0, "attempts": {}, "blocked": None}
    n = len(route)
    if n == 0:
        result.update(feasible=True, cost=0, transfer_cost=0)
        return result
    capacity = problem_instance.capacity_SEV
    if any(node.demand > capacity for node in route):
        result["attempts"]["capacity"] = "a customer demand exceeds the SEV capacity"
        return result
    t = problem_instance.travel_matrix
    unit = problem_instance.cost_unit_travel
    transfer = problem_instance.TransferDuration

    prefix = [0]
    for node in route:
        prefix.append(prefix[-1] + node.demand)
    previous = [0] + [node.index for node in route[:-1]]
    detours = refill_detours(problem_instance, route)
    #time relaxation: every refill adds a detour >= 0, so the fastest schedule refills only before
    #the first customer (capacity ignored); if it misses a window, no refill plan is on time
    bounds = refill_time_bounds(problem_instance, route, detours)
    if bounds is None:
        finish = problem_instance.depot.start_interval + detours[0][0] + transfer
        for pos, node in enumerate(route):
            arrival = finish + t[previous[pos]][node.index]
            if arrival > node.end_interval + TIME_EPS:
                result["blocked"] = (pos, node.index, arrival - node.end_interval)
                break
            finish = max(arrival, node.start_interval) + node.unloading_time
        result["attempts"]["time"] = "the fastest schedule (no refill after the first customer) is late"
        return result
    earliest, latest, can_refill, refill_slack = bounds
    fixed_cost = problem_instance.cost_vehicle_use + unit*(
        sum(t[previous[pos]][node.index] for pos, node in enumerate(route)) + t[route[-1].index][0])

    where = {node.index: pos for pos, node in enumerate(route)}
    forward = [where[i] for i in forward_checking(problem_instance, route)]
    backward = [where[i] for i in backward_checking(problem_instance, route)]
    m = len(forward)
    result["min_trips"] = m
    max_trips = min(n, sum(can_refill))
    if extra_trips is not None:
        max_trips = min(max_trips, m + extra_trips)

    def add_label(labels, key, label):
        front = labels.setdefault(key, [])
        for other in front:
            if other[0] <= label[0] + 1e-9 and other[1] <= label[1] + 1e-9:
                return #dominated
        front[:] = [o for o in front if not (label[0] <= o[0] + 1e-9 and label[1] <= o[1] + 1e-9)]
        front.append(label)

    best = None
    for n_trips in range(m, max_trips + 1):
        if best is not None and fixed_cost + n_trips*problem_instance.trip_unit_cost >= best[1] - 1e-9:
            result["attempts"][n_trips] = "lower bound >= incumbent"
            break #more refills cannot be cheaper
        lo, hi = refill_windows(problem_instance, route, n_trips, forward, backward)
        #shrink the windows to the positions that can take a refill on time
        empty = None
        for k in range(1, n_trips):
            feasible_positions = [pos for pos in range(lo[k], hi[k] + 1) if can_refill[pos]]
            if not feasible_positions:
                empty = k
                break
            lo[k], hi[k] = feasible_positions[0], feasible_positions[-1]
        if empty is not None:
            if lo[empty] <= hi[empty]:
                late = ", ".join(f"customer {route[pos].index} late by {round(-refill_slack[pos], 1)}"
                                 for pos in range(lo[empty], hi[empty] + 1))
                result["attempts"][n_trips] = (f"refill {empty} must be at positions {lo[empty]}..{hi[empty]} "
                                               f"and no position can take a refill on time ({late})")
                if n_trips == m:
                    pos = max(range(lo[empty], hi[empty] + 1), key=lambda p: refill_slack[p])
                    result["blocked"] = (pos, route[pos].index, -refill_slack[pos])
            else:
                result["attempts"][n_trips] = f"window {empty} is empty"
            continue
        #feasibility first: earliest finish per (trips started, start of the current trip)
        fast = fastest_refill_plan(problem_instance, route, n_trips, lo, hi, can_refill, detours)
        if not fast["feasible"]:
            pos, late = fast["blocked"]
            result["attempts"][n_trips] = (f"late at position {pos} (customer {route[pos].index}) by {round(late, 1)}"
                                           if late < math.inf else f"no label reaches position {pos} within capacity")
            if n_trips == m:
                result["blocked"] = (pos, route[pos].index, late)
            continue
        result["attempts"][n_trips] = "feasible"
        #cost: labels (trips started, start position of current trip) -> Pareto set of (finish time, cost, refills)
        labels = {}
        node = route[0]
        detour, _ = detours[0]
        arrival = problem_instance.depot.start_interval + t[0][node.index] + detour + transfer
        add_label(labels, (1, 0), (max(arrival, node.start_interval) + node.unloading_time, unit*detour, (0,)))
        for pos in range(1, n):
            node = route[pos]
            drive = t[previous[pos]][node.index]
            new_labels = {}
            for (started, start), front in labels.items():
                for finish, cost, refills in front:
                    #continue the current trip
                    must_cut_before = started < n_trips and pos > hi[started]
                    if not must_cut_before and prefix[pos + 1] - prefix[start] <= capacity:
                        arrival = finish + drive
                        if arrival <= node.end_interval + TIME_EPS:
                            add_label(new_labels, (started, start),
                                      (max(arrival, node.start_interval) + node.unloading_time, cost, refills))
                    #refill before this customer: start trip number `started`
                    if started < n_trips and lo[started] <= pos <= hi[started] and can_refill[pos]:
                        detour, _ = detours[pos]
                        arrival = finish + drive + detour + transfer
                        if arrival <= node.end_interval + TIME_EPS:
                            add_label(new_labels, (started + 1, pos),
                                      (max(arrival, node.start_interval) + node.unloading_time,
                                       cost + unit*detour, refills + (pos,)))
            result["labels"] += sum(len(front) for front in new_labels.values())
            labels = new_labels
        complete = [label for (started, start), front in labels.items() if started == n_trips for label in front]
        finish, cost, refills = min(complete, key=lambda x: (x[1], x[0]))
        total = fixed_cost + cost + n_trips*problem_instance.trip_unit_cost
        if best is None or total < best[1] - 1e-9:
            best = (n_trips, total, finish, refills)

    if best is not None:
        n_trips, total, finish, refills = best
        result.update(feasible=True, n_trips=n_trips, cost=total, finish=finish,
                      transfer_cost=total - fixed_cost, #detours + refills: what street_problem_MIP returns as cost
                      refills=[route[pos].index for pos in refills],
                      satellites={route[pos].index: detours[pos][1] for pos in refills})
    return result

def two_pass_transshipment_checking(problem_instance, route, extra_trips = None, min_trips_first = False,
                                    objective = "cost"):
    """
    Optimal refill (transfer) plan of a fixed street route in two passes; same rules, arguments
    and returned dict as bounded_transshipment_checking (the street MIP for one route).

    Notation (route positions 0..n-1): det[q] = closest-satellite detour of a refill just before
    q (refill_detours), the transfer duration is added to the arrival after every refill,
    L[i] = latest start of service at i from which the rest is on time without refills
    (refill_time_bounds), window(k, K) = [lo_k, hi_k] of the start of trip k of a K-trip split
    between the backward and forward greedy positions (refill_windows).

    WALK(p, tau): trip starting at p with service start tau, followed while it is on time
    (tau <= L[i]) and within capacity; every position i+1 it passes is a possible start
    (refill) of the next trip, with departure tau_i + s_i.

    Pass 1 (minimum number of trips): A[k][p] = earliest service start at p when trip k starts
    at p. K = m, m+1, ...: the states of window(k, K) are expanded (again when their time
    improved since the last expansion) unless a state with fewer trips at p was expanded with
    an earlier or equal time; the first WALK that reaches the end gives K*.
    Pass 2 (minimum cost with K trips): Pareto labels (time, cost) per (k, p) inside the
    windows; a refill before q adds c^s * det[q]. K = K*, K*+1, ... while the lower bound
    vehicle + distance + det[0] + K * trip_unit_cost is below the incumbent: with
    trip_unit_cost >= the detour cost of the K* plan this stops after K*.
    Cost: every trip counts one trip_unit_cost (the first load included), as the street MIP.
    min_trips_first: pass 2 only for K = K* (two_way_transshipment_checking).
    objective: plan chosen among the K-trip plans of pass 2: "cost" = cheapest (then earliest),
    "makespan" = earliest end of the last service, i.e. earliest return to the garage for the fixed
    route (then cheapest). The labels are Pareto-optimal in (time, cost), so both are exact.
    """
    result = {"feasible": False, "refills": [], "satellites": {}, "cost": BigNumber, "transfer_cost": BigNumber,
              "n_trips": 0, "min_trips": 0, "finish": None, "labels": 0, "attempts": {}, "blocked": None}
    n = len(route)
    if n == 0:
        result.update(feasible=True, cost=0, transfer_cost=0)
        return result
    capacity = problem_instance.capacity_SEV
    if any(node.demand > capacity for node in route):
        result["attempts"]["capacity"] = "a customer demand exceeds the SEV capacity"
        return result
    t = problem_instance.travel_matrix
    unit = problem_instance.cost_unit_travel
    trip_cost = problem_instance.trip_unit_cost
    transfer = problem_instance.TransferDuration
    eps = TIME_EPS

    previous = [0] + [node.index for node in route[:-1]]
    drive = [t[previous[pos]][node.index] for pos, node in enumerate(route)]
    demand = [node.demand for node in route]
    opening = [node.start_interval for node in route]
    service = [node.unloading_time for node in route]
    detours = refill_detours(problem_instance, route)
    det = [d for d, _ in detours]
    bounds = refill_time_bounds(problem_instance, route, detours)
    if bounds is None: #the fastest schedule (one refill before the first customer) is late
        finish = problem_instance.depot.start_interval + det[0] + transfer
        for pos, node in enumerate(route):
            arrival = finish + drive[pos]
            if arrival > node.end_interval:
                result["blocked"] = (pos, node.index, arrival - node.end_interval)
                break
            finish = max(arrival, node.start_interval) + node.unloading_time
        result["attempts"]["time"] = "the fastest schedule (no refill after the first customer) is late"
        return result
    earliest, latest, can_refill, refill_slack = bounds
    L = latest #latest[i] <= window end of i
    fixed_cost = problem_instance.cost_vehicle_use + unit*(sum(drive) + t[route[-1].index][0])

    where = {node.index: pos for pos, node in enumerate(route)}
    forward = [where[i] for i in forward_checking(problem_instance, route)]
    backward = [where[i] for i in backward_checking(problem_instance, route)]
    m = len(forward)
    result["min_trips"] = m
    max_trips = min(n, sum(can_refill))
    if extra_trips is not None:
        max_trips = min(max_trips, m + extra_trips)

    def walk(p, tau):
        """Trip from p with service start tau: ([(q, departure from q-1)], reached the end, last start)."""
        steps, load, i = [], 0, p
        while True:
            if i > p:
                tau = max(opening[i], tau + service[i - 1] + drive[i])
                if tau > L[i] + eps:
                    return steps, False, None
            load += demand[i]
            if load > capacity:
                return steps, False, None
            if i == n - 1:
                return steps, True, tau
            steps.append((i + 1, tau + service[i]))
            i += 1

    def refill_arrival(q, departure):
        """Service start at q after a refill before q, or None when that is late."""
        arrival = max(opening[q], departure + drive[q] + det[q] + transfer)
        return arrival if arrival <= L[q] + eps else None

    def failure_reason(K):
        """Why no K-trip split is on time (diagnostics of a failed trip count)."""
        lo, hi = refill_windows(problem_instance, route, K, forward, backward)
        for k in range(1, K):
            if lo[k] > hi[k]:
                return f"window {k} is empty", None
            if not any(can_refill[pos] for pos in range(lo[k], hi[k] + 1)):
                late = ", ".join(f"customer {route[pos].index} late by {round(-refill_slack[pos], 1)}"
                                 for pos in range(lo[k], hi[k] + 1))
                pos = max(range(lo[k], hi[k] + 1), key=lambda x: refill_slack[x])
                return (f"refill {k} must be at positions {lo[k]}..{hi[k]} and no position can take a refill "
                        f"on time ({late})"), (pos, route[pos].index, -refill_slack[pos])
        if K == m: #where the earliest K-trip schedule dies (only for the minimum, it is the reported cause)
            fast = fastest_refill_plan(problem_instance, route, K, lo, hi, can_refill, detours)
            pos, late = fast["blocked"]
            if late < math.inf:
                return (f"late at position {pos} (customer {route[pos].index}) by {round(late, 1)}",
                        (pos, route[pos].index, late))
            return f"no label reaches position {pos} within capacity", (pos, route[pos].index, late)
        return "no split into this number of trips is on time (pass 1)", None

    start = max(opening[0], problem_instance.depot.start_interval + drive[0] + det[0] + transfer)
    if start > L[0] + eps: #covered by refill_time_bounds, kept for safety
        result["attempts"]["time"] = "the first customer is late"
        result["blocked"] = (0, route[0].index, start - L[0])
        return result

    # ---- pass 1: minimum number of trips K* ---- #
    A = {(0, 0): start}      #(trip k, start position p) -> earliest service start at p
    expanded = {}            #(k, p) -> time it was expanded with
    best_at = {}             #p -> sorted list of (k, expanded time), for the dominance test
    k_star = None
    for K in range(m, max_trips + 1):
        lo, hi = refill_windows(problem_instance, route, K, forward, backward)
        for k in range(K):
            for p in range(lo[k], hi[k] + 1):
                a = A.get((k, p))
                if a is None or expanded.get((k, p), math.inf) <= a + eps:
                    continue
                if any(kk < k and time <= a + eps for kk, time in best_at.get(p, ())):
                    continue #dominated: fewer trips at p, not later
                expanded[(k, p)] = a
                best_at.setdefault(p, []).append((k, a))
                steps, reached, _ = walk(p, a)
                result["labels"] += 1
                if reached:
                    k_star = k + 1
                    break
                for q, departure in steps:
                    arrival = refill_arrival(q, departure)
                    if arrival is not None and arrival < A.get((k + 1, q), math.inf) - eps:
                        A[(k + 1, q)] = arrival
            if k_star is not None:
                break
        if k_star is not None:
            break

    for K in range(m, (k_star if k_star is not None else max_trips + 1)):
        reason, blocked = failure_reason(K)
        result["attempts"][K] = reason
        if K == m:
            result["blocked"] = blocked
    if k_star is None:
        return result

    # ---- pass 2: cheapest plan with K trips, K = K*, K*+1, ... ---- #
    def add_label(front, label):
        for other in front:
            if other[0] <= label[0] + eps and other[1] <= label[1] + eps:
                return False #dominated
        front[:] = [o for o in front if not (label[0] <= o[0] + eps and label[1] <= o[1] + eps)]
        front.append(label)
        return True

    first_cost = unit*det[0]
    best = None #(n_trips, total, finish, label)
    for K in range(k_star, max_trips + 1):
        if best is not None and min_trips_first:
            break #lexicographic: the plan with the minimum number of transshipments K* is kept
        if best is not None and fixed_cost + first_cost + K*trip_cost >= best[1] - eps:
            result["attempts"][K] = "lower bound >= incumbent"
            break
        lo, hi = refill_windows(problem_instance, route, K, forward, backward)
        labels = {(0, 0): [(start, 0.0, None, 0)]} #(service start, detour cost after position 0, parent, position)
        found = None #(cost, finish, label)
        for k in range(K):
            last = k == K - 1
            for p in range(lo[k], hi[k] + 1):
                for label in labels.get((k, p), ()):
                    tau, cost = label[0], label[1]
                    steps, reached, end_start = walk(p, tau)
                    if last:
                        if reached:
                            finish = end_start + service[-1]
                            if objective == "makespan": #earliest end, ties (within eps) by cost
                                better = (found is None or finish < found[1] - eps
                                          or (finish <= found[1] + eps and cost < found[0] - eps))
                            else:
                                better = found is None or (cost, finish) < (found[0] - eps, found[1])
                            if better:
                                found = (cost, finish, label)
                        continue
                    for q, departure in steps:
                        if not lo[k + 1] <= q <= hi[k + 1]:
                            continue
                        arrival = refill_arrival(q, departure)
                        if arrival is None:
                            continue
                        if add_label(labels.setdefault((k + 1, q), []), (arrival, cost + unit*det[q], label, q)):
                            result["labels"] += 1
        if found is None:
            result["attempts"][K] = "no split into this number of trips is on time (pass 2)"
            continue
        result["attempts"][K] = "feasible"
        total = fixed_cost + first_cost + found[0] + K*trip_cost
        if best is None or total < best[1] - eps:
            best = (K, total, found[1], found[2])

    n_trips, total, finish, label = best
    refills = []
    while label is not None:
        refills.append(label[3])
        label = label[2]
    refills.reverse()
    result.update(feasible=True, n_trips=n_trips, cost=total, finish=finish,
                  transfer_cost=total - fixed_cost,
                  refills=[route[pos].index for pos in refills],
                  satellites={route[pos].index: detours[pos][1] for pos in refills})
    return result

#route check in front of / instead of the street MIP:
#  "dp"  -> optimal_transshipment_checking (same plan and cost as the MIP, no solver)
#  "mip" -> MathModels.street_problem_MIP
ROUTE_CHECK = "dp"

def route_refill_plan(problem_instance, route):
    """
    Refill plan of one fixed route by the DP of REFILL_RULE (the rule of is_route_feasible):
    "two_way" -> two_way_transshipment_checking (minimum transshipments K*, then cheapest),
    "optimal" -> bothway_optimal_checking (exactly m), otherwise optimal_transshipment_checking
    (cheapest with any number). :return: the DP result dict (cost includes the vehicle cost)
    """
    if REFILL_RULE == "two_way":
        return two_way_transshipment_checking(problem_instance, route)
    if REFILL_RULE == "optimal":
        return bothway_optimal_checking(problem_instance, route)
    return optimal_transshipment_checking(problem_instance, route)

def two_way_solution_cost(problem_instance, routes):
    """
    Street cost of a complete route set decided by the two-way check: every route gets the plan
    of two_way_transshipment_checking (minimum number of on-time transshipments K*, then the
    cheapest plan with K*), the solution cost is the sum of the route costs, each = vehicle use +
    c^s (distance + detours) + trip_unit_cost per trip. Used to compare initial solutions of
    different methods (heuristic construction, free routing MIP) on one scale, independent of
    REFILL_RULE and of the MIP's own objective.
    :return: dict feasible, cost (BigNumber if a route is infeasible), n_vehicles, n_trips,
             served, route_costs, route_trips, infeasible_routes (indices)
    """
    out = {"feasible": True, "cost": 0.0, "n_vehicles": 0, "n_trips": 0, "served": 0,
           "route_costs": [], "route_trips": [], "infeasible_routes": []}
    for r, route in enumerate(routes):
        if len(route) == 0:
            continue
        out["n_vehicles"] += 1
        out["served"] += len(route)
        plan = two_way_transshipment_checking(problem_instance, route)
        if not plan["feasible"]:
            out["feasible"] = False
            out["infeasible_routes"].append(r)
            out["route_costs"].append(BigNumber)
            out["route_trips"].append(None)
            continue
        out["route_costs"].append(plan["cost"])
        out["route_trips"].append(plan["n_trips"])
        out["n_trips"] += plan["n_trips"]
        out["cost"] += plan["cost"]
    if out["served"] != len(problem_instance.customers):
        out["feasible"] = False
    if not out["feasible"]:
        out["cost"] = BigNumber
    return out

def check_route_transshipments(problem_instance, route):
    """
    Transshipment plan of ONE fixed route (DP of REFILL_RULE, route_refill_plan; default
    "two_way"), the street_problem_MIP interface:
    (feasible, cost, visits, route) with cost = the street travel cost of the route, c^s *
    (distance + detours) + trip_unit_cost per transshipment (no vehicle cost), and
    visits = [[customer, satellite], ...] in route order.
    An infeasible route returns (False, MathModels.bigNumber, [], []) and goes to the tabu list
    when the tabu search is active (as the MIP does).
    """
    if ROUTE_CHECK == "mip":
        return route_mip_check(problem_instance, route)
    if len(route) == 0:
        return True, 0, [], route
    result = route_refill_plan(problem_instance, route)
    if result["feasible"]:
        visits = [[i, result["satellites"][i]] for i in result["refills"]]
        return True, result["cost"] - problem_instance.cost_vehicle_use, visits, route
    if not problem_instance.consolidated and problem_instance.tabu_search_active:
        if not is_in_tabu_list(problem_instance, route):
            problem_instance.tabu_list.append([x.index for x in route])
    return False, MathModels.bigNumber, [], []

def route_mip_check(problem_instance, route):
    """
    SolveFixedSEVRouteMILP with memo (manuscript Algorithm 10): MathModels.street_problem_MIP of the
    fixed customer order, which places the transshipments at minimum cost (no minimum-transshipment
    bound). The result is cached per customer sequence and satellite set; an infeasible route goes
    to the tabu list inside the MIP call (tabu search active, not during consolidation).
    :return: (feasible, cost, visits, route), the check_route_transshipments interface
    """
    if len(route) == 0:
        return True, 0, [], route
    cache = problem_instance.__dict__.setdefault("_route_mip_cache", {})
    key = (tuple(x.index for x in route), tuple(p.index for p in problem_instance.satellites))
    hit = cache.get(key)
    if hit is None:
        feasible, cost, visits, _ = MathModels.street_problem_MIP(problem_instance, problem_instance.timelimit_SEmodel, route)
        hit = (feasible, cost, [list(v) for v in visits])
        cache[key] = hit
    feasible, cost, visits = hit
    #callers append the trip load to every visit: hand out copies
    return feasible, cost, [list(v) for v in visits], (route if feasible else [])

def mip_checking(problem_instance, route):

    feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, route)
     #visit [i,p]
    if feasibility:
        transfer_points = [visit[0] for visit in visits ]   
    else:
        transfer_points = []
    
    return transfer_points, cost

def comparison(problem_instance, route):
    vrp_cost = problem_instance.cost_unit_travel*sum(problem_instance.distance_matrix[route[i].index][route[i+1].index] for i in range(len(route)-1))
    vrp_cost += problem_instance.cost_unit_travel*problem_instance.distance_matrix[route[-1].index][0] 
    vrp_cost += problem_instance.cost_unit_travel*problem_instance.distance_matrix[0][route[0].index] #return to depot and departure from depot
    #transfer costs
    #sets of solutions
    #forward = forward_checking(problem_instance, route) 
    forward, backward = bothway_checking(problem_instance, route)
    
    #distance increase to visit the closest satellite before the forward checking points
    forward_cost = check_cost_VRP_given_transfer_points(problem_instance, route, forward)
    backward_cost = check_cost_VRP_given_transfer_points(problem_instance, route, backward)

    if forward_cost < backward_cost: #solution is the same for routing but different for the transfer points
        
        #if feasible return the forward solution
        trips = forward
        #feasibility = check_tw(problem_instance, route, forward)
    
    return vrp_cost, forward_cost, backward_cost

def check_cost_VRP_given_transfer_points(problem_instance, route, transfer_points):
    #visit [i,p]
    transfer_cost = problem_instance.trip_unit_cost*len(transfer_points)
    for i in range(len(route)):
        index_out = route[i].index
        if index_out in transfer_points:
            if i == 0:
                index_in = 0
            else:
                index_in = route[i-1].index
                
            transfer_cost += problem_instance.closest_satellite_network.check_arc_cost(index_in, index_out)

    return transfer_cost 

def categorize_orders(problem_instance, solution, pool_nodes, partial = False):
    #separate node ordering function
    workloads = []
    satellite_assignements = []
    for s in problem_instance.satellites:
        workloads.append([])
        satellite_assignements.append([])

    
    for node in pool_nodes:
        closest = [problem_instance.distance_matrix[node.index][satellite.index] for satellite in problem_instance.satellites]
        index_close = closest.index(min(closest))
        workloads[index_close].append(node)    
        satellite_assignements[index_close].append(node)
    
    workloads = [x for x in workloads if len(x) > 0]
    workloads.sort(key=lambda x: len(x), reverse = True)
    
    #workloads = []
    #workloads.append(pool_nodes) 
    #max_number =min(problem_instance.removal_max, len(workloads[0])) 
    max_number =min(1000, len(workloads[0]))
    
    pool_nodes = workloads[0][:max_number]
    if max_number == len(workloads[0]):
        workloads.pop(0)
    else:
        workloads[0] = [node for node in workloads[0] if node not in pool_nodes]

    w = sum([len(list) for list in workloads])
    if not partial and sum(len(x) for x in solution) + w + len(pool_nodes) != len(problem_instance.customers) :
        print(" more nodes inserted than customers")    

    return pool_nodes, workloads



def k_best_insertion_repair(problem_instance, solution, pool_nodes, partial = False): 
    
    pool_nodes, workloads = categorize_orders(problem_instance, solution, pool_nodes, partial)    
    
    while len(pool_nodes) > 0:
        update_intervals(problem_instance, solution)
        
        before = [node.index for node in pool_nodes]
        node = pool_nodes[0] 
        last_node = node

        if node.index == "67":
            print("here")
            
        # Calculate insertion costs for the current node into the route's sequence
        insertion_costs = k_best_insertion_costs_per_route(problem_instance, solution, node )
        # Populate insertion_costs based on the calculation
        # Find the best insertion cost for the current node and add it to the result
        if not insertion_costs: # If a feasible solution was found                
            #worst case scenario is to open a new route    
            new_route = [node]
            pool_nodes.remove(node)
            solution.append(new_route)   

        else: #feasible insertion exist    
            
            # Insertion costs is sorted so start with best and see if there is a better one battery wise
            insertion_costs.sort(key=lambda x: x[2])

            best_insertion = insertion_costs[0] 
                
            # Now take the best insertion and add it to the routes
            route_new = best_insertion[1]
            insertion_index = route_new.index(node)
            route_new.remove(node)   
            solution.remove(route_new)
            
            #append update route(important for list based references with addres pointer information)
            route_new_s = route_new[:insertion_index] + [node] + route_new[insertion_index:]                        
            solution.append(route_new_s)
            pool_nodes.remove(node)
                
        
         
        after = [node.index for node in pool_nodes]
        
        if before == after:
            print("no insertion")   
        if sum( x.demand for x in pool_nodes) < problem_instance.capacity_SEV:
            if len(workloads) > 0:     
                selected = [problem_instance.distance_matrix[last_node.index][x[0].index] for x in workloads ]           
                
                index_chosen = selected.index(min(selected))
                chosen = workloads[index_chosen]                                

                max_number =min(problem_instance.removal_max, len(chosen)) 
    
                new_pool_nodes = chosen[0:max_number]
                if max_number == len(chosen):
                    workloads.remove(chosen)
                else:
                    workloads[index_chosen] = [node for node in workloads[index_chosen] if node not in new_pool_nodes]

                pool_nodes.extend(new_pool_nodes) # workloads[0]
                #workloads.pop(0)
                terminated = True

        w = sum([len(list) for list in workloads]) 
        if not partial and sum(len(x) for x in solution) + w + len(pool_nodes) != len(problem_instance.customers) :
            print(" more nodes inserted than customers")   

        if not pool_nodes:
            break

    if sum(len(x) for x in solution) == len(problem_instance.customers):
        partial = False

    if not pool_nodes and not partial: #important to track and store all possible solutions in the feasible solution list for the future reference and to avoid infeasible insertions in the future
        add_to_feasible_solution(problem_instance, solution)

    if len(pool_nodes)>0: #complete the solution all the time
        for node in pool_nodes:
            xx = [node]
            solution.append(xx)
        pool_nodes.clear()
    
    return solution, pool_nodes


def k_best_insertion_costs_per_route(problem_instance, solution, customer_node):
        
    """Every insertion of customer_node that passes the slack-time prefilter is checked by the
    two-way route check (is_route_feasible: minimum on-time transshipments, then the cheapest
    plan) and costed exactly: [True, new route, two-way cost with the customer - without].
    Infeasible insertions are dropped; an empty list means a new route has to be opened."""
    k_best_costs = []
    for route in solution:
        insertions = insert_into_every_index_slack(problem_instance, route, customer_node)
        if not insertions:
            continue
        base_feasible, _, base_cost = is_route_feasible(problem_instance, route)
        base_cost = base_cost if base_feasible else 0.0
        for x in insertions:
            feasible, candidate, cost = is_route_feasible(problem_instance, x[1])
            if feasible:
                k_best_costs.append([True, candidate, cost - base_cost])
    return k_best_costs

def insert_into_every_index_slack(problem_instance, input_list, item):
    #problem_instance.earliest_times_feasible[node.index] = service_starts[i]
    #problem_instance.latest_times_feasible[node.index] = latest_times_feasible[i]
    # Add possibility of route with one location
    feasible_solutions = [] 
    initial_cost = get_min_multi_trip_cost(problem_instance, input_list)
    
    for i in range(len(input_list) + 1):
        candidate = []
        if i == 0:
            prev_node = problem_instance.depot
            next_node = input_list[0]
        elif i == len(input_list):
            prev_node = input_list[-1]
            next_node = problem_instance.depot
        else:   
            prev_node = input_list[i-1]
            next_node = input_list[i]
        service_start = max(item.start_interval, problem_instance.earliest_times_feasible[prev_node.index] + prev_node.unloading_time + problem_instance.distance_matrix[prev_node.index][item.index])
        if service_start > item.end_interval:
            continue
        next_arrival = service_start + item.unloading_time + problem_instance.distance_matrix[item.index][next_node.index]
        if next_arrival <= problem_instance.latest_times_feasible[next_node.index]:
            candidate = input_list[:i] + [item] + input_list[i:]
            distance_cost = get_min_multi_trip_cost(problem_instance, candidate) - initial_cost            
            
            feasible_solutions.append([True, candidate, distance_cost]) 
   
    #each insertion: [True, candidate route, increase of the minimum multi-trip distance cost]
    return feasible_solutions

def get_min_multi_trip_cost(problem_instance, route):
    
    arcs_no_trip = MathModels.generate_arcs(route)

    arcs_with_trip = [satellite_detour(problem_instance, i, j, "distance_matrix")[0]
                       for [i,j] in arcs_no_trip if j!=0]

    min_transfer = math.ceil((sum([node.demand for node in route]) )/problem_instance.capacity_SEV) 
    arcs_with_trip.sort()

    #num_bins = min_transfer + problem_instance.min_relaxed_transfer_num
    #num_bins = min(len(route), num_bins)
    #bin_size = len(route) // num_bins
    #cutting_points = [i * bin_size for i in range(0, num_bins)]    

    min_satellite_visit_cost =  sum(arcs_with_trip[:min_transfer]) 

    #min_satellite_visit_cost =  sum(arcs_with_trip[x] for x in cutting_points) 

    
    arc_cost = sum(problem_instance.distance_matrix[i][j] for [i,j] in arcs_no_trip)
    cost_v = problem_instance.cost_unit_travel*(min_satellite_visit_cost + arc_cost)
    cost_trips =  problem_instance.trip_unit_cost*min_transfer
    
    return cost_v   + cost_trips #+ problem_instance.cost_vehicle_use

def update_intervals(problem_instance, routes):
    problem_instance.earliest_times_feasible = [problem_instance.depot.start_interval]*(len(problem_instance.nodes))
    problem_instance.latest_times_feasible = [problem_instance.depot.end_interval]*(len(problem_instance.nodes))
    problem_instance.arrivals = [problem_instance.depot.end_interval]*(len(problem_instance.nodes))
    for route in routes:
        service_starts, latest_times_feasible,arrivals = get_latest_times(problem_instance, route)
        for i, node in enumerate(route):
            problem_instance.earliest_times_feasible[node.index] = service_starts[i]
            problem_instance.latest_times_feasible[node.index] = latest_times_feasible[i]
            problem_instance.arrivals[node.index] = arrivals[i]
    return
def get_latest_times(problem_instance, route):    
    #departure and return
    indices = [node.index for node in route]
    indices.insert(0,0)
    indices.append(0)
    size = len(indices) #- 1

    arrivals = [0 for i in range(len(indices) )]

    service_starts = [problem_instance.nodes[ind].start_interval for ind in indices]       
    unloading_times = [problem_instance.nodes[ind].unloading_time for ind in indices]
    latest_times_feasible = [problem_instance.nodes[ind].end_interval for ind in indices] 
    distances = [problem_instance.distance_matrix[indices[i]][indices[i+1]] for i in range(size-1)]
    
    for i in range(1,size):
        arrivals[i] = service_starts[i - 1] + distances[i - 1] + unloading_times[i - 1]
        service_starts[i] = max(service_starts[i - 1] + distances[i - 1] + unloading_times[i - 1], service_starts[i])
    
    for i in range(size-1, 0, -1):
        latest_times_feasible[i - 1] = min(latest_times_feasible[i] - distances[i-1] - unloading_times[i - 1], latest_times_feasible[i -1])
    
    slack_times = [latest_times_feasible[i] - service_starts[i] for i in range(size)]

    if min(slack_times) < -0.001:
        return False, route, BigNumber, BigNumber, BigNumber   
        
    service_starts.pop(0)
    latest_times_feasible.pop(0)
    service_starts.pop()
    latest_times_feasible.pop()
    arrivals.pop(0)
    arrivals.pop()

    return service_starts, latest_times_feasible, arrivals

def contains_subset(main_array, sub_array):
    it = iter(main_array)
    return 
def is_in_tabu_list(problem_instance, route):
    
    list_in = [x.index for x in route]

    for ii,tabu in enumerate(problem_instance.tabu_list):
        
        it = iter(list_in)
       
        if all(x in it for x in tabu):            
            return True
    return False

def is_route_feasible(problem_instance, route):      
           
    
    if is_in_tabu_list(problem_instance, route):
        return False, route, BigNumber
    
    vrp_cost = problem_instance.cost_unit_travel*sum(problem_instance.distance_matrix[route[i].index][route[i+1].index] for i in range(len(route)-1))
    vrp_cost += problem_instance.cost_unit_travel*problem_instance.distance_matrix[route[-1].index][0] 
    vrp_cost += problem_instance.cost_unit_travel*problem_instance.distance_matrix[0][route[0].index] #return to depot and departure from depot

    if REFILL_RULE in ("two_way", "optimal", "optimal_cost"):
        #refill points chosen by the DP:
        #  "two_way":      minimum number of transshipments that is on time (K* >= m), then the
        #                  cheapest plan with K* (two_way_transshipment_checking)
        #  "optimal":      exactly the capacity minimum m (rejects routes that need m+1)
        #  "optimal_cost": cheapest plan with any number >= m (the street MIP's plan)
        if REFILL_RULE == "two_way":
            optimal = two_way_transshipment_checking(problem_instance, route)
        elif REFILL_RULE == "optimal":
            optimal = bothway_optimal_checking(problem_instance, route)
        else:
            optimal = optimal_transshipment_checking(problem_instance, route)
        if REFILL_RULE != "two_way": #two_way_transshipment_checking checks the return itself
            returns_in_time(problem_instance, route, optimal)
        if not optimal["feasible"]:
            return False, route, BigNumber
        if ROUTE_CHECK == "mip":
            feas, cost, visits, route_return = check_route_transshipments(problem_instance, route)
            return feas, route, cost
        #exact travel + transshipment cost of the route (vehicle cost excluded, as get_min_multi_trip_cost)
        return True, route, optimal["cost"] - problem_instance.cost_vehicle_use

    trans_points = refill_points(problem_instance, route)
    current_time = problem_instance.depot.start_interval #vehicles leave the garage as early as needed

    prev_node_index = 0
    # Calculate penalties for arriving late
    for ind, node in enumerate(route):
        #first node is always a transfer point, so add transfer time if it's the first node
        current_time += problem_instance.closest_satellite_network.distance_matrix[prev_node_index][node.index]            
        if node.index in trans_points: #transfer point
            current_time += problem_instance.closest_satellite_network.check_arc_cost(prev_node_index, node.index)
            current_time += problem_instance.TransferDuration  # Add transfer time if it's a transfer point
        
        
        if current_time < node.start_interval:
            current_time = node.start_interval  # Wait until the start_interval time
        
        elif current_time > node.end_interval: # If hard TW is violated
            return False, route, BigNumber
        
        current_time += node.unloading_time  # Add service time
        prev_node_index = node.index       
    
    
    end = current_time
    feas = True
    if end > problem_instance.nodes[0].end_interval:
        #return False, route, BigNumber
        opt_tri = mip_checking(problem_instance, route)[0]
        if len(opt_tri) == 0:            
            return False, route, BigNumber
    # Get the distances for each route segment, starting and ending at 0
    
    
    feas, cost, visits, route_return = check_route_transshipments(problem_instance, route)
    return feas, route, cost 

def remove_infeasible_nodes_slack(problem_instance, route):
    pool = []    
    initial_served = len(route)
    

    removed = True
    while removed:
        service_starts, latest_times_feasible, arrivals = get_latest_times(problem_instance, route)
        slacks = [[route[i].index, latest_times_feasible[i] - arrivals[i]] for i in range(len(service_starts))]
        slacks.sort(key=lambda x: x[1], reverse = False)
        
        removed = False
        if len(pool) + len(route) != initial_served:
            print("stop")
        feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, route)
        if feasibility:
            break
        
        if len(pool) + len(route) != initial_served:
            print("stop")
        ii = slacks[0][0]    
        node =[x for x in route if x.index == ii][0]       
        sequence_init = [problem_instance.customers[x.index] for x in route if x.index != ii]
        #sequence_init.pop(ii)
        feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, sequence_init)
            
        if not feasibility:            
            route.remove(node)
            pool.append(node)     
            removed = True
            if len(pool) + len(route) != initial_served:
                print("stop")                      
    
    
    return pool
def remove_infeasible_nodes_binary(problem_instance, route, pool = []):
    #loop breaking
    removed_bef = [x.index for x in pool]
    route = [x for x in route if x.index not in removed_bef] #remove the nodes removed before
    feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, route)
    if feasibility:
        return pool #breaking point for the recurive function
    
    start = 1
    
    if len(route) > 2:
        half = len(route)//2    
        removed_right =  route[:half]  #remove left part
        removed_left =  route[half:] #remove right part

        before = len(pool)
        remove_infeasible_nodes(problem_instance, removed_right, pool)   

             
        start = len(removed_right) - 1 #index first removed
        remove_infeasible_nodes(problem_instance, removed_left, pool)     

        if len(pool) > before: #any change?

            removed_bef = [x.index for x in pool] #remove
            
            route = [x for x in route if x.index not in removed_bef]

            
            feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, route)
            if feasibility:
                return pool #breaking point for the recurive function
        
    
    initial_served = len(route) 
        
    for ii, node in enumerate(route):
        if ii < start:
            continue
        sequence_init = copy.deepcopy(route[:ii+1])
        #sequence_init.pop(ii)
        feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, sequence_init)
        if not feasibility:                
            route.remove(node)
            pool.append(node)                
            
            if len(pool) + len(route) != initial_served:
                print("stop")
            return remove_infeasible_nodes(problem_instance, route, pool)   
    
    return pool

def remove_infeasible_nodes(problem_instance, route):
    pool = []
    
    initial_served = len(route)
    start = 0
    
    if len(route) > 4:
        divisions = [0.25, 0.5, 0.75, 0.9, 0.95]

        divisions = [ int(len(route)*x) for x in divisions]

        divisions = set(divisions)
        for length in divisions:
            seq_left = route[:length]
            feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, seq_left)
            if feasibility:
                start = length
            else:
                break
    
    removed = True
    while removed:
        removed = False
        if len(pool) + len(route) != initial_served:
            print("stop")
        feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, route)
        if feasibility:
            break
        
        if len(pool) + len(route) != initial_served:
            print("stop")
        for ii, node in enumerate(route):
            if ii < start:
                continue
            sequence_init = copy.deepcopy(route[:ii+1])
            #sequence_init.pop(ii)
            feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, sequence_init)
            if not feasibility:                
                route.remove(node)
                pool.append(node)                
                start = ii
                removed = True
                if len(pool) + len(route) != initial_served:
                    print("stop")
                break          
    
    
    return pool

def plot_found_solutions(problem_instance):

    found_unique = []

    for time, sequence in problem_instance.points_found:
        routes = []
        for seq in sequence:
            r = [[node for node in problem_instance.nodes if node.index == ind][0] for ind in seq]
            routes.append(r)
        
        sol = check_VRP_solution(problem_instance, routes)
        found_unique.append([time, sol.travelStreet, sol.nSEVs,  sol.is_feasible])

    problem_instance.points_found = [x for x in found_unique if x[3] == True]

    fleet = [x[2] for x in problem_instance.points_found]
    distance = [x[1] for x in problem_instance.points_found]
    time = [x[0] for x in problem_instance.points_found]
    plt.scatter(y=distance,x=time, c = "black", marker='o' )
    plt.title("relaxed cost of the solutions found over time")
    #plt.scatter(time, fleet, c = "black", marker='X', label='fleet size of the solutions found over time')
    #plt.legend(loc="upper right")
    plt.xlabel('time')
    plt.ylabel('relaxed cost')
    
    plt.savefig(problem_instance.name.split(".")[0]+"_distance_solutions_over.png")
    plt.close()

    plt.scatter( y=fleet,x=time, c = "black", marker='o', label='fleet size of the solutions found over time')
    plt.title("fleet size of the solutions found over time")
    #plt.legend(loc="upper right")
    plt.xlabel('time')
    plt.ylabel('fleet size')
    
    plt.savefig(problem_instance.name.split(".")[0]+"_fleet_solutions_over.png")
    plt.close()

    return
def add_feasible_points(problem_instance, solution_indices):
    found_time = round(time.time() - problem_instance.modelStartTime,2)  
    problem_instance.points_found.append([found_time, solution_indices])
    return

def solution_sequence_key(sequences):
    """Key of a route set (customer-index sequences): the order of the routes does not matter,
    the order of the customers in a route does."""
    return tuple(sorted(tuple(sequence) for sequence in sequences))

def add_to_feasible_solution(problem_instance, solution, tabu_search = False):
    """
    Store a complete route set (all customers served) once: in problem.heuristics_solutions, or in
    problem.tabu_list with tabu_search. Duplicates are found by a set of keys
    (solution_sequence_key) kept next to the list; the list is append-only (its order is the order
    of discovery) and the key set is rebuilt when the list was replaced or changed elsewhere.
    :return: True if the route set was new
    """
    if sum(len(x) for x in solution) != len(problem_instance.customers): #partial solution
        return False
    name = "tabu_list" if tabu_search else "heuristics_solutions"
    list_of_found = getattr(problem_instance, name)
    keys = problem_instance.__dict__.setdefault("_found_keys", {})
    entry = keys.get(name)
    if entry is None or entry[0] is not list_of_found or entry[1] != len(list_of_found):
        entry = [list_of_found, len(list_of_found), set(solution_sequence_key(x) for x in list_of_found)]
        keys[name] = entry
    sequence_current = [[x.index for x in route] for route in solution]
    sequence_current.sort(key=lambda x: x[0])
    key = solution_sequence_key(sequence_current)
    if key in entry[2]:
        return False
    list_of_found.append(sequence_current)
    entry[1] += 1
    entry[2].add(key)
    add_feasible_points(problem_instance, sequence_current)
    return True

def check_found_solutions(problem_instance):     
    solutions_to_scheck = []

    for solution in problem_instance.heuristics_solutions:
        sequence_current = solution
        sequence_current.sort(key=lambda x: x[0])
        if sequence_current not in problem_instance.checked_solutions:
            problem_instance.checked_solutions.append(copy.deepcopy(sequence_current))        
            routes = [[problem_instance.nodes[x] for x in sequence] for sequence in sequence_current]
            current_solution = check_VRP_solution(problem_instance, routes) #relaxed VRP costs
            if current_solution.is_feasible:
                solutions_to_scheck.append([current_solution.objective, routes])

    best_street = math.inf
    best_FEV_number = math.inf
    best_SEV_feasible = math.inf

    if len(solutions_to_scheck) > 0:
        solutions_to_scheck.sort(key=lambda x: len(x[1]))
        
        if problem_instance.best_found != None:
            best_FEV_number = problem_instance.best_found.nFEVs
            best_SEV_feasible = problem_instance.best_found.nSEVs    
    
    multi_trips = []
    for solution in solutions_to_scheck:
        solution_s = check_multi_trips(problem_instance, solution[1])
        multi_trips.append(solution_s)
    solutions_to_scheck = multi_trips
    solutions_to_scheck.sort(key=lambda x: x.objective)
    min_check = max(50, len(solutions_to_scheck)) 
    for solution_mt in solutions_to_scheck[:min_check]:         
        routes = solution_mt.routes
        if problem_instance.minVessel_bound == best_FEV_number and len(routes) > best_SEV_feasible:
            continue

        if problem_instance.minVessel_bound == best_FEV_number and len(routes) > best_SEV_feasible:
            continue
        solution_two_echelon = check_MVRP_solution(problem_instance, routes)  #MVRP costs
        
        if solution_two_echelon.is_feasible:
            obj, routes, solution_two_echelon = solution_two_echelon.objective, solution_two_echelon.routes_sevs, solution_two_echelon
    
            if obj > problem_instance.cutoff - 1:
                solution_two_echelon.checked_optimality = True
                continue
            if problem_instance.minVessel_bound == best_FEV_number and len(routes) > best_SEV_feasible:
                solution_two_echelon.checked_optimality = True
                continue
            solution_two_echelon = check_2EVRP_solution(problem_instance, solution_two_echelon) #two echelon costs
            
            solution_two_echelon.checked_optimality = True
            if solution_two_echelon.is_feasible:      
                before = 1000
                after = 0
                status_before = problem_instance.consolidated
                problem_instance.consolidated = True
                while abs(after - before) > 1 :
                    before = round ( solution_two_echelon.travelTotal,1)
                
                    solution_two_echelon = check_2EVRP_solution_consolidated(problem_instance, solution_two_echelon)
                    
                    after = round ( solution_two_echelon.travelTotal,1) #if new solution is provided

                problem_instance.feas_solutions.append(solution_two_echelon)
                problem_instance.consolidated = status_before
                
                if solution_two_echelon.nFEVs < best_FEV_number:
                    best_FEV_number = solution_two_echelon.nFEVs
                if solution_two_echelon.objective < best_street:
                    best_street =  solution_two_echelon.travelStreet  + problem_instance.cost_vehicle_use*solution_two_echelon.nSEVs
                    if solution_two_echelon.nSEVs < best_SEV_feasible:
                        best_SEV_feasible = solution_two_echelon.nSEVs
            else:
                problem_instance.inf_solutions.append(solution_two_echelon)   
        else:
            problem_instance.inf_solutions.append(solution_two_echelon)   
        
    return

#can be called for VRP, MVRP, and 2e-MVRP
def check_feasibility(problem_instance,  routes,  VRP_only = True, multi_trip = False, single_route_MVRP = False, multi_trip_solution = None): #integrated solution or single

    if VRP_only:
        #add transfers here
        return check_VRP_solution(problem_instance, routes)
       
    elif multi_trip:
        if single_route_MVRP: #SINGLE - PARTIAL MVRP
            feasibility, cost, visits, routes = check_route_transshipments(problem_instance, routes) #single array is sent
            return feasibility, cost
        else: #COMPLETE MVRP
            return check_MVRP_solution(problem_instance, routes)
    
    else:#two echelon
        if not multi_trip_solution.is_feasible:
            return multi_trip_solution
        return check_2EVRP_solution(problem_instance, multi_trip_solution)   
      
def check_multi_trips(problem_instance, routes):   
    current_solution =  sev.two_echelon_solution() 
    for route in routes: 
        newRoute = sev.street_vehicle()
        newRoute.sequence.extend(route)
        feasibility, cost, visits, route = check_route_transshipments(problem_instance, route) #single array is sent
        newRoute.cost = cost
        newRoute.visits = visits
        newRoute.is_feasible = feasibility
        current_solution.routes_sevs.append(newRoute)

    current_solution.routes = routes    
    resultss = [is_route_feasible(problem_instance, route)[0:] for route in routes]
    current_solution.objective =  round(sum(result[2] for result in resultss),2)
    current_solution.travelStreet =  round(sum(result[2] for result in resultss),2)
    current_solution.is_feasible =  all(result[0] for result in resultss)
    current_solution.nSEVs = len(routes)

    return current_solution
  
def check_VRP_solution(problem_instance, routes):   
    current_solution =  sev.two_echelon_solution() 
    for route in routes: 
        newRoute = sev.street_vehicle()
        newRoute.sequence.extend(route)
        current_solution.routes_sevs.append(newRoute)
    current_solution.routes = routes    
    resultss = [is_route_feasible(problem_instance, route)[0:] for route in routes]
    current_solution.objective =  round(sum(result[2] for result in resultss),2)
    current_solution.travelStreet =  round(sum(result[2] for result in resultss),2)
    current_solution.is_feasible =  all(result[0] for result in resultss)
    current_solution.nSEVs = len(routes)

    return current_solution
def check_2EVRP_solution_consolidated(problem_instance, solution_mvrp): 
    if len(problem_instance.satellites) == 1:
        return solution_mvrp
    
    routes = [copy.deepcopy(route.sequence) for route in solution_mvrp.routes_sevs]

    satellites_before = [satelitte for satelitte in problem_instance.satellites]

    satellites_used = [satellite for satellite in problem_instance.satellites if satellite.index in solution_mvrp.satellites_used]
    satellites_consolidated = [satellite for satellite in satellites_before if satellite.index != solution_mvrp.satellite_least_used]

    problem_instance.satellites = satellites_consolidated
    solution = check_multi_trips(problem_instance, routes)
    if not solution.is_feasible:
        problem_instance.satellites = satellites_before
        return solution_mvrp
    solution = check_MVRP_solution(problem_instance, routes)
    if not solution.is_feasible:
        problem_instance.satellites = satellites_before
        return solution_mvrp
    solution_two_echelon = check_2EVRP_solution(problem_instance, solution)

    problem_instance.satellites = satellites_before

    if not solution_two_echelon.is_feasible:
        return solution_mvrp
    else:    
        if solution_two_echelon.objective > solution_mvrp.objective-0.1:    
            return solution_mvrp
        else:
            return solution_two_echelon

def check_MVRP_solution(problem_instance, routes):
    final_solution = sev.two_echelon_solution() #for single routes, no need to generate vehicles every time
    final_solution.objective = BigNumber    
    final_solution.is_feasible = True  

    #check street solution
    #trip generation, street schedules feasibility check spatio-temporal
    assignments = []
    demand_to_supply = 0
    infeasible_nodes = []
    
    #remove infeasible nodes
    for route in routes:
        demand_to_supply += sum([x.demand for x in route])
        feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, route)
        if not feasibility:            
            pool = remove_infeasible_nodes(problem_instance, route)
            infeasible_nodes.extend(pool)
            route = [x for x in route if x not in pool]

            if route:                
                feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, route)
                if feasibility:
                    newRoute = sev.street_vehicle()
                    newRoute.sequence.extend(route) 
                    newRoute.cost = cost
                    newRoute.visits = visits
                    assignments.extend(visits)
                    final_solution.routes_sevs.append(newRoute)
                else:
                    final_solution.is_feasible = False
                    return final_solution

        else:
            newRoute = sev.street_vehicle()
            newRoute.sequence.extend(route) 
            newRoute.cost = cost
            newRoute.visits = visits
            assignments.extend(visits)
            final_solution.routes_sevs.append(newRoute)
    
    #re-insert unserved and repair the solution
    routes = [x.sequence for x in final_solution.routes_sevs]
    if infeasible_nodes:
        print("infeasible nodes", len(infeasible_nodes))
        print(len(infeasible_nodes) + sum(len(x) for x in routes))
        final_solution = sev.two_echelon_solution() #reinitilize
        final_solution.is_feasible = True
        routes, infeasible_nodes = k_best_insertion_repair(problem_instance, routes, infeasible_nodes, 3)
        print(len(infeasible_nodes) + sum(len(x) for x in routes))
        if not infeasible_nodes: #feasible
            assignments = []    
            for route in routes:
                feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, route)
                if not feasibility:
                    final_solution.is_feasible = False
                    return final_solution
                else:
                    newRoute = sev.street_vehicle()
                    newRoute.sequence.extend(route) 
                    newRoute.cost = cost
                    newRoute.visits = visits
                    assignments.extend(visits)
                    final_solution.routes_sevs.append(newRoute)
            
            routes = [x.sequence for x in final_solution.routes_sevs] #update routes
            add_to_feasible_solution(problem_instance, routes)
        else:
            final_solution.is_feasible = False
            return final_solution
    
    #routes = [x.sequence for x in final_solution.routes_sevs]
    #solution is feasible, now generate water level problem
    arcList = []    
    for r in routes:
        arcList.extend(MathModels.generate_arcs(r))
    final_solution.arcList = arcList
    transfer_points = [i for [i,p] in assignments]
    satellites = [p for [i,p] in assignments]
    #get the trip loads
    loads = MathModels.generate_triploads(problem_instance, arcList, transfer_points)
    for ii, trip in enumerate(assignments):
        trip.append(loads[ii])
    sum_trips = sum([x[2] for x in assignments])

    if sum_trips != demand_to_supply:
        print("Error in demand supply check", sum_trips, demand_to_supply)
    final_solution.nSEVs= len(final_solution.routes_sevs)
    final_solution.objective = sum([x.cost for x in final_solution.routes_sevs]) + problem_instance.cost_vehicle_use*final_solution.nSEVs
    final_solution.travelStreet = sum([x.cost for x in final_solution.routes_sevs])
    final_solution.trips = assignments
    final_solution.transfer_points = transfer_points
    final_solution.satellites_used = satellites
    routes = [route.sequence for route in final_solution.routes_sevs] #= routes
    
    served = sum(len(route) for route in routes)
    if served != len(problem_instance.customers):
        print ("stop")
    #place, time, load is found
    tasks = []    
    for ii, trip in enumerate(assignments):
        satellite = next(x for x in problem_instance.satellites if x.index == trip[1]) 
        task = instanceGenerator.Node("S", (satellite.index), satellite.xcoordinate, satellite.ycoordinate, trip[2], satellite.start_interval,satellite.end_interval,
                            satellite.unloading_time)
        task.transfer_point = trip[0]
        task.load = trip[2] #used one
        task.demand = trip[2] #might be needed for simulating operations and also math formulations
        tasks.append(task)
        task.name = satellite.name
    final_solution.transshipments = tasks

    used = [0 for _ in final_solution.satellites]
    used = [sum(x.demand for x in final_solution.transshipments if x.index == satellite.index) for satellite in  problem_instance.satellites]
    
    min_load = min(x for x in used if x > 0)
    index_least = used.index(min_load)
    final_solution.satellite_least_used = problem_instance.satellites[index_least].index
    routes = [route.sequence for route in final_solution.routes_sevs]    

    return final_solution
def check_2EVRP_solution(problem_instance, multi_trip_solution, remove_satellite = False):
    final_solution = multi_trip_solution

    routes = [x.sequence for x in final_solution.routes_sevs]
    #remove below after verifying feasibility of multi trip solution any time called
    hasks = []
    for route in routes:
        feasibility, cost, visits, route_return = check_route_transshipments(problem_instance, route)
        hasks.append(visits)
        if not feasibility:
            print("infeasible solution for multi-trip") #normally should not happen since it is checked before
            final_solution.is_feasible = False
            return final_solution

    tasks = final_solution.transshipments
    for task in tasks:
        print(task.name, task.transfer_point, task.load)

    if sum(x.demand for x in tasks) != sum(x.demand for x in problem_instance.nodes if x.index != 0):
        print("Error in demand supply check", sum(x.demand for x in tasks), sum(x.demand for x in problem_instance.nodes if x.index != 0))
    
    #check water solution
    SE_cost = final_solution.travelStreet + problem_instance.cost_vehicle_use*len(final_solution.routes_sevs)
    feas, obj_value, fleet, distance_travel_cost, used_arcs, locations_arcs = FEVmodels.water_problem_MIP(problem_instance,tasks, routes,  SE_cost)
    final_solution.is_feasible = feas
    if not feas:
        
        return final_solution
    
    final_solution.water_objective = obj_value
    final_solution.water_fleet = fleet
    final_solution.nFEVs = fleet
    final_solution.travelWater = distance_travel_cost
    final_solution.travelTotal = round(final_solution.travelStreet + final_solution.travelWater, 2)
    final_solution.water_used_arcs = used_arcs #operation sequence (temporal location)
    final_solution.routes_fevs = locations_arcs #satellite sequence (spatial location)
    final_solution.objective = (final_solution.travelStreet + final_solution.travelWater
                                + problem_instance.cost_vehicle_use*final_solution.nSEVs
                                + problem_instance.cost_vehicle_use_FEV*final_solution.nFEVs) #eq. (1): z^s + z^w
    final_solution.checked_optimality = True

    routes =  FEVmodels.get_routes(problem_instance, used_arcs)
    sequences  = [[[task for task in tasks if task.transfer_point == i][0] for i in route_curr] for route_curr in routes]
    loads = [sum([task.load for task in sequence]) for sequence in sequences]
    sum_loads = sum(loads)
    sum_tasks = sum([task.demand for task in tasks])
    sum_demand = sum([node.demand for node in problem_instance.customers])
    served_tasks = sum([len(sequence) for sequence in sequences])
    total_tasks = len(tasks)
    
    final_solution.operation_sequence_fevs = sequences
    
    #number of satellites visited
    list_locs = []
    for [i,j] in final_solution.routes_fevs:
        if i not in list_locs:
            list_locs.append(i)

    final_solution.nVisited_locations = len(list_locs) - 1
    
    #update incumbent
    
    if problem_instance.cutoff > final_solution.objective + 0.0001:
        problem_instance.cutoff = final_solution.objective
        updated = False
        if problem_instance.best_found is None:
            problem_instance.best_found = final_solution
        else:
            if final_solution.nFEVs <= problem_instance.best_found.nFEVs:
                 if final_solution.nFEVs < problem_instance.best_found.nFEVs:
                     updated = True
                 elif final_solution.nSEVs < problem_instance.best_found.nSEVs:
                     updated = True
                 elif final_solution.nSEVs == problem_instance.best_found.nSEVs and final_solution.nFEVs == problem_instance.best_found.nFEVs:
                     if final_solution.travelTotal < problem_instance.best_found.travelTotal :
                         updated = True
                 if updated:
                    problem_instance.best_found = final_solution
                    
            
    return final_solution

def check_MVRP_given(problem_instance, routes, assignments):
    final_solution = sev.two_echelon_solution() #reinitilize
    final_solution.is_feasible = True
    for route in routes:        
        feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance, problem_instance.timelimit_SEmodel, route,None, assignments )
        newRoute = sev.street_vehicle()
        newRoute.sequence.extend(route) 
        newRoute.cost = cost
        newRoute.visits = visits
        final_solution.routes_sevs.append(newRoute)

    routes = [x.sequence for x in final_solution.routes_sevs]

    #solution is feasible, now generate water level problem
    arcList = []    
    for r in routes:
        arcList.extend(MathModels.generate_arcs(r))
    final_solution.arcList = arcList
    transfer_points = [i for [i,p] in assignments]
    #get the trip loads
    loads = MathModels.generate_triploads(problem_instance, arcList, transfer_points)
    for ii, trip in enumerate(assignments):
        trip.append(loads[ii])
    
    final_solution.nSEVs= len(final_solution.routes_sevs)
    final_solution.objective = sum([x.cost for x in final_solution.routes_sevs]) + problem_instance.cost_vehicle_use*final_solution.nSEVs
    final_solution.travelStreet = sum([x.cost for x in final_solution.routes_sevs])
    final_solution.trips = assignments
    routes = [route.sequence for route in final_solution.routes_sevs] #= routes
    
    served = sum(len(route) for route in routes)
    if served != len(problem_instance.customers):
        print ("stop")
    #place, time, load is found
    tasks = []    
    for ii, trip in enumerate(assignments):
        satellite = next(x for x in problem_instance.satellites if x.index == trip[1]) 
        task = instanceGenerator.Node("S", (satellite.index), satellite.xcoordinate, satellite.ycoordinate, trip[2], satellite.start_interval,satellite.end_interval,
                            satellite.unloading_time)
        task.transfer_point = trip[0]
        task.load = trip[2] #used one
        task.demand = trip[2] #might be needed for simulating operations
        tasks.append(task)
        task.name = satellite.name
    final_solution.transshipments = tasks
    routes = [route.sequence for route in final_solution.routes_sevs]

    final_solution = check_2EVRP_solution(problem_instance, final_solution)
    return final_solution
def read_initial_SEV_routes_use_case(problem_instance, file_content): #single uncapacitated VRP route
    with open(file_content, 'r') as file:
        line = file.readline()
        
        while line:
            indices = line.strip().split()
            break
    routes = []
    route = []
    for ind in indices:
        node = [node for node in problem_instance.nodes if node.index == int(ind)][0]
        route.append(node)
    routes.append(route)
    return routes
def read_initial_SEV_routes_benchmark(problem_instance, demand_set_given,  usecase = True):
    
    if usecase:
        projectPath = os.path.dirname(os.path.abspath(__file__))
        files_use_case = projectPath+"/usecase/initials/"
        text = "init_demand_set"+str(demand_set_given)+"_"+str(len(problem_instance.customers))+"_best.txt"

        file_content = files_use_case + text
    else:
        file_content = "initials_multi\init_demand_set" + str(problem_instance.name) +"_"+str(len(problem_instance.customers))+ ".txt"
    with open(file_content, 'r') as file:
        line = file.readline() #"problem demand set\t"+ str(problem_instance.name)
        line = file.readline() # number of customers\t"+ str(len(problem_instance.customers)) 
        line = file.readline() #"SEV capacity\t"+ str(problem_instance.capacity_SEV) 
        line = file.readline() # file.write("vehicle capacity\t"+ str(problem_instance.capacity_FEV) + "\n")        
        line = file.readline() # file.write("runtime\t"+ str(runtime) + "\n")
        
        #the routes with customer indices
        line = file.readline()
        sequences_given = line.strip().split()

        routes_given = []
        visited = []
        route_current = []

        for item in sequences_given:
            if item != "-":
                ind = int(item)
                visited.append(ind)
                route_current.append(ind)
            else:
                routes_given.append(route_current)
                route_current = []
    
    initial_routes_asynch = [[problem_instance.nodes[i] for i in route] for route in routes_given]
    add_to_feasible_solution(problem_instance, initial_routes_asynch)
    check_found_solutions(problem_instance)
    return problem_instance.best_found



def write_out_solution(problem_instance, final_solution):
    
    with open("init_demand_set" + str(problem_instance.name) +"_"+str(len(problem_instance.customers))+"_"+str(problem_instance.synch_check)+"_"+str(len(problem_instance.satellites))+ ".txt", 'w') as file:
        file.write("problem demand set\t"+ str(problem_instance.name) + "\n")
        file.write("number of customers\t"+ str(len(problem_instance.customers)) + "\n")
        file.write("SEV capacity\t"+ str(problem_instance.capacity_SEV) + "\n")
        file.write("vehicle capacity\t"+ str(problem_instance.capacity_FEV) + "\n")
        
        runtime = round(time.time() - problem_instance.modelStartTime, 0)
        file.write("runtime\t"+ str(runtime) + "\n")
        
        for route in final_solution.routes_sevs:
            for node in route.sequence:
                file.write(str(node.index) + "\t")
            file.write( "-"+ "\t")
        file.write("\n")
        
        #total demand
        demands = [x.demand for x in problem_instance.customers]
        file.write("total demand\t"+ str(sum(demands)) + "\n")
        #minimum number of trips
        file.write("minimum number of trips\t"+ str(math.ceil(sum(demands)/problem_instance.capacity_SEV)) + "\n")
        file.write("minimum number of vessels\t"+ str(math.ceil(sum(demands)/problem_instance.capacity_FEV)) + "\n")
        loads_trips = [x.load for x in final_solution.transshipments]
        loads_satellites = [sum(x.load for x in final_solution.transshipments if x.index == satellite.index) for satellite in  problem_instance.satellites]
        used_satellites = [sum(1 for x in final_solution.transshipments if x.index == satellite.index) for satellite in  problem_instance.satellites]

        used_satellites = [x for x in used_satellites if x > 0]
        loads_satellites = [x for x in loads_satellites if x > 0]
        file.write("nof trips\t"+ str(len( final_solution.transshipments)) + "\n")
        file.write("load per trip\t"+ str(loads_trips)+ "\n")
        file.write("used satellites\t"+ str(used_satellites)+ "\n") 
        file.write("loads_satellites\t"+ str(loads_satellites)+ "\n") 
        file.write("objective\t"+ str(final_solution.objective)+ "\n") 
        file.write("is feasible\t"+ str(final_solution.is_feasible)+ "\n") 
        file.write("nSEVs\t"+ str(final_solution.nSEVs)+ "\n") 
        file.write("nFEVs\t"+ str(final_solution.nFEVs)+ "\n") 
        file.write("travel street\t"+ str(final_solution.travelStreet)+ "\n") 
        file.write("travel water\t"+ str(final_solution.travelWater)+ "\n") 
        if final_solution.is_feasible:
            file.write("n visited locations\t"+ str(final_solution.nVisited_locations)+ "\n") 

            cumulative = 0
            for route in final_solution.routes_sevs:
                for node in route.sequence:
                    cumulative+=node.demand
                    file.write(str(cumulative) + "\t")
                file.write( "-"+ "\t")
            file.write("\n")

            file.write("demand to supply "+ str(sum(node.demand for node in problem_instance.customers))+ "\n") 
            file.write("supplied demand by SEVs " + str(cumulative)+ "\n") 
            file.write("supplied demand by FEVs " + str(sum(x.demand for x in final_solution.transshipments))+ "\n") 
        
            file.write("n visited satellites by name")
            chosen_id = []
            for id in final_solution.satellites_used:
                if id not in chosen_id:
                    chosen_id.append(id)
                    file.write("-"+ str(problem_instance.network[id].name))
        

    return 