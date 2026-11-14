"""
Local search on the SEV (street) routes: relocate and 2-opt* (tail exchange), every move decided
by the two-way check.

Revised versions of the two operators of Multi_level_destroy (perform_N_relocation and
perform_N_two_point_crossover / one_point_crossover):

* relocate  - a customer u leaves its route X and is inserted into a route Y (Y = X allowed),
              next to one of its nearest neighbours (before or after it).
* 2-opt*    - tail exchange of two routes X, Y at a new arc u -> v (u in X, v in Y):
              X' = X[..u] + Y[v..], Y' = Y[..v) + X(u..]. This is the one-point crossover with
              both cut points chosen; an empty head or tail merges the two routes (fleet
              reduction).

Two-way check. A move is accepted only if both new routes pass the exact route check of the
two-way rule (RH.two_way_transshipment_checking: the minimum number of on-time transshipments K*,
then the cheapest plan with K*, the vehicle back at the garage in time) and the tabu list, i.e. the
same rule as RH.is_route_feasible with REFILL_RULE = "two_way". The cost compared is the two-way
cost of RH.two_way_solution_cost: vehicle use + c^s (distance + detours) + trip_unit_cost per trip,
so a move that empties a route gains the vehicle cost.
Before that exact DP, the time windows are screened in both directions in O(1):
  forward  earliest[pos]: end of service at pos in the fastest schedule (one refill before the
           first customer, the relaxation of refill_time_bounds: refills only add time);
  backward latest[pos]:   latest start of service at pos from which the rest of the route is on
           time without refills (an upper bound of the real latest start).
A joined sequence head + tail with  max(open, earliest_head + t(head end, tail start)) > latest_tail
is late in every refill plan, so the screen never removes a feasible move.

Data structures that cut the work per move:
  RouteData    per route: positions, prefix distance, prefix load, earliest / latest, the
               exact two-way cost; rebuilt in O(n_route) only for the two routes a move changes.
  where        customer -> (route id, position): O(1) location of a neighbour.
  neighbours   k nearest customers per customer (granular neighbourhood): O(n k) candidate
               moves per pass instead of O(n^2) (relocate) / O(n^2) pairs of cut points (2-opt*).
  lower bound  vehicle + c^s (distance + detour of the first refill) + trip_unit_cost *
               ceil(load / Q) of each new route, from the prefix arrays in O(1): a move whose two
               bounds do not beat the current cost of the two routes is never sent to the DP.
  cache        exact route checks by customer sequence (tuple): a sequence is checked once per
               search, however often a move produces it again.
"""
import math
import random
import time

import Repair_heuristcs as RH

EPS = RH.TIME_EPS
MIN_GAIN = 1e-3          # a move must lower the two-way cost by more than this
NEIGHBOURS = 20          # size of the granular neighbour lists (k nearest customers)
MAX_PASSES = 50          # passes over all customers per operator before giving up
STOP_ON_STAGNATION = True  # True: an operator stops after a pass without improvement, the descent after a
#                            round without moves; False: always max_passes passes and max_rounds rounds


class RouteEvaluator:
    """Exact two-way route check with a cache keyed by the customer sequence."""

    def __init__(self, problem):
        self.problem = problem
        self.cache = {}
        self.calls = 0      # exact checks requested
        self.dp_runs = 0    # exact checks actually computed (cache misses)
        self._tabu_size = len(getattr(problem, "tabu_list", []) or [])

    def evaluate(self, route):
        """(feasible, two-way cost incl. vehicle use); the empty route costs 0."""
        self.calls += 1
        if not route:
            return True, 0.0
        tabu = getattr(self.problem, "tabu_list", []) or []
        if len(tabu) != self._tabu_size:  # a new tabu sequence can make a cached route infeasible
            self.cache.clear()
            self._tabu_size = len(tabu)
        key = tuple(node.index for node in route)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        self.dp_runs += 1
        p = self.problem
        if RH.is_in_tabu_list(p, route):
            out = (False, RH.BigNumber)
        else:
            plan = RH.two_way_transshipment_checking(p, route)
            out = (True, plan["cost"]) if plan["feasible"] else (False, RH.BigNumber)
        self.cache[key] = out
        return out


class RouteData:
    """Prefix / suffix arrays of one route (positions 0..n-1, garage = node 0)."""
    __slots__ = ("nodes", "n", "dist_pre", "dist", "load_pre", "load", "earliest", "latest",
                 "feasible", "cost")

    def __init__(self, ctx, nodes):
        t = ctx.t
        self.nodes = nodes
        n = self.n = len(nodes)
        self.dist_pre = []      # distance garage -> nodes[pos] along the route
        self.load_pre = [0]     # load of nodes[:pos]
        d, prev = 0.0, 0
        for node in nodes:
            d += t[prev][node.index]
            self.dist_pre.append(d)
            self.load_pre.append(self.load_pre[-1] + node.demand)
            prev = node.index
        self.dist = d + t[prev][0] if n else 0.0
        self.load = self.load_pre[-1]
        # forward: fastest schedule, refill before the first customer only
        self.earliest = []
        if n:
            finish = ctx.start + ctx.det_start[nodes[0].index] + ctx.transfer
            prev = 0
            for node in nodes:
                finish = max(finish + t[prev][node.index], node.start_interval) + node.unloading_time
                self.earliest.append(finish)
                prev = node.index
        # backward: latest start of service without refills
        self.latest = [node.end_interval for node in nodes]
        for pos in range(n - 2, -1, -1):
            self.latest[pos] = min(self.latest[pos], self.latest[pos + 1]
                                   - t[nodes[pos].index][nodes[pos + 1].index] - nodes[pos].unloading_time)
        self.feasible, self.cost = ctx.evaluator.evaluate(nodes)

    def tail_dist(self, pos):
        """Distance nodes[pos] -> ... -> garage (pos = n: 0)."""
        return self.dist - self.dist_pre[pos] if pos < self.n else 0.0


class SearchContext:
    def __init__(self, problem, routes, neighbours = NEIGHBOURS, evaluator = None):
        self.problem = problem
        self.t = problem.travel_matrix
        self.start = problem.depot.start_interval
        self.transfer = problem.TransferDuration
        self.capacity = problem.capacity_SEV
        self.unit = problem.cost_unit_travel
        self.trip = problem.trip_unit_cost
        self.vehicle = problem.cost_vehicle_use
        self.evaluator = evaluator or RouteEvaluator(problem)
        customers = [node for route in routes for node in route]
        t = self.t
        sats = [s.index for s in problem.satellites]
        # detour of the first refill (garage -> satellite -> first customer)
        self.det_start = {c.index: max(0.0, min(t[0][s] + t[s][c.index] - t[0][c.index] for s in sats))
                          for c in customers}
        # granular neighbour lists: k nearest customers (symmetric travel time)
        k = min(neighbours, len(customers) - 1)
        self.neighbours = {}
        for u in customers:
            others = sorted((min(t[u.index][v.index], t[v.index][u.index]), v.index, v) for v in customers if v is not u)
            self.neighbours[u.index] = [v for _, _, v in others[:k]]
        self.routes = [RouteData(self, list(route)) for route in routes if route]
        self.where = {}
        for r in range(len(self.routes)):
            self._index(r)

    def _index(self, r):
        for pos, node in enumerate(self.routes[r].nodes):
            self.where[node.index] = (r, pos)

    def replace(self, r, nodes, data = None):
        self.routes[r] = data if data is not None else RouteData(self, nodes)
        self._index(r)

    def lower_bound(self, n_nodes, dist, first, load):
        """Lower bound of the two-way cost of a route with these figures (0 for an empty route)."""
        if n_nodes == 0:
            return 0.0
        return (self.vehicle + self.unit*(dist + self.det_start[first.index])
                + self.trip*math.ceil(load/self.capacity - 1e-12))

    def total_cost(self):
        return sum(rd.cost for rd in self.routes if rd.n)

    def solution(self):
        return [list(rd.nodes) for rd in self.routes if rd.n]


def joins_in_time(ctx, head, i, tail, j):
    """Two-way screen of head.nodes[:i] + tail.nodes[j:] (O(1)): False when the first tail
    customer is late in every refill plan."""
    if j >= tail.n:
        return True
    v = tail.nodes[j]
    if i == 0:
        finish, u = ctx.start, 0  # lower bound: no detour
    else:
        finish, u = head.earliest[i - 1], head.nodes[i - 1].index
    return max(finish + ctx.t[u][v.index], v.start_interval) <= tail.latest[j] + EPS


def try_relocate(ctx, u, stats):
    """Best relocation of customer u next to one of its neighbours; applies it if it improves."""
    t = ctx.t
    rx, p = ctx.where[u.index]
    X = ctx.routes[rx]
    prev = X.nodes[p - 1].index if p > 0 else 0
    nxt = X.nodes[p + 1].index if p + 1 < X.n else 0
    removal_dist = X.dist - t[prev][u.index] - t[u.index][nxt] + t[prev][nxt]
    first_x = X.nodes[1] if p == 0 and X.n > 1 else X.nodes[0]
    lb_x = ctx.lower_bound(X.n - 1, removal_dist, first_x, X.load - u.demand)

    candidates = set()
    for v in ctx.neighbours[u.index]:
        ry, q = ctx.where[v.index]
        candidates.add((ry, q))      # u before v
        candidates.add((ry, q + 1))  # u after v
    best = None
    for ry, q in candidates:
        Y = ctx.routes[ry]
        if ry == rx:
            if q == p or q == p + 1:
                continue
            nodes = X.nodes[:p] + X.nodes[p + 1:]
            q2 = q if q < p else q - 1
            new_y = nodes[:q2] + [u] + nodes[q2:]
            a = new_y[q2 - 1].index if q2 > 0 else 0
            b = new_y[q2 + 1].index if q2 + 1 < len(new_y) else 0
            dist = removal_dist + t[a][u.index] + t[u.index][b] - t[a][b]
            lb = ctx.lower_bound(len(new_y), dist, new_y[0], X.load)
            old = X.cost
            if lb >= old - MIN_GAIN:
                stats["pruned_bound"] += 1
                continue
            stats["exact"] += 1
            feasible, cost = ctx.evaluator.evaluate(new_y)
            if feasible and cost < old - MIN_GAIN and (best is None or old - cost > best[0]):
                best = (old - cost, rx, new_y, None, None)
            continue
        a = Y.nodes[q - 1] if q > 0 else None
        b = Y.nodes[q] if q < Y.n else None
        ai, bi = (a.index if a else 0), (b.index if b else 0)
        # two-way time screen: forward arrival at u, then at b against its backward latest start
        finish = Y.earliest[q - 1] if q > 0 else ctx.start
        service_u = max(finish + t[ai][u.index], u.start_interval)
        if service_u > u.end_interval + EPS:
            stats["pruned_time"] += 1
            continue
        if b is not None and max(service_u + u.unloading_time + t[u.index][bi], b.start_interval) > Y.latest[q] + EPS:
            stats["pruned_time"] += 1
            continue
        dist_y = Y.dist + t[ai][u.index] + t[u.index][bi] - t[ai][bi]
        lb = lb_x + ctx.lower_bound(Y.n + 1, dist_y, u if q == 0 else Y.nodes[0], Y.load + u.demand)
        old = X.cost + Y.cost
        if lb >= old - MIN_GAIN:
            stats["pruned_bound"] += 1
            continue
        new_x = X.nodes[:p] + X.nodes[p + 1:]
        new_y = Y.nodes[:q] + [u] + Y.nodes[q:]
        stats["exact"] += 1
        fy, cy = ctx.evaluator.evaluate(new_y)
        if not fy:
            continue
        fx, cx = ctx.evaluator.evaluate(new_x)
        if fx and cx + cy < old - MIN_GAIN and (best is None or old - cx - cy > best[0]):
            best = (old - cx - cy, ry, new_y, rx, new_x)
    if best is None:
        return False
    _, ry, new_y, rx2, new_x = best
    ctx.replace(ry, new_y)
    if rx2 is not None:
        ctx.replace(rx2, new_x)
    return True


def try_two_opt_star(ctx, u, stats):
    """Best tail exchange at a new arc u -> v (v a neighbour of u in another route)."""
    t = ctx.t
    rx, p = ctx.where[u.index]
    X = ctx.routes[rx]
    i = p + 1                     # X' = X[:i] + Y[j:], Y' = Y[:j] + X[i:]
    best = None
    for v in ctx.neighbours[u.index]:
        ry, j = ctx.where[v.index]
        if ry == rx:
            continue
        Y = ctx.routes[ry]
        if not joins_in_time(ctx, X, i, Y, j):
            stats["pruned_time"] += 1
            continue
        if i < X.n and not joins_in_time(ctx, Y, j, X, i):
            stats["pruned_time"] += 1
            continue
        nx, ny = i + Y.n - j, j + X.n - i
        dist_x = X.dist_pre[i - 1] + t[u.index][v.index] + Y.tail_dist(j)
        if j > 0:
            w = Y.nodes[j - 1].index
            dist_y = Y.dist_pre[j - 1] + (t[w][X.nodes[i].index] + X.tail_dist(i) if i < X.n else t[w][0])
        else:
            dist_y = t[0][X.nodes[i].index] + X.tail_dist(i) if i < X.n else 0.0
        first_y = Y.nodes[0] if j > 0 else (X.nodes[i] if i < X.n else None)
        lb = (ctx.lower_bound(nx, dist_x, X.nodes[0], X.load_pre[i] + Y.load - Y.load_pre[j])
              + ctx.lower_bound(ny, dist_y, first_y, Y.load_pre[j] + X.load - X.load_pre[i]))
        old = X.cost + Y.cost
        if lb >= old - MIN_GAIN:
            stats["pruned_bound"] += 1
            continue
        new_x = X.nodes[:i] + Y.nodes[j:]
        new_y = Y.nodes[:j] + X.nodes[i:]
        stats["exact"] += 1
        fx, cx = ctx.evaluator.evaluate(new_x)
        if not fx:
            continue
        fy, cy = ctx.evaluator.evaluate(new_y)
        if fy and cx + cy < old - MIN_GAIN and (best is None or old - cx - cy > best[0]):
            best = (old - cx - cy, rx, new_x, ry, new_y)
    if best is None:
        return False
    _, rx, new_x, ry, new_y = best
    ctx.replace(rx, new_x)
    ctx.replace(ry, new_y)
    return True


OPERATORS = {"relocate": try_relocate, "two_opt_star": try_two_opt_star}


def run_operator(ctx, name, rng, max_passes = MAX_PASSES, stats = None):
    """First improvement over the customers in random order (best move per customer), repeated
    until a full pass improves nothing. :return: number of applied moves"""
    move = OPERATORS[name]
    stats = stats if stats is not None else new_stats()
    applied = 0
    for _ in range(max_passes):
        order = [node for rd in ctx.routes for node in rd.nodes]
        rng.shuffle(order)
        improved = False
        for u in order:
            if move(ctx, u, stats):
                applied += 1
                improved = True
        stats["passes"][name] += 1
        if not improved and STOP_ON_STAGNATION:
            break
    stats["moves"][name] += applied
    return applied


def new_stats():
    return {"moves": {k: 0 for k in OPERATORS}, "passes": {k: 0 for k in OPERATORS},
            "exact": 0, "pruned_time": 0, "pruned_bound": 0}


def local_search(problem, routes, operators = ("relocate", "two_opt_star"), seed = None,
                 max_rounds = 20, max_passes = MAX_PASSES, neighbours = NEIGHBOURS, store = True,
                 evaluator = None):
    """
    Variable neighbourhood descent over the operators (in the given order) until no operator
    improves the two-way cost or max_rounds is reached.
    Every route of the start must pass the two-way check (an infeasible route has no cost to
    improve): otherwise the start is returned unchanged with report["feasible_start"] False.
    store: the improved complete route set goes to problem.heuristics_solutions
    (RH.add_to_feasible_solution) so check_found_solutions sees it at the two-echelon level.
    evaluator: a RouteEvaluator to share its cache of exact route checks (e.g. with the ALNS).
    :return: (routes, report) with report = start / final cost, number of SEVs, moves, exact
             checks, pruned moves, cache hits, time
    """
    started = time.time()
    rng = random.Random(seed)
    routes = [list(route) for route in routes if route]
    ctx = SearchContext(problem, routes, neighbours, evaluator)
    report = {"start_cost": ctx.total_cost(), "start_sevs": len(ctx.routes),
              "feasible_start": all(rd.feasible for rd in ctx.routes)}
    stats = new_stats()
    rounds = 0
    if report["feasible_start"]:
        for rounds in range(1, max_rounds + 1):
            if sum(run_operator(ctx, name, rng, max_passes, stats) for name in operators) == 0 and STOP_ON_STAGNATION:
                break
    out = ctx.solution()
    served = sum(len(r) for r in out)
    assert served == sum(len(r) for r in routes), "local search lost or duplicated a customer"
    report.update(stats)
    report.update(final_cost = ctx.total_cost(), final_sevs = len(out), rounds = rounds,
                  dp_runs = ctx.evaluator.dp_runs, cache_hits = ctx.evaluator.calls - ctx.evaluator.dp_runs,
                  time = time.time() - started)
    report["improvement"] = report["start_cost"] - report["final_cost"]
    if store and report["feasible_start"]:
        RH.add_to_feasible_solution(problem, out)
    return out, report


def format_report(report):
    return (f"two-way cost {round(report['start_cost'], 1)} -> {round(report['final_cost'], 1)} "
            f"(-{round(report['improvement'], 1)}), SEVs {report['start_sevs']} -> {report['final_sevs']}, "
            f"moves relocate {report['moves']['relocate']} / 2-opt* {report['moves']['two_opt_star']}, "
            f"rounds {report['rounds']}, exact two-way checks {report['exact']} "
            f"(DP runs {report['dp_runs']}, cache hits {report['cache_hits']}), "
            f"pruned by the time screen {report['pruned_time']}, by the cost bound {report['pruned_bound']}, "
            f"{report['time']:.2f}s")
