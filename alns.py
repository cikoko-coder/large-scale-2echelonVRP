"""
Adaptive large neighbourhood search (ALNS) on the SEV (street) routes, every route decided by the
two-way check (the cost and rule of local_search.py and RH.two_way_solution_cost).

Solution    a set of SEV routes (lists of customer nodes); its cost is the two-way cost: per route
            vehicle use + c^s (distance + detours) + trip_unit_cost per trip, with the
            transshipments placed by RH.two_way_transshipment_checking (minimum on-time number K*,
            then cheapest). A route that fails the check (or the horizon bound / tabu list) makes
            the solution infeasible; the search only moves between feasible solutions.

Iteration   destroy q customers (q uniform in [q_min, q_max]) with a destroy operator, re-insert
            them with a repair operator, accept the new solution by simulated annealing, update the
            adaptive operator weights. A new global best is improved by the relocate + 2-opt*
            descent (local_search.local_search) before it is stored.

Destroy     random    q customers uniformly
            worst     customers with the largest distance saving, randomized (Ropke & Pisinger, p_worst)
            related   Shaw removal: distance, window opening and demand relatedness to a removed
                      customer, randomized (p_related)
            route     a whole route (small routes more likely: fleet reduction), topped up with the
                      customers nearest to it up to q
            string    strings of consecutive customers around a seed and its neighbours, one string
                      per route (SISR-like): frees a stretch of time on several routes at once
            A route left by a destroy is checked again: if it is not feasible any more (a refill
            plan moved), its customers go to the pool as well.

Repair      greedy        cheapest insertion over all pool customers, repeatedly
            greedy_noise  the same with a random noise on the insertion cost (diversification)
            regret2/3     regret-k insertion: the customer with the largest gap between its best
                          and its k-th best route goes first (looks ahead for customers with few
                          good options, e.g. tight windows)
            sequential    customers in order of their window opening, each into its best place
            Insertion of u into a route: every position passes an O(1) time screen (forward
            earliest end of service / backward latest start of the route, as local_search); the
            ALNS_EXACT_POSITIONS cheapest by distance increase get the exact two-way check (unless
            the cost lower bound of local_search shows they cannot beat the best one found), the best
            of them is the cost of the route. Opening a new route ([u] alone, vehicle cost
            included) is always an option, so a repair fails only when u cannot be served alone.

Acceptance  simulated annealing on the two-way cost. The start temperature is set from the variable
            part of the start cost (distance + detours, without vehicles and trips): a solution
            ALNS_START_WORSE (5%) worse in distance is accepted with probability 1/2; the
            temperature falls geometrically to ALNS_END_TEMPERATURE times the start over the run
            (iterations or time limit, whichever is reached first). A move that adds a vehicle or a
            trip (10000) is in effect never accepted, a move that removes one always is.

Weights     segments of ALNS_SEGMENT iterations; scores sigma1 new global best, sigma2 better than
            the current, sigma3 accepted and not seen before; w <- (1 - r) w + r score / uses.
"""
import math
import random
import time

import Repair_heuristcs as RH
import local_search as LS

# --- parameters -----------------------------------------------------------------------------
ALNS_ITERATIONS = 1000
ALNS_TIME_LIMIT = 120.0          # seconds (None: iterations only)
ALNS_MIN_REMOVE = 0.05           # q_min as a share of the customers (at least 2)
ALNS_MAX_REMOVE = 0.30           # q_max as a share of the customers
ALNS_MAX_REMOVE_ABS = 40         # q_max at most this many customers
ALNS_EXACT_POSITIONS = 3         # positions per route that get the exact two-way check
ALNS_SEGMENT = 50
ALNS_REACTION = 0.1
ALNS_SCORES = (33.0, 9.0, 13.0)  # sigma1, sigma2, sigma3
ALNS_START_WORSE = 0.05
ALNS_END_TEMPERATURE = 0.002
ALNS_LOCAL_SEARCH_BEST = True    # relocate + 2-opt* descent on every new global best
ALNS_ELITE = 5                   # distinct best solutions kept (stored for the two-echelon check)
P_WORST = 3.0
P_RELATED = 6.0
NOISE = 0.025                    # greedy_noise: noise = NOISE * max distance, uniform in [-noise, noise]
RELATED_WEIGHTS = (9.0, 3.0, 2.0)  # Shaw: distance, window opening, demand

DESTROY = ("random", "worst", "related", "route", "string")
REPAIR = ("greedy", "greedy_noise", "regret2", "regret3", "sequential")

EPS = LS.EPS
BIG = RH.BigNumber


class Infeasible(Exception):
    """A customer of the pool cannot be served, not even by a route of its own."""


class ALNS:
    def __init__(self, problem, routes, seed = None, evaluator = None):
        self.problem = problem
        self.rng = random.Random(seed)
        self.evaluator = evaluator or LS.RouteEvaluator(problem)
        routes = [list(route) for route in routes if route]
        # the context of local_search: travel times, detour of the first refill, neighbour lists
        self.ctx = LS.SearchContext(problem, routes, evaluator = self.evaluator)
        self.t = problem.travel_matrix
        self.customers = [node for route in routes for node in route]
        self.n = len(self.customers)
        t = self.t
        self.max_dist = max((t[u.index][v.index] for u in self.customers for v in self.customers), default = 1.0) or 1.0
        opens = [u.start_interval for u in self.customers]
        demands = [u.demand for u in self.customers]
        self.window_range = (max(opens) - min(opens)) or 1.0
        self.demand_range = (max(demands) - min(demands)) or 1.0

    # --- solution helpers -------------------------------------------------------------------
    def route_data(self, nodes):
        return LS.RouteData(self.ctx, list(nodes))

    @staticmethod
    def cost(rds):
        return sum(rd.cost for rd in rds if rd.n)

    @staticmethod
    def key(rds):
        return tuple(sorted(tuple(node.index for node in rd.nodes) for rd in rds if rd.n))

    def q_range(self):
        q_min = max(2, int(round(ALNS_MIN_REMOVE*self.n)))
        q_max = max(q_min, min(ALNS_MAX_REMOVE_ABS, int(round(ALNS_MAX_REMOVE*self.n))))
        return min(q_min, self.n), min(q_max, self.n)

    def relatedness(self, u, v):
        a, b, c = RELATED_WEIGHTS
        return (a*min(self.t[u.index][v.index], self.t[v.index][u.index])/self.max_dist
                + b*abs(u.start_interval - v.start_interval)/self.window_range
                + c*abs(u.demand - v.demand)/self.demand_range)

    # --- destroy ----------------------------------------------------------------------------
    def destroy(self, name, rds, q):
        """:return: (routes as node lists without the removed customers, removed customers)"""
        routes = [list(rd.nodes) for rd in rds if rd.n]
        removed = getattr(self, "_destroy_" + name)(routes, q)
        gone = set(u.index for u in removed)
        return [[u for u in r if u.index not in gone] for r in routes], removed

    def _destroy_random(self, routes, q):
        pool = [u for r in routes for u in r]
        return self.rng.sample(pool, min(q, len(pool)))

    def _destroy_worst(self, routes, q):
        t = self.t
        saving = []
        for r in routes:
            for pos, u in enumerate(r):
                a = r[pos - 1].index if pos > 0 else 0
                b = r[pos + 1].index if pos + 1 < len(r) else 0
                saving.append((t[a][u.index] + t[u.index][b] - t[a][b], u.index, u))
        saving.sort(key = lambda x: (-x[0], x[1]))
        removed = []
        while saving and len(removed) < q:
            k = int(len(saving)*self.rng.random()**P_WORST)
            removed.append(saving.pop(k)[2])
        return removed

    def _destroy_related(self, routes, q):
        pool = [u for r in routes for u in r]
        removed = [pool.pop(self.rng.randrange(len(pool)))]
        while pool and len(removed) < q:
            ref = removed[self.rng.randrange(len(removed))]
            pool.sort(key = lambda v: (self.relatedness(ref, v), v.index))
            removed.append(pool.pop(int(len(pool)*self.rng.random()**P_RELATED)))
        return removed

    def _destroy_route(self, routes, q):
        if len(routes) < 2:
            return self._destroy_related(routes, q)
        weights = [1.0/len(r) for r in routes]
        r = self.rng.choices(range(len(routes)), weights = weights, k = 1)[0]
        removed = list(routes[r])
        if len(removed) < q:
            rest = [u for k, route in enumerate(routes) if k != r for u in route]
            rest.sort(key = lambda v: (min(self.t[u.index][v.index] for u in removed), v.index))
            removed += rest[:q - len(removed)]
        return removed

    def _destroy_string(self, routes, q):
        where = {u.index: (k, pos) for k, r in enumerate(routes) for pos, u in enumerate(r)}
        seed = self.rng.choice([u for r in routes for u in r])
        max_len = max(2, int(round(2*q/max(1, len(routes)))))  # strings of about 2 q / routes
        done, removed = set(), []
        for v in [seed] + self.ctx.neighbours[seed.index]:
            if len(removed) >= q:
                break
            k, pos = where[v.index]
            if k in done:
                continue
            done.add(k)
            route = routes[k]
            length = min(len(route), max_len, q - len(removed))
            start = self.rng.randint(max(0, pos - length + 1), min(pos, len(route) - length))
            removed += route[start:start + length]
        return removed

    # --- insertion --------------------------------------------------------------------------
    def best_insertion(self, u, rd):
        """Cheapest insertion of u into route rd: (delta two-way cost, new nodes) or None."""
        t, ctx = self.t, self.ctx
        ui = u.index
        options = []
        for q in range(rd.n + 1):
            ai = rd.nodes[q - 1].index if q > 0 else 0
            b = rd.nodes[q] if q < rd.n else None
            bi = b.index if b is not None else 0
            finish = rd.earliest[q - 1] if q > 0 else ctx.start
            service = max(finish + t[ai][ui], u.start_interval)
            if service > u.end_interval + EPS:
                continue
            if b is not None and max(service + u.unloading_time + t[ui][bi], b.start_interval) > rd.latest[q] + EPS:
                continue
            options.append((t[ai][ui] + t[ui][bi] - t[ai][bi], q))
        options.sort()
        best = None
        load = rd.load + u.demand
        for delta, q in options[:ALNS_EXACT_POSITIONS]:
            # cost bound of local_search (vehicle + c^s (distance + first detour) + trips by load):
            # a position whose bound does not beat the best exact insertion of this route is skipped
            if best is not None and ctx.lower_bound(rd.n + 1, rd.dist + delta, u if q == 0 else rd.nodes[0],
                                                    load) - rd.cost >= best[0]:
                continue
            nodes = rd.nodes[:q] + [u] + rd.nodes[q:]
            feasible, cost = self.evaluator.evaluate(nodes)
            if feasible and (best is None or cost - rd.cost < best[0]):
                best = (cost - rd.cost, nodes)
        return best

    def alone(self, u):
        feasible, cost = self.evaluator.evaluate([u])
        return (cost, [u]) if feasible else None

    def repair(self, name, rds, pool):
        """Insert every customer of pool into rds (a list of RouteData, changed in place)."""
        pool = list(pool)
        if name == "sequential":
            pool.sort(key = lambda u: (u.start_interval, u.end_interval, u.index))
            for u in pool:
                self._insert(rds, u, self._options(rds, u, {}))
            return rds
        k = {"greedy": 1, "greedy_noise": 1, "regret2": 2, "regret3": 3}[name]
        noise = NOISE*self.max_dist if name == "greedy_noise" else 0.0
        cache = {u.index: {} for u in pool}   # u -> route id -> best insertion (None: infeasible)
        while pool:
            choice = None
            for u in pool:
                opts = self._options(rds, u, cache[u.index])
                if not opts:
                    raise Infeasible(u.index)
                costs = [c for c, _, _ in opts]
                if k == 1:
                    value = -(costs[0] + (noise*(2*self.rng.random() - 1) if noise else 0.0))
                else:
                    value = sum((costs[h] if h < len(costs) else BIG) - costs[0] for h in range(1, k))
                tie = -costs[0]
                if choice is None or (value, tie) > choice[0]:
                    choice = ((value, tie), u, opts)
            _, u, opts = choice
            r = self._insert(rds, u, opts)
            pool.remove(u)
            del cache[u.index]
            for c in cache.values():  # the changed route has to be evaluated again
                c.pop(r, None)
        return rds

    def _options(self, rds, u, cache):
        """Insertion options of u sorted by cost: (delta, route id or None for a new route, nodes)."""
        opts = []
        for r, rd in enumerate(rds):
            if r not in cache:
                cache[r] = self.best_insertion(u, rd) if rd.n else None
            if cache[r] is not None:
                opts.append((cache[r][0], r, cache[r][1]))
        if "new" not in cache:
            cache["new"] = self.alone(u)
        if cache["new"] is not None:
            opts.append((cache["new"][0], None, cache["new"][1]))
        opts.sort(key = lambda o: (o[0], -1 if o[1] is None else o[1]))
        return opts

    def _insert(self, rds, u, opts):
        if not opts:
            raise Infeasible(u.index)
        _, r, nodes = opts[0]
        if r is None:
            rds.append(self.route_data(nodes))
            return len(rds) - 1
        rds[r] = self.route_data(nodes)
        return r

    # --- one iteration ----------------------------------------------------------------------
    def neighbour(self, rds, destroy, repair):
        q_min, q_max = self.q_range()
        q = self.rng.randint(q_min, q_max)
        routes, removed = self.destroy(destroy, rds, q)
        new = []
        for nodes in routes:
            if not nodes:
                continue
            rd = self.route_data(nodes)
            if rd.feasible:
                new.append(rd)
            else:   # removing customers moved the refill plan out of time: free the whole route
                removed += nodes
        try:
            self.repair(repair, new, removed)
        except Infeasible:
            return None
        return [rd for rd in new if rd.n]

    def local_search(self, rds):
        routes, report = LS.local_search(self.problem, [rd.nodes for rd in rds], seed = self.rng.random(),
                                         max_rounds = 5, store = False, evaluator = self.evaluator)
        if report["improvement"] > LS.MIN_GAIN:
            return [self.route_data(r) for r in routes], report
        return rds, report

    # --- main loop --------------------------------------------------------------------------
    def run(self, iterations = ALNS_ITERATIONS, time_limit = ALNS_TIME_LIMIT, log = None):
        started = time.time()
        current = [self.route_data(rd.nodes) for rd in self.ctx.routes]
        report = {"start_cost": self.cost(current), "start_sevs": len(current),
                  "feasible_start": all(rd.feasible for rd in current)}
        if not report["feasible_start"] or self.n < 2:
            report.update(final_cost = report["start_cost"], final_sevs = len(current), iterations = 0,
                          best_iteration = 0, time = time.time() - started, improvement = 0.0, elite = [],
                          accepted = 0, improved = 0, failed = 0, ls_moves = 0,
                          weights = {}, uses = {}, successes = {}, dp_runs = 0, cache_hits = 0)
            return [list(rd.nodes) for rd in current], report

        best = current
        cur_cost = best_cost = self.cost(current)
        variable = RH.two_way_solution_cost(self.problem, [rd.nodes for rd in current])
        variable = variable["cost"] - self.problem.cost_vehicle_use*variable["n_vehicles"] \
            - self.problem.trip_unit_cost*variable["n_trips"]
        t_start = max(1e-6, -ALNS_START_WORSE*variable/math.log(0.5))
        temperature = t_start

        ops = {"destroy": list(DESTROY), "repair": list(REPAIR)}
        weights = {g: {o: 1.0 for o in ops[g]} for g in ops}
        seg_score = {g: {o: 0.0 for o in ops[g]} for g in ops}
        seg_uses = {g: {o: 0 for o in ops[g]} for g in ops}
        uses = {g: {o: 0 for o in ops[g]} for g in ops}
        successes = {g: {o: 0 for o in ops[g]} for g in ops}
        seen = {self.key(current)}
        elite = {self.key(current): (cur_cost, [list(rd.nodes) for rd in current])}
        stats = {"accepted": 0, "improved": 0, "failed": 0, "ls_moves": 0, "best_iteration": 0}

        it = 0
        for it in range(1, iterations + 1):
            elapsed = time.time() - started
            if time_limit is not None and elapsed >= time_limit:
                it -= 1
                break
            progress = it/iterations
            if time_limit:
                progress = max(progress, elapsed/time_limit)
            temperature = t_start*ALNS_END_TEMPERATURE**progress

            d = self.rng.choices(ops["destroy"], weights = [weights["destroy"][o] for o in ops["destroy"]])[0]
            r = self.rng.choices(ops["repair"], weights = [weights["repair"][o] for o in ops["repair"]])[0]
            for g, o in (("destroy", d), ("repair", r)):
                uses[g][o] += 1
                seg_uses[g][o] += 1

            candidate = self.neighbour(current, d, r)
            score = 0.0
            if candidate is None:
                stats["failed"] += 1
            else:
                cost = self.cost(candidate)
                key = self.key(candidate)
                if cost < best_cost - LS.MIN_GAIN:
                    if ALNS_LOCAL_SEARCH_BEST:
                        candidate, ls = self.local_search(candidate)
                        stats["ls_moves"] += sum(ls["moves"].values())
                        cost, key = self.cost(candidate), self.key(candidate)
                    best, best_cost, current, cur_cost = candidate, cost, candidate, cost
                    stats["best_iteration"] = it
                    score = ALNS_SCORES[0]
                    if log:
                        log(f"  ALNS iteration {it}: new best {round(cost, 1)}, {len(candidate)} SEVs ({d} + {r})")
                elif cost < cur_cost - LS.MIN_GAIN:
                    current, cur_cost = candidate, cost
                    score = ALNS_SCORES[1]
                    stats["improved"] += 1
                elif key not in seen and self.rng.random() < math.exp(-(cost - cur_cost)/temperature):
                    current, cur_cost = candidate, cost
                    score = ALNS_SCORES[2]
                    stats["accepted"] += 1
                if score > 0:
                    seen.add(key)
                    for g, o in (("destroy", d), ("repair", r)):
                        successes[g][o] += 1
                if key not in elite and (len(elite) < ALNS_ELITE or cost < max(c for c, _ in elite.values())):
                    elite[key] = (cost, [list(rd.nodes) for rd in candidate])
                    if len(elite) > ALNS_ELITE:
                        del elite[max(elite, key = lambda k: elite[k][0])]
            for g, o in (("destroy", d), ("repair", r)):
                seg_score[g][o] += score

            if it % ALNS_SEGMENT == 0:
                for g in ops:
                    for o in ops[g]:
                        if seg_uses[g][o]:
                            weights[g][o] = (1 - ALNS_REACTION)*weights[g][o] \
                                + ALNS_REACTION*seg_score[g][o]/seg_uses[g][o]
                        weights[g][o] = max(weights[g][o], 0.05)
                        seg_score[g][o], seg_uses[g][o] = 0.0, 0

        out = [list(rd.nodes) for rd in best]
        assert sorted(u.index for r in out for u in r) == sorted(u.index for u in self.customers), \
            "ALNS lost or duplicated a customer"
        report.update(stats)
        report.update(final_cost = best_cost, final_sevs = len(out), iterations = it,
                      time = time.time() - started, improvement = report["start_cost"] - best_cost,
                      start_temperature = t_start, end_temperature = temperature,
                      weights = weights, uses = uses, successes = successes,
                      dp_runs = self.evaluator.dp_runs, cache_hits = self.evaluator.calls - self.evaluator.dp_runs,
                      elite = sorted(elite.values(), key = lambda x: x[0]))
        return out, report


def alns(problem, routes, seed = None, iterations = ALNS_ITERATIONS, time_limit = ALNS_TIME_LIMIT,
         store = True, log = None):
    """
    ALNS from a feasible route set (every route passes the two-way check; otherwise the start is
    returned unchanged with report["feasible_start"] False).
    store: the ALNS_ELITE best distinct route sets go to problem.heuristics_solutions
    (RH.add_to_feasible_solution) for the two-echelon check (check_found_solutions).
    :return: (best routes, report)
    """
    search = ALNS(problem, routes, seed = seed)
    out, report = search.run(iterations = iterations, time_limit = time_limit, log = log)
    if store and report["feasible_start"]:
        for _, routes_e in report["elite"]:
            RH.add_to_feasible_solution(problem, routes_e)
    return out, report


def format_report(report):
    if not report.get("iterations"):
        return "ALNS not run (start not feasible)" if not report.get("feasible_start", True) else "ALNS: no iterations"
    lines = [f"two-way cost {round(report['start_cost'], 1)} -> {round(report['final_cost'], 1)} "
             f"(-{round(report['improvement'], 1)}), SEVs {report['start_sevs']} -> {report['final_sevs']}, "
             f"{report['iterations']} iterations in {report['time']:.1f}s, last best at iteration {report['best_iteration']}",
             f"  accepted: improving {report['improved']}, worse (SA) {report['accepted']}, failed repairs {report['failed']}; "
             f"local search moves on new bests {report['ls_moves']}; temperature {report['start_temperature']:.2f} -> "
             f"{report['end_temperature']:.3f}; exact two-way checks: DP runs {report['dp_runs']}, cache hits {report['cache_hits']}"]
    for g in ("destroy", "repair"):
        lines.append(f"  {g}: " + ", ".join(
            f"{o} w={report['weights'][g][o]:.2f} used {report['uses'][g][o]} success {report['successes'][g][o]}"
            for o in report["weights"][g]))
    lines.append("  elite (two-way cost, SEVs): " + ", ".join(
        f"{round(c, 1)}/{len(r)}" for c, r in report["elite"]))
    return "\n".join(lines)
