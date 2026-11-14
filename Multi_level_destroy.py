import random 
import numpy
import SEVs 
import math 
import time
import copy
import MathModels 
import Repair_heuristcs as RH
import local_search as LS
import SEVs
BigNumber = 1000000

# --- ITS (manuscript Section 5, Figure 4) ---------------------------------------------------
# One call of Solve_by_decomposition is one ITS iteration on the SE level (its_iter = k):
#   k = 1: Init (multi_start_initial, N_init starts) -> process the pool -> R* = best,
#          LS(R*) (Local_search_incumbent, R_LS repetitions, N_LS) -> process -> R*
#   every k: ALNS(R*, N_ALNS) (Solve_by_multi_level_ALNS: MLD destroy + the single repair
#          RH.k_best_insertion_repair) -> LS on the best ALNS solution -> process the pool -> R*
# R* = problem.best_SE (multi-trip solution), compared by SE fleet size first, then two-way cost.
# Diversification falls with k: initial scores S_I = S_R = 10/k and the SA rate p_acc/k; the
# caller raises N_LS with k. All solutions found are stored (problem.heuristics_solutions);
# the FE level is optimized afterwards on the min-fleet ones (Optimize_FE).
# The route-level exact check (SolveFixedSEVRouteMILP) is RH.check_route_transshipments: the
# two-way refill DP (ROUTE_CHECK = "dp") or the street MIP of the fixed route ("mip").

SA_ACCEPTANCE = 0.05               # p_acc: temperature = p_acc / its_iter * travel cost of the ALNS start
SCORE_SAME, SCORE_IMPROVING, SCORE_WORSE = 0.1, 1.0, 0.0   # alpha_0, alpha_1, alpha_-1 of eq. (24)
SCORE_SAME_TOLERANCE = 1.0         # |Delta z| <= 1: insignificant change
INITIAL_SCORE_DIVERSIFICATION = 10.0   # S_I (equal selection) at its_iter = 1 (divided by its_iter), S_W = S_N = 1
ROUTE_REMOVAL_INITIAL_SCORE = 20.0     # S_R (route removal) at its_iter = 1 (divided by its_iter)
EQUAL_INITIAL_SCORES = False   # True: every operator of every level starts at score 1 (no diversification bias);
#                                False: S_I = INITIAL_SCORE_DIVERSIFICATION / its_iter, S_R = ROUTE_REMOVAL_INITIAL_SCORE / its_iter
REMOVAL_FRACTIONS = [0.05, 0.10, 0.25, 0.35]   # additional removals A, B, C, D (share of the remaining customers)
REPAIR_OPERATOR = "k_best_insertion"           # the single repair operator (Algorithm 7)
RESET_SCORES = False           # True: operator scores re-initialized at every ITS iteration (S_I = S_R = 10/k);
#                                False: initialized once per instance (k = 1), then carried over between ITS iterations
RECORD_SCORES = False          # True: snapshot of the operator scores after every ALNS iteration in problem.score_history


def solution_key(problem_instance, routes):
    """(SE fleet size, two-way cost) of a route set: the order used for R* (fleet first)."""
    two_way = RH.two_way_solution_cost(problem_instance, routes)
    return (len([r for r in routes if r]), two_way["cost"]), two_way


def travel_cost(problem_instance, two_way):
    """Travel part of a two-way cost: c^s (distance + detours), without vehicle and trip costs."""
    return two_way["cost"] - problem_instance.cost_vehicle_use*two_way["n_vehicles"] \
        - problem_instance.trip_unit_cost*two_way["n_trips"]


def set_best_SE(problem_instance, routes, phase):
    """Make routes the incumbent R* if better (fleet, then two-way cost); returns True if updated."""
    if routes is None or sum(len(r) for r in routes) != len(problem_instance.customers):
        return False
    key, two_way = solution_key(problem_instance, routes)
    if not two_way["feasible"]:
        return False
    best = getattr(problem_instance, "best_SE_key", None)
    if best is not None and key >= (best[0], best[1] - LS.MIN_GAIN):
        return False
    solution = RH.check_feasibility(problem_instance, routes, VRP_only = False, multi_trip = True, single_route_MVRP = False)
    if not solution.is_feasible:
        return False
    set_two_way_cost(problem_instance, solution)
    problem_instance.best_SE = solution
    problem_instance.best_SE_key = (solution.nSEVs, solution.two_way_cost)
    problem_instance.best_SE_phase = phase
    return True


def process_neighborhood(problem_instance, fleet_max = None):
    """
    Algorithm 1 ProcessNeighborhood on the solutions of problem.heuristics_solutions not processed
    before (the pool of the last phase; RH.add_to_feasible_solution re-sorts the list, so they are
    recognized by their sequences): keep those with at most fleet_max routes, update the tabu list by
    the route-level exact check of every route (UpdateTabuList, Algorithm 3) and repair the schedules
    with infeasible routes (RepairInfeasibleSchedules, Algorithm 2: check_MVRP_solution removes the
    customers leading to infeasibility and reinserts them by the repair, the repaired schedule is
    stored). :return: list of multi-trip feasible route sets
    """
    processed = problem_instance.__dict__.setdefault("processed_solutions", set())
    feasible = []
    for sequences in list(problem_instance.heuristics_solutions):
        key = tuple(sorted(tuple(sequence) for sequence in sequences))
        if key in processed:
            continue
        processed.add(key)
        if fleet_max is not None and len(sequences) > fleet_max:
            continue
        routes = [[problem_instance.nodes[x] for x in sequence] for sequence in sequences]
        solution = RH.check_MVRP_solution(problem_instance, routes)
        if solution.is_feasible:
            feasible.append([list(route.sequence) for route in solution.routes_sevs])
    return feasible


def Solve_by_decomposition(random_num, problem_instance, nstart = 5, repLS = 5, iterLS = 500, iterALNS = 500):
    """
    One ITS iteration (problem_instance.its_iter = k) on the SE level, see the outline above.
    Appends a record of the phases to problem_instance.its_log. :return: R* (problem.best_SE)
    """
    random.seed(random_num + problem_instance.its_iter)
    problem_instance.customers = [node for node in problem_instance.nodes if node.index > 0]
    log = {"its_iter": problem_instance.its_iter}
    start_time = time.time()

    if getattr(problem_instance, "best_SE", None) is None:
        # Init: N_init starts, Algorithm 4 (the repaired schedules are stored)
        t = time.time()
        multi_start_initial(problem_instance, nstart)
        for routes in process_neighborhood(problem_instance):
            set_best_SE(problem_instance, routes, "init")
        log["init"] = {"time": time.time() - t, "key": problem_instance.best_SE_key}
        # LS on the cheapest initial solution
        t = time.time()
        _, routes = Local_search_incumbent(problem_instance, problem_instance.best_SE, repLS, iterLS)
        log["init_ls_report"] = problem_instance.local_search_report
        for found in [routes] + process_neighborhood(problem_instance):
            set_best_SE(problem_instance, found, "init+ls")
        log["init_ls"] = {"time": time.time() - t, "key": problem_instance.best_SE_key}

    # ALNS from R*
    t = time.time()
    stored = len(problem_instance.heuristics_solutions)
    best_alns, alns_report = Solve_by_multi_level_ALNS(problem_instance, problem_instance.best_SE, iterALNS)
    log["alns_report"] = alns_report
    log["alns"] = {"time": time.time() - t, "stored": len(problem_instance.heuristics_solutions) - stored}
    log["alns_best_key"] = alns_report["best_key"]
    # LS on the best ALNS solution
    t = time.time()
    _, routes = Local_search_incumbent(problem_instance, best_alns, repLS, iterLS)
    log["ls_report"] = problem_instance.local_search_report
    fleet_max = problem_instance.best_SE_key[0]
    candidates = [routes, best_alns] + process_neighborhood(problem_instance, fleet_max)
    for found in candidates:
        set_best_SE(problem_instance, found, f"its {problem_instance.its_iter}")
    log["ls"] = {"time": time.time() - t, "key": problem_instance.best_SE_key}
    log["time"] = time.time() - start_time
    log["tabu_list"] = len(problem_instance.tabu_list)
    problem_instance.its_log = getattr(problem_instance, "its_log", []) + [log]
    return problem_instance.best_SE


def Solve_by_multi_level_ALNS(problem_instance, start_solution, max_iterations = 500):
    """
    Algorithm 6: in each iteration an MLD destroy (multi_direction_random_destroy_repair with the
    three levels chosen by Select_operators), the single repair (RH.k_best_insertion_repair,
    Algorithm 7), the score update of every level (eq. 24) and the simulated annealing acceptance
    with the aspiration criterion on the SE fleet size. The temperature is fixed during the run at
    SA_ACCEPTANCE / its_iter times the travel cost of the start. Every complete solution of the repair
    is stored by the repair (S_ALNS). Costs are two-way costs (RH.two_way_solution_cost).
    :return: (best route set: fleet first, then cost; report dict)
    """
    current = _routes_of(start_solution)
    if RESET_SCORES or getattr(problem_instance, "scores_roulette_selection_type", None) is None:
        generate_operators_statistics(problem_instance)
    current_key, current_two_way = solution_key(problem_instance, current)
    best, best_key = current, current_key
    best_fleet = current_key[0]
    temperature = max(1e-6, SA_ACCEPTANCE/problem_instance.its_iter*travel_cost(problem_instance, current_two_way))
    report = {"start_key": current_key, "temperature": temperature, "iterations": 0, "accepted_improving": 0,
              "accepted_worse": 0, "aspiration": 0, "rejected": 0, "infeasible": 0, "best_iteration": 0}
    for iteration in range(1, max_iterations + 1):
        report["iterations"] = iteration
        roulette_type, removal_type, removal_number, repair_type = Select_operators(problem_instance)
        partial_solution, pool = multi_direction_random_destroy_repair(problem_instance, current, roulette_type,
                                                                       removal_type, removal_number)
        new_solution, pool = RH.k_best_insertion_repair(problem_instance, partial_solution, pool)
        new_key, new_two_way = solution_key(problem_instance, new_solution)
        if not new_two_way["feasible"]:
            update_scores(problem_instance, roulette_type, removal_type, removal_number, repair_type, SCORE_WORSE)
            report["infeasible"] += 1
            record_scores(problem_instance, iteration, (roulette_type, removal_type, removal_number), SCORE_WORSE)
            continue
        score = calculate_score(current_key[1], new_key[1])
        update_scores(problem_instance, roulette_type, removal_type, removal_number, repair_type, score)
        record_scores(problem_instance, iteration, (roulette_type, removal_type, removal_number), score)

        if new_key[0] < best_fleet:                               # aspiration: fewer SEVs than ever
            accept = True
            best_fleet = new_key[0]
            report["aspiration"] += 1
        elif new_key[0] > current_key[0]:
            accept = False
        elif new_key[1] < current_key[1]:
            accept = True
            report["accepted_improving"] += 1
        else:
            accept = simulated_annealing_acceptance(new_key[1], current_key[1], temperature)
            report["accepted_worse"] += accept
        if not accept:
            report["rejected"] += 1
            continue
        current, current_key = [list(route) for route in new_solution], new_key
        if current_key < best_key:
            best, best_key = current, current_key
            report["best_iteration"] = iteration
    report["best_key"] = best_key
    report["scores"] = {
        "roulette": dict(zip(problem_instance.operators_roulette_selection_type, problem_instance.scores_roulette_selection_type)),
        "removal": dict(zip(problem_instance.operators_removal_type, problem_instance.scores_removal_type)),
        "size": dict(zip(problem_instance.operators_removal_number, problem_instance.scores_removal_number))}
    report["used"] = {
        "roulette": dict(zip(problem_instance.operators_roulette_selection_type, problem_instance.nof_used_roulette_selection_type)),
        "removal": dict(zip(problem_instance.operators_removal_type, problem_instance.nof_used_removal_type)),
        "size": dict(zip(problem_instance.operators_removal_number, problem_instance.nof_used_removal_number))}
    return best, report


def record_scores(problem_instance, iteration, chosen, score):
    """RECORD_SCORES: append the scores of every MLD level after this ALNS iteration to problem.score_history."""
    if not RECORD_SCORES:
        return
    problem_instance.__dict__.setdefault("score_history", []).append({
        "its_iter": problem_instance.its_iter, "iteration": iteration, "chosen": list(chosen), "score": score,
        "roulette": list(problem_instance.scores_roulette_selection_type),
        "removal": list(problem_instance.scores_removal_type),
        "size": list(problem_instance.scores_removal_number)})


def simulated_annealing_acceptance(new_cost, current_cost, temperature):
    """Accept a non-improving solution with probability exp(-(new - current) / temperature)."""
    change = new_cost - current_cost
    if change <= 0:
        return True
    return random.random() < math.exp(-change/temperature)


def multi_direction_random_destroy_repair(problem_instance, solution, roulette_type, removal_type, removal_number):
    """
    MLD destroy (three levels): roulette_type "equal" (identical, I) or "weighted" (W); seed
    removal of a whole route ("route", R) or one customer ("node", N), under W biased to routes with
    a high cost per customer / customers with a high removal saving; additional removal of
    removal_number of the remaining customers, under W biased to customers close to the removed ones.
    :return: (partial solution, removed customers in removal order)
    """
    pool = []
    new_solution = [list(route) for route in solution]
    customers_to_select = [node for route in new_solution for node in route]

    # level 2: seed removal
    if removal_type == "route" and len(new_solution) > 1:
        if roulette_type == "equal":
            weights = [1 for _ in new_solution]
        else:
            weights = get_weights_for_route_removal(problem_instance, new_solution)
        route_removed = random.choices(new_solution, weights = weights, k = 1)[0]
        pool.extend(route_removed)
        new_solution.remove(route_removed)
        customers_to_select = [node for node in customers_to_select if node not in route_removed]
    elif len(customers_to_select) > 1:
        if roulette_type == "equal":
            weights = [1 for _ in customers_to_select]
        else:
            weights = get_weights_for_node_removal(problem_instance, customers_to_select, new_solution)
        node_removed = random.choices(customers_to_select, weights = weights, k = 1)[0]
        pool.append(node_removed)
        customers_to_select.remove(node_removed)

    # level 3: additional removals, a share of the remaining customers
    num_customers_to_remove = max(int(len(customers_to_select)*removal_number), 1)
    for _ in range(num_customers_to_remove):
        if len(customers_to_select) <= 1:  # leave at least one customer in the remaining routes
            break
        if roulette_type == "equal":
            weights = [1 for _ in customers_to_select]
        else:
            weights = get_weights_for_nodes_removal(customers_to_select, pool)
        node_removed = random.choices(customers_to_select, weights = weights, k = 1)[0]
        pool.append(node_removed)
        customers_to_select.remove(node_removed)

    removed = set(node.index for node in pool)
    new_solution = [[node for node in route if node.index not in removed] for route in new_solution]
    new_solution = [route for route in new_solution if route]
    assert sum(len(route) for route in new_solution) + len(pool) == sum(len(route) for route in solution)
    return new_solution, pool


def generate_operators_statistics(problem_instance, destroy_in = None):
    """Operators and initial scores of the MLD levels. EQUAL_INITIAL_SCORES: every operator starts
    at 1; otherwise (manuscript Section 6.1) S_I = INITIAL_SCORE_DIVERSIFICATION / its_iter,
    S_R = ROUTE_REMOVAL_INITIAL_SCORE / its_iter,
    S_W = S_N = 1, every neighbour size 1; one repair."""
    k = problem_instance.its_iter
    first = 1.0 if EQUAL_INITIAL_SCORES else INITIAL_SCORE_DIVERSIFICATION/k
    route_first = 1.0 if EQUAL_INITIAL_SCORES else ROUTE_REMOVAL_INITIAL_SCORE/k
    problem_instance.selection_counts = [1]*len(problem_instance.customers)
    problem_instance.selection_counts.insert(0, 0)

    problem_instance.operators_roulette_selection_type = ["equal", "weighted"]
    problem_instance.scores_roulette_selection_type = [first, 1.0]
    problem_instance.nof_used_roulette_selection_type = [0, 0]
    problem_instance.nof_success_roulette_selection_type = [0, 0]

    problem_instance.operators_removal_type = ["route", "node"]
    problem_instance.scores_removal_type = [route_first, 1.0]
    problem_instance.nof_used_removal_type = [0, 0]
    problem_instance.nof_success_removal_type = [0, 0]

    problem_instance.operators_removal_number = list(REMOVAL_FRACTIONS)
    problem_instance.scores_removal_number = [1.0]*len(REMOVAL_FRACTIONS)
    problem_instance.nof_used_removal_number = [0]*len(REMOVAL_FRACTIONS)
    problem_instance.nof_success_removal_number = [0]*len(REMOVAL_FRACTIONS)

    problem_instance.operators_repair_type = [REPAIR_OPERATOR]
    problem_instance.scores_repair_type = [1.0]
    problem_instance.nof_used_repair_type = [0]
    problem_instance.nof_success_repair_type = [0]
    return


def Select_operators(problem_instance):
    """Roulette-wheel choice of each MLD level by its scores (the repair is the single operator)."""
    roulette_type = random.choices(problem_instance.operators_roulette_selection_type, problem_instance.scores_roulette_selection_type, k=1)[0]
    removal_type = random.choices(problem_instance.operators_removal_type, problem_instance.scores_removal_type, k=1)[0]
    removal_number = random.choices(problem_instance.operators_removal_number, problem_instance.scores_removal_number, k=1)[0]
    repair_type = problem_instance.operators_repair_type[0]

    problem_instance.nof_used_roulette_selection_type[problem_instance.operators_roulette_selection_type.index(roulette_type)] += 1
    problem_instance.nof_used_removal_type[problem_instance.operators_removal_type.index(removal_type)] += 1
    problem_instance.nof_used_removal_number[problem_instance.operators_removal_number.index(removal_number)] += 1
    problem_instance.nof_used_repair_type[0] += 1
    return roulette_type, removal_type, removal_number, repair_type


def calculate_score(cost_old, cost_new):
    """Eq. (24): alpha_0 for an insignificant change (|Delta z| <= 1), alpha_1 if improving,
    alpha_-1 if deteriorating (two-way cost)."""
    if abs(cost_new - cost_old) <= SCORE_SAME_TOLERANCE:
        return SCORE_SAME
    return SCORE_IMPROVING if cost_new < cost_old else SCORE_WORSE


def update_scores(problem_instance, roulette_type, removal_type, removal_number, repair_type, score):
    problem_instance.scores_roulette_selection_type[problem_instance.operators_roulette_selection_type.index(roulette_type)] += score
    problem_instance.scores_removal_type[problem_instance.operators_removal_type.index(removal_type)] += score
    problem_instance.scores_removal_number[problem_instance.operators_removal_number.index(removal_number)] += score
    problem_instance.scores_repair_type[problem_instance.operators_repair_type.index(repair_type)] += score
    if score > 0:
        problem_instance.nof_success_roulette_selection_type[problem_instance.operators_roulette_selection_type.index(roulette_type)] += 1
        problem_instance.nof_success_removal_type[problem_instance.operators_removal_type.index(removal_type)] += 1
        problem_instance.nof_success_removal_number[problem_instance.operators_removal_number.index(removal_number)] += 1
        problem_instance.nof_success_repair_type[problem_instance.operators_repair_type.index(repair_type)] += 1
    return


def get_weights_for_route_removal(problem_instance, solution):
    """Weighted route removal: two-way cost per customer (vehicle cost included, so small and
    expensive routes are removed more likely)."""
    weights = []
    for route in solution:
        plan = RH.two_way_transshipment_checking(problem_instance, route)
        cost = plan["cost"] if plan["feasible"] else BigNumber
        weights.append(cost/len(route))
    return weights


def get_weights_for_node_removal(problem_instance, customers_to_select, solution):
    """Weighted node removal: distance saving of removing the customer from its route,
    d(prev, u) + d(u, next) - d(prev, next) (garage = 0), plus a small floor."""
    d = problem_instance.distance_matrix
    saving = {}
    for route in solution:
        for pos, node in enumerate(route):
            a = route[pos - 1].index if pos > 0 else 0
            b = route[pos + 1].index if pos + 1 < len(route) else 0
            saving[node.index] = d[a][node.index] + d[node.index][b] - d[a][b]
    return [max(saving[node.index], 0.0) + 1e-3 for node in customers_to_select]


def get_weights_for_nodes_removal(customers_to_select, pool):
    """Weighted additional removal: inverse mean Euclidean distance to the removed customers."""
    weights = []
    for node_p in customers_to_select:
        mean = sum(math.sqrt((node_p.xcoordinate - node_s.xcoordinate)**2 + (node_p.ycoordinate - node_s.ycoordinate)**2)
                   for node_s in pool)/len(pool)
        weights.append(1/max(mean, 1e-6))
    total = sum(weights)
    return [x/total for x in weights]


def Optimize_FE(problem_instance, max_candidates = None, route_check = "dp"):
    """
    Algorithm 8 on S*_SE: every stored route set with the fewest SEVs (solutions with more SEVs are
    discarded; ranked by two-way cost, at most max_candidates, None = all) goes through
    RH.check_found_solutions. Its route-level checks run with RH.ROUTE_CHECK = route_check: "dp"
    keeps the two-way plans of the search (minimum transshipments, then TWO_WAY_OBJECTIVE); "mip"
    re-optimizes every route by the fixed-route street MILP (SolveFixedSEVRouteMILP, memoized), which
    places the transshipments at minimum SE cost without the minimum-transshipment bound. Then the multi-trip / MVRP check, the FE routing MILP (FEVmodels.water_problem_MIP)
    and the satellite-removal consolidation, repeated while it improves. Incumbents are compared by
    the hierarchical objective (RH.hierarchical_key: FEVs, SEVs, total distance).
    :return: (problem.best_found, number of SE solutions sent to the FE level)
    """
    ranked = []
    for sequences in problem_instance.heuristics_solutions:
        routes = [[problem_instance.nodes[x] for x in sequence] for sequence in sequences]
        key, two_way = solution_key(problem_instance, routes)
        if two_way["feasible"]:
            ranked.append((key, sequences))
    if not ranked:
        return problem_instance.best_found, 0
    fewest = min(key[0] for key, _ in ranked)
    ranked = sorted([x for x in ranked if x[0][0] == fewest], key = lambda x: x[0])
    if max_candidates is not None:
        ranked = ranked[:max_candidates]
    problem_instance.heuristics_solutions = [sequences for _, sequences in ranked]
    incumbent = problem_instance.best_found
    problem_instance.fe_fleet_bound = incumbent.nFEVs if incumbent is not None else math.inf
    before = RH.ROUTE_CHECK
    RH.ROUTE_CHECK = route_check
    try:
        RH.check_found_solutions(problem_instance)
    finally:
        RH.ROUTE_CHECK = before
    return problem_instance.best_found, len(ranked)


# Local search: the two operators (relocate, 2-opt* = tail exchange, the revised one-point
# crossover) live in local_search.py; every move is decided by the two-way check with the
# forward / backward time screen, the cost lower bound, granular neighbour lists and a route cache.
def _routes_of(solution):
    if type(solution) != list: # a two-echelon / multi-trip object
        return [list(route.sequence) for route in solution.routes_sevs]
    return [list(route) for route in solution]

def perform_N_two_point_crossover(problem_instance, routes_in, N, points = 1):
    """2-opt* (tail exchange) descent, at most N passes. :return: (relative change of the two-way cost, routes)"""
    return _run_operators(problem_instance, routes_in, ("two_opt_star",), N)

def perform_N_relocation(problem_instance, current_solution, N):
    """Relocate descent, at most N passes. :return: (relative change of the two-way cost, routes)"""
    return _run_operators(problem_instance, current_solution, ("relocate",), N)

def _run_operators(problem_instance, routes_in, operators, passes):
    routes = _routes_of(routes_in)
    if len(routes) < 2 or passes < 1:
        return 0, routes
    routes, report = LS.local_search(problem_instance, routes, operators = operators, max_rounds = 1,
                                     max_passes = passes, seed = random.random())
    return (report["final_cost"] - report["start_cost"])/report["start_cost"], routes

def Local_search_incumbent(problem_instance, current_solution_in, replications = 5,  iters = 50):
    """
    Relocate + 2-opt* descent (local_search.local_search) on the SEV routes of the incumbent:
    rounds of both operators until neither improves the two-way cost (at most `replications`
    rounds, `iters` passes per operator and round). The improved route set is stored for the
    two-echelon check. :return: (relative change of the two-way cost, routes)
    """
    routes = _routes_of(current_solution_in)
    routes, report = LS.local_search(problem_instance, routes, max_rounds = max(1, replications),
                                     max_passes = max(1, iters), seed = random.random())
    problem_instance.local_search_report = report
    if not report["start_cost"]:
        return 0, routes
    return (report["final_cost"] - report["start_cost"])/report["start_cost"], routes

def local_search_initial(problem_instance, solution, seed = None, max_rounds = 20):
    """
    Improve an initial multi-trip solution by the relocate + 2-opt* descent (local_search.py,
    two-way check). The improved route set is stored for the two-echelon check and checked again
    as a multi-trip solution (transshipments by the DP), with its two-way cost set.
    :return: (improved multi-trip solution, or the given one when nothing improved or the start
             is not feasible, local search report)
    """
    if solution is None or not solution.is_feasible:
        return solution, None
    routes, report = LS.local_search(problem_instance, _routes_of(solution), seed = seed, max_rounds = max_rounds)
    if report["improvement"] <= LS.MIN_GAIN:
        return solution, report
    improved = RH.check_feasibility(problem_instance, routes, VRP_only = False, multi_trip = True, single_route_MVRP = False)
    set_two_way_cost(problem_instance, improved)
    if not improved.is_feasible:
        return solution, report
    return improved, report

def select_seed_customers(problem_instance, nstart = 5):
    distances = [[x, min(problem_instance.distance_matrix[problem_instance.depot.index][satellite.index]
                         #transferduration
                         +problem_instance.distance_matrix[x.index][problem_instance.depot.index]
                         +problem_instance.distance_matrix[satellite.index][x.index] for satellite in problem_instance.satellites)] for x in problem_instance.customers]
    distances.sort(key=lambda x: x[1], reverse=False) #sort by waiting times

    nstart = min(len(distances), nstart)  #order for the minimum waiting
    #first half
    seed_customer_list = [x[0] for x in distances[:nstart]] #+ [x[0] for x in distances[len(distances)-half:]]

    return seed_customer_list

#multi_start_initial: repair + store every constructed start (True) or only the best one (False)
REPAIR_ALL_STARTS = False

def multi_start_initial(problem_instance, nstart = 5, report_initials = False, last_route = []):
        
    start_time = time.time()
    
    
    initial_solutions = []       
    
    seed_customers = select_seed_customers(problem_instance, nstart)
    nstart = min(len(seed_customers), nstart) 

    for start in range(nstart):

        seed_customer =  seed_customers[start]  #seed_customers[start]
        new_solution = [[seed_customer]]

        pool = [node for node in problem_instance.customers if node != seed_customer]
        
        #after each repair we check and store the solution
        constructed_solution, pool = RH.k_best_insertion_repair(problem_instance, new_solution, pool)    

        cost = sum(RH.is_route_feasible(problem_instance, route)[2] for route in constructed_solution)
        initial_solutions.append([start, 
                                   len(constructed_solution), cost, constructed_solution])
    
    runtime_all = round(time.time() - start_time, 2)

    initial_solutions.sort(key=lambda x: x[2])   #sort by dist cost 
    initial_solutions.sort(key=lambda x: x[1])  #sort by fleet size
         
         
    if report_initials:
        report_initial_solutions_stats(problem_instance, initial_solutions, runtime_all)
    
    #multi-trip (MVRP) check with repair of the infeasible routes: the best constructed route set
    #only, or every start (REPAIR_ALL_STARTS): each repaired set is stored in
    #problem_instance.heuristics_solutions, so check_found_solutions evaluates all of them at the
    #two-echelon level and the best FE objective wins, not the best street objective
    checked = []
    for start, n_routes, cost, routes in (initial_solutions if REPAIR_ALL_STARTS else initial_solutions[:1]):
        solution_mvrp = RH.check_feasibility(problem_instance, routes, VRP_only = False, multi_trip=True, single_route_MVRP = False)
        checked.append(solution_mvrp)
    problem_instance.constructed_solutions = checked
    feasible = [s for s in checked if s.is_feasible]
    if feasible:
        feasible.sort(key=lambda s: (s.nSEVs, s.objective))
        return feasible[0]
    return checked[0]

def set_two_way_cost(problem_instance, solution):
    """solution.two_way = RH.two_way_solution_cost of its SEV routes, solution.two_way_cost = its cost."""
    if solution is None:
        return None
    if solution.is_feasible:
        solution.two_way = RH.two_way_solution_cost(problem_instance, [route.sequence for route in solution.routes_sevs])
    else:
        solution.two_way = {"feasible": False, "cost": BigNumber, "n_vehicles": 0, "n_trips": 0, "served": 0,
                            "route_costs": [], "route_trips": [], "infeasible_routes": []}
    solution.two_way_cost = solution.two_way["cost"]
    return solution.two_way

def get_initial_solution(problem_instance, nof_initializations = 1, read_initials = False):    

      
    if read_initials: 
        best_initial_routes = RH.read_initial_SEV_routes_benchmark(problem_instance)
    else:
        best_initial_routes = multi_start_initial(problem_instance,nof_initializations, True)  #generate nof_initializations many initial solutions and True write out initials

    relaxed_initial = RH.check_feasibility(problem_instance, best_initial_routes)
    relaxed_cost_initial = relaxed_initial.objective
    RH.check_found_solutions(problem_instance)

    runtime = round(time.time() - problem_instance.modelStartTime,1)
    write_out_all_initials(problem_instance, runtime)

    return relaxed_initial, relaxed_cost_initial

def write_out_all_initials(problem_instance, runtime):
    #write out all initial solutions
    path = "init_solutions_" + str(problem_instance.name) +"_"+str(len(problem_instance.customers))+ ".txt"

    with open(path, 'w') as file:
        for sol_array in problem_instance.heuristics_solutions:
            for route in sol_array:
                for ind in route:
                    file.write(str(ind) + "\t")
                file.write( "-"+ "\t")
            file.write("\n")
    return
def report_initial_solutions_stats(problem_instance, initial_solutions, runtime):  
    
    average_fleet_size = sum([x[1] for x in initial_solutions])/len(initial_solutions)
    average_cost = sum([x[2] for x in initial_solutions])/len(initial_solutions)
    best_solution = initial_solutions[0]

    with open(problem_instance.initial_report, 'a') as file:
        file.write("problem_num\t"+str(problem_instance.name) +"\tnumber_of_customers\t"+str(len(problem_instance.nodes)-1) 
                +"\tBest_initial_solution_fleet_size\t" + str(best_solution[1])+ "\ttotal_distance\t"+str(best_solution[2])+ "\tsolution_time_secs\t"+str(runtime)
                +"\tnofinitial_solutions\t" + str(len(initial_solutions))
                +"\tAverage_initials_fleet_size\t" + str(average_fleet_size) + "\tAverage_initials_cost\t" + str(average_cost)+ 
                "\n")
    
    return 

