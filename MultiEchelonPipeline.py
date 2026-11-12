import os
import time
import math
import collections
from datetime import date
from os import walk # assuming necessary for other files/folders
# The heuristics (Repair_heuristcs, SEVs) build their nodes and read
# closest_satellite_network from the instanceGenerator module, so the pipeline
# generates its problems with it. ProblemGenerator (refactored copy) is kept
# importable but lacks closest_satellite_network and uses other vehicle costs.
import instanceGenerator
import ProblemGenerator
import Multi_level_destroy as MLD
import Repair_heuristcs as RH

# --- CONFIGURATION ---
# Parameters are now defined once as clear constants
BENCHMARK_MODE = True
DEMAND_SET_INDEX = 1
N_SATELLITES = 8
REP_ITS = 2
START_CONS = 2
REP_LOCAL_SEARCH = 2
ITER_LOCAL_SEARCH = 2
ITER_ALNS = 2
IS_SYNCHRONIZED = True  # Directly use boolean
TABU_SEARCH_ACTIVE = False # Directly use boolean
INSTANCES_TO_RUN = ["c201.txt", "c208.txt", "r201.txt", "r208.txt", "rc201.txt", "rc208.txt"]

# Which part of the method to run:
#   "initial" -> problem generation + initial solution construction only
#   "full"    -> the complete iterative method (run_instance)
STAGE = "initial"
RANDOM_SEED = 32           # same seed rule as Solve_by_decomposition: seed + its_iter
LOG_MIP_CALLS = True       # count every Gurobi solve (model name, status) during the run
OUTPUT_DIR = "results"     # reports, solution files and plots are written here
UNBOUNDED_VEHICLE_TIME = True  # vehicles free in time; customer time windows bound the operations

# --- HELPER FUNCTIONS ---

def create_report_file(prefix: str) -> str:
    """Creates/clears a report file with a standardized name."""
    tabu_int = int(TABU_SEARCH_ACTIVE) # 1 or 0 for filename
    filename = (
        f"{prefix}11_report_{IS_SYNCHRONIZED}_{DEMAND_SET_INDEX}_"
        f"{N_SATELLITES}_toinstance{tabu_int}_{date.today()}.txt"
    )
    # The 'w' mode opens the file, clears it, and closes it (DRY principle)
    with open(filename, "w"):
        pass
    return filename


# Gurobi status codes that appear in the log (gurobipy.GRB.Status)
GUROBI_STATUS = {1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
                 6: "CUTOFF", 9: "TIME_LIMIT", 11: "INTERRUPTED", 13: "SUBOPTIMAL"}


def enable_mip_logging():
    """Wrap gurobipy.Model.optimize to record every solve: (model name, status, vars, constrs, time).

    Returns the list that collects the records. Calling it twice does not wrap twice.
    """
    import gurobipy
    if getattr(gurobipy.Model.optimize, "_pipeline_log", None) is not None:
        return gurobipy.Model.optimize._pipeline_log
    records = []
    original = gurobipy.Model.optimize

    def optimize(self, *args, **kwargs):
        t0 = time.time()
        result = original(self, *args, **kwargs)
        records.append((self.ModelName, self.Status, self.NumVars, self.NumConstrs,
                        self.SolCount, time.time() - t0))
        return result

    optimize._pipeline_log = records
    gurobipy.Model.optimize = optimize
    return records


def summarize_mip_calls(records, since=0):
    """Text block: per model and status, number of solves and total solve time."""
    groups = collections.OrderedDict()
    for name, status, nvars, ncons, solcount, secs in records[since:]:
        g = groups.setdefault((name, status), [0, 0.0, 0, 0])
        g[0] += 1
        g[1] += secs
        g[2] = max(g[2], nvars)
        g[3] = max(g[3], ncons)
    lines = [f"MIP solves: {len(records) - since}"]
    for (name, status), (n, secs, nvars, ncons) in groups.items():
        lines.append(f"  {name:<22} {GUROBI_STATUS.get(status, status):<11} solves={n:<4} "
                     f"time={secs:7.2f}s  max vars={nvars} max constrs={ncons}")
    return "\n".join(lines)


def describe_solution(problem, solution, title):
    """Text block describing a (multi-trip or two-echelon) solution object."""
    if solution is None:
        return f"{title}: no solution"
    lines = [f"{title}:",
             f"  feasible        {solution.is_feasible}",
             f"  objective       {round(solution.objective, 2)}",
             f"  nSEVs           {solution.nSEVs}",
             f"  travel street   {round(solution.travelStreet, 2)}"]
    if getattr(solution, "nFEVs", 0):
        lines += [f"  nFEVs           {solution.nFEVs}",
                  f"  travel water    {round(solution.travelWater, 2)}",
                  f"  travel total    {round(getattr(solution, 'travelTotal', math.nan), 2)}"]
    trips = getattr(solution, "trips", [])
    if trips:
        lines.append(f"  satellite trips {len(trips)}  (transfer point, satellite, load)")
    served = 0
    for k, route in enumerate(solution.routes_sevs):
        seq = [node.index for node in route.sequence]
        served += len(seq)
        demand = sum(node.demand for node in route.sequence)
        lines.append(f"  SEV {k}: {len(seq)} customers, demand {demand}, cost {round(route.cost, 2)}: {seq}")
    lines.append(f"  customers served {served} / {len(problem.customers)}")
    return "\n".join(lines)


def replay_route_timeline(problem, route, transfer_points, satellite_of=None):
    """Replay the time check of RH.is_route_feasible for a route and given transfer points.

    Same rules as is_route_feasible: the vehicle leaves the garage at the vehicle start;
    at a transfer point the vehicle goes through a satellite (detour = closest satellite,
    as closest_satellite_network.check_arc_cost, unless satellite_of gives the satellite)
    and the transfer duration is added; it waits for the window opening and fails when it
    arrives after the window end. Returns (rows, first_violation or None, end time).
    """
    network = problem.closest_satellite_network
    t = problem.travel_matrix
    current_time = problem.depot.start_interval
    prev = 0
    rows, violation = [], None
    for pos, node in enumerate(route):
        drive = network.distance_matrix[prev][node.index]
        satellite, detour = None, 0.0
        if node.index in transfer_points:
            if satellite_of and node.index in satellite_of:
                satellite = satellite_of[node.index]
                detour = t[prev][satellite] + t[satellite][node.index] - t[prev][node.index]
            else:
                detour, satellite = network.get_closest_satellite_metrics(prev, node.index)
        arrival = current_time + drive + detour + (problem.TransferDuration if satellite is not None else 0)
        start = max(arrival, node.start_interval)
        late = arrival - node.end_interval
        rows.append((pos, node.index, satellite, round(float(detour), 1), round(float(arrival), 1),
                     node.start_interval, node.end_interval, round(float(start), 1), round(float(late), 1) if late > 0 else 0))
        if late > 0 and violation is None:
            violation = (pos, node.index, round(float(arrival), 1), node.end_interval, round(float(late), 1))
            break
        current_time = start + node.unloading_time
        prev = node.index
    return rows, violation, current_time


def trace_route_check(problem, route):
    """Explain the result of RH.is_route_feasible for one route and compare it with the street MIP."""
    import MathModels
    info = {"n": len(route), "demand": sum(node.demand for node in route)}
    info["makespan_vehicles"] = RH.get_makespan(problem, route)
    info["tabu"] = RH.is_in_tabu_list(problem, route)
    info["heuristic_transfers"] = RH.refill_points(problem, route)
    info["relaxed_feasible"] = RH.is_route_feasible(problem, route)[0]

    rows, violation, end = replay_route_timeline(problem, route, info["heuristic_transfers"])
    info["rows"], info["violation"], info["end"] = rows, violation, end
    if info["makespan_vehicles"] > 1:
        info["reason"] = f"makespan needs {info['makespan_vehicles']} vehicles"
    elif info["tabu"]:
        info["reason"] = "route contains a tabu sequence"
    elif violation:
        pos, idx, arr, l, late = violation
        info["reason"] = f"time window: customer {idx} (position {pos}) reached at {arr} > end {l} (late {late})"
    elif end > problem.depot.end_interval:
        info["reason"] = f"return {round(end, 1)} after the garage end {problem.depot.end_interval} and mip_checking failed"
    else:
        info["reason"] = "-"

    feasible, cost, visits, _ = MathModels.street_problem_MIP(problem, problem.timelimit_SEmodel, route)
    info["mip_feasible"], info["mip_cost"] = feasible, cost
    info["mip_transfers"] = [i for i, p in visits] if feasible else []
    if feasible:
        satellite_of = {i: p for i, p in visits}
        rows2, violation2, _ = replay_route_timeline(problem, route, info["mip_transfers"], satellite_of)
        info["mip_rows"], info["mip_replay_violation"] = rows2, violation2
    return info


def diagnose_two_echelon_filter(problem, emit, details):
    """Replays the first gate of RH.check_found_solutions for every stored solution.

    check_found_solutions keeps a stored route set only if check_VRP_solution says it is
    feasible, i.e. RH.is_route_feasible holds for every route; only those sets reach the
    multi-trip / MVRP checks and the water (FE) MIP. Nothing in problem is changed.
    """
    emit(f"stored complete solutions: {len(problem.heuristics_solutions)} (cutoff {problem.cutoff}, "
         f"min vessels bound {problem.minVessel_bound})")
    summary = []
    for k, sequences in enumerate(problem.heuristics_solutions):
        routes = [[problem.nodes[x] for x in sequence] for sequence in sequences]
        checked = sorted(sequences, key=lambda x: x[0]) in problem.checked_solutions
        traces = [trace_route_check(problem, route) for route in routes]
        passes = all(tr["relaxed_feasible"] for tr in traces)
        emit(f"  solution {k}: {len(routes)} routes, sizes {[len(r) for r in routes]}, already checked {checked} -> "
             + ("passes the relaxed check, goes on to the multi-trip/MVRP/FE checks" if passes
                else "REJECTED by the relaxed check (never reaches the FE solver)"))
        for r, tr in enumerate(traces):
            if tr["relaxed_feasible"]:
                continue
            mip = (f"street MIP feasible (cost {round(tr['mip_cost'], 1)})" if tr["mip_feasible"]
                   else "street MIP infeasible")
            replay = ""
            if tr["mip_feasible"]:
                v = tr["mip_replay_violation"]
                replay = (", timeline with the MIP transfer points: on time" if v is None
                          else f", timeline with the MIP transfer points: late at customer {v[1]} by {v[4]}")
            emit(f"    route {r} ({tr['n']} customers, demand {tr['demand']}): {tr['reason']}; {mip}{replay}")
            emit(f"      transfer points relaxed check {tr['heuristic_transfers']}  |  street MIP {tr['mip_transfers']}")
            summary.append((k, r, tr))
            details.append(f"solution {k} route {r}: {tr['reason']}")
            details.append("  relaxed-check timeline (pos, customer, satellite, detour, arrival, e, l, start, late)")
            details += [f"    {row}" for row in tr["rows"]]
            if tr["mip_feasible"]:
                details.append("  timeline with the street MIP transfer points and satellites")
                details += [f"    {row}" for row in tr["mip_rows"]]
    return summary


def build_problem(instance_name: str, reports: dict):
    """Generate the instance and set the run parameters (shared by every stage)."""
    if BENCHMARK_MODE:
        problem = instanceGenerator.readGenerate_benchmark(instance_name, 100,
                                                           unbounded_vehicle_time=UNBOUNDED_VEHICLE_TIME)
    else:
        problem = instanceGenerator.read_generate_case(
            DEMAND_SET_INDEX, nof_horeca=1600, nof_satellites=N_SATELLITES,
            unbounded_vehicle_time=UNBOUNDED_VEHICLE_TIME
        )

    problem.case_study = not BENCHMARK_MODE
    problem.initial_report = reports['initial']
    problem.multi_report = reports['multi']
    problem.report = reports['main']
    problem.synch_check = IS_SYNCHRONIZED
    problem.points_found = []
    problem.min_relaxed_transfer_num = 0
    problem.tabu_search_active = TABU_SEARCH_ACTIVE
    return problem


def run_initial_construction(instance_name: str, reports: dict):
    """Problem generation + initial solution construction (first block of Solve_by_decomposition).

    1. multi_start_initial: START_CONS seed customers, k-best insertion builds the SEV
       routes, then check_feasibility(multi_trip=True) solves the street MIP of every
       route (MathModels.street_problem_MIP) -> multi-trip (MVRP) solution.
    2. RH.check_found_solutions: the complete route sets stored during the construction
       are checked again; feasible ones get the water (FE) MIP (FEVmodels) -> the
       two-echelon solution, stored in problem.best_found.
    Local search and ALNS are not run.
    """
    import random

    mip_log = enable_mip_logging() if LOG_MIP_CALLS else None
    out = []

    def emit(text=""):
        print(text, flush=True)
        out.append(text)

    t_build = time.time()
    problem = build_problem(instance_name, reports)
    emit(f"=== {instance_name}: initial solution construction ===")
    emit(f"customers {len(problem.customers)}, satellites {len(problem.satellites)}, "
         f"SEV capacity {problem.capacity_SEV}, FEV capacity {problem.capacity_FEV}, "
         f"synchronized {problem.synch_check}")
    emit(f"vehicle use cost SE {problem.cost_vehicle_use}, FE {problem.cost_vehicle_use_FEV}, "
         f"street MIP time limit {problem.timelimit_SEmodel}s, min vessels bound {problem.minVessel_bound}")
    customer_windows = [(node.start_interval, node.end_interval) for node in problem.customers]
    emit(f"customer windows: earliest open {min(e for e, l in customer_windows)}, latest close {max(l for e, l in customer_windows)}")
    emit(f"vehicle time: unbounded {problem.unbounded_vehicle_time}, garage [{problem.depot.start_interval}, {problem.depot.end_interval}], "
         f"satellites [{problem.satellites[0].start_interval}, {problem.satellites[0].end_interval}], "
         f"CDC [{problem.CDC.start_interval}, {problem.CDC.end_interval}]")
    emit(f"instance generation {time.time() - t_build:.2f}s")

    # customers that cannot be served even alone (same check as Solve.py)
    removed = instanceGenerator.check_feasibility_all_nodes(problem)
    emit(f"customers infeasible on their own: {removed if removed else 'none'}")

    # time required for building the problem is not counted (as in Solve.py)
    problem.modelStartTime = time.time()
    problem.its_iter = 1
    random.seed(RANDOM_SEED + problem.its_iter)
    problem.customers = [node for node in problem.nodes if node.index > 0]

    # --- step 1: multi-start construction + street MIPs --------------------
    n_mip = len(mip_log) if mip_log is not None else 0
    t1 = time.time()
    construction = MLD.multi_start_initial(problem, START_CONS, report_initials=True)
    t_construction = time.time() - t1
    emit("")
    emit(f"--- step 1: multi-start construction ({START_CONS} starts) + street MIP per route: {t_construction:.2f}s")
    emit(describe_solution(problem, construction, "constructed multi-trip solution (street level)"))
    if mip_log is not None:
        emit(summarize_mip_calls(mip_log, n_mip))
    emit(f"complete route sets stored for the two-echelon check: {len(problem.heuristics_solutions)}")

    # --- step 2a: the gate in front of the FE solver, traced -----------------
    emit("")
    emit("--- step 2a: relaxed feasibility check of the stored solutions (gate of check_found_solutions)")
    details = []
    diagnose_two_echelon_filter(problem, emit, details)
    with open(f"filter_check_{instance_name.split('.')[0]}_{date.today()}.txt", "w") as f:
        f.write("\n".join(details) + "\n")

    # --- step 2: two-echelon check of the stored solutions (water MIPs) ----
    n_mip = len(mip_log) if mip_log is not None else 0
    t2 = time.time()
    RH.check_found_solutions(problem)
    t_two_echelon = time.time() - t2
    emit("")
    emit(f"--- step 2b: check_found_solutions (street + water MIPs): {t_two_echelon:.2f}s")
    emit(f"two-echelon feasible {len(problem.feas_solutions)}, infeasible {len(problem.inf_solutions)}, "
         f"rejected before the water level {len(problem.checked_solutions) - len(problem.feas_solutions) - len(problem.inf_solutions)}")
    if mip_log is not None:
        emit(summarize_mip_calls(mip_log, n_mip))
    emit(describe_solution(problem, problem.best_found, "initial two-echelon solution (problem.best_found)"))
    if problem.best_found is None:
        emit("NOTE: no two-echelon solution -> Solve_by_decomposition would stop in Local_search_incumbent.")

    runtime = time.time() - problem.modelStartTime
    emit("")
    emit(f"total construction time {runtime:.2f}s")

    with open(f"initial_construction_{instance_name.split('.')[0]}_{date.today()}.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    if problem.best_found is not None:
        RH.write_out_solution(problem, problem.best_found)
    return problem, construction


def run_instance(instance_name: str, reports: dict):
    """Initializes and solves a single problem instance iteratively."""
    # 1. Problem Initialization (Assuming external module 'instanceGenerator')
    if BENCHMARK_MODE:
        problem = instanceGenerator.readGenerate_benchmark(instance_name, 100)
    else:
        problem = instanceGenerator.read_generate_case(
            DEMAND_SET_INDEX, nof_horeca=1600, nof_satellites=N_SATELLITES
        )

    # 2. Problem Configuration (Simplified assignment)
    problem.case_study = False
    problem.modelStartTime = time.time()
    problem.initial_report = reports['initial']
    problem.multi_report = reports['multi']
    problem.report = reports['main']
    problem.synch_check = IS_SYNCHRONIZED
    problem.points_found = []
    problem.min_relaxed_transfer_num = 0
    problem.tabu_search_active = TABU_SEARCH_ACTIVE

    # 3. Single solution placeholder
    final_solution = None
    before_objective = math.inf

    # 4. Iterative Solver Loop (Using range() is more Pythonic and efficient)
    for iters in range(1, REP_ITS + 1):
        problem.its_iter = iters

        # MLD.Solve_by_iterative call (Parameters kept for fidelity)
        final_solution = MLD.Solve_by_iterative(
            31 + iters, problem, final_solution, START_CONS, REP_LOCAL_SEARCH,
            iters * ITER_LOCAL_SEARCH, ITER_ALNS, report_initials=False
        )

        # Stagnation check
        change = abs(final_solution.objective - before_objective)
        if change < 0.01:
            problem.min_relaxed_transfer_num += 1
        else:
            before_objective = final_solution.objective

    # 5. Final Output (Assuming external module 'RH')
    runtime = time.time() - problem.modelStartTime
    RH.write_out_solution(problem, final_solution, runtime, len(problem.satellites))
    RH.plot_found_solutions(problem)


# --- MAIN EXECUTION ---

if __name__ == "__main__":

    # 0. All outputs go to OUTPUT_DIR (data paths are resolved from the module files)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)

    # 1. Setup Report Files
    reports = {
        'main': create_report_file("report"),
        'initial': create_report_file("initial"),
        'multi': create_report_file("multi")
    }

    # 2. Path check (Original code's logic, kept simple)
    # path = os.path.dirname(os.path.abspath(__file__))+"/data/dataset"
    # filenames = next(walk(path), (None, None, []))[2]

    # 3. Run Instances
    for instance_name in INSTANCES_TO_RUN:
        if STAGE == "initial":
            run_initial_construction(instance_name, reports)
        else:
            run_instance(instance_name, reports)
