import os
import time
import math
import collections
from datetime import date
# The heuristics (Repair_heuristcs, SEVs) build their nodes and read
# closest_satellite_network from the instanceGenerator module, so the pipeline
# generates its problems with it.
import instanceGenerator
import Multi_level_destroy as MLD
import Repair_heuristcs as RH
import local_search as LS

# Solution procedure (run_instance):
#   1  initial solution: multi-start insertion (START_CONS starts), transshipments by the DP
#   1+ local search: relocate + 2-opt* descent, two-way check (local_search.py)
#   2  ITS: ITS_ITERATIONS rounds of ALNS (alns.py) + local search, each round restarts from the
#      best solution so far with a new seed; stops after ITS_STAGNATION rounds without improvement
#   3  FE solution optimization: the FE_CANDIDATES best stored route sets (two-way cost) go
#      through check_found_solutions (multi-trip / MVRP check, water MIP, consolidation)
#   4  report: run_<instance>_<date>.txt, a line in the main report, summary_<date>.csv

# --- CONFIGURATION ---
BENCHMARK_MODE = True
DEMAND_SET_INDEX = 1
N_SATELLITES = 8
START_CONS = 2
IS_SYNCHRONIZED = True
TABU_SEARCH_ACTIVE = False
INSTANCES_TO_RUN = ["c201.txt", "c208.txt", "r201.txt", "r208.txt", "rc201.txt", "rc208.txt"]

RANDOM_SEED = 32           # random.seed(RANDOM_SEED + its_iter); ITS round k uses seed RANDOM_SEED + k
LOG_MIP_CALLS = True       # count every MIP solve (model name, status) during the run
MIP_SOLVER = None          # None: environment variable MIP_SOLVER, else the free HiGHS solver; "highs", "gurobi" or "scipy"
OUTPUT_DIR = "results"     # reports, solution files and plots are written here
UNBOUNDED_VEHICLE_TIME = True  # vehicles free in time; customer time windows bound the operations
ROUTE_CHECK = "dp"         # per-route transshipment check: "dp" (refill DP) or "mip" (street MIP)
REFILL_RULE = "two_way"    # route check rule: "two_way" (minimum transshipments K*, then cheapest), "optimal_cost" or "optimal"
REPAIR_ALL_STARTS = True   # construction: MVRP-repair and store every start, not only the best one
LOCAL_SEARCH_INITIAL = True  # step 1+
ITS_ITERATIONS = 4         # step 2: rounds of ALNS + local search (0: no ITS)
ITS_STAGNATION = 2         # stop after this many rounds without improvement
ALNS_ITERATIONS = 500      # per ITS round
ALNS_TIME_LIMIT = 60       # seconds per ITS round (None: iterations only)
FE_CANDIDATES = 10         # step 3: route sets sent to the FE optimization (None: all stored)

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


# MIP status names in the log (twoe.solver: OPTIMAL, FEASIBLE = stopped with an incumbent,
# INFEASIBLE, NO_SOLUTION = stopped without one, CUTOFF = proven worse than the cutoff)
MIP_STATUS = {}


def configure_mip_solver():
    """Set the backend of every MIP of the run (MIP_SOLVER; None keeps the environment
    variable MIP_SOLVER or the free HiGHS default). Returns the backend name."""
    from twoe import solver as mip
    mip.set_default_backend(MIP_SOLVER)
    return mip.default_backend()


def enable_mip_logging():
    """Record every MIP solve: (model name, status, vars, constrs, solutions found, time).

    Returns the list that collects the records. Calling it twice does not register twice.
    """
    from twoe import solver as mip
    global _MIP_RECORDS
    if _MIP_RECORDS is not None:
        return _MIP_RECORDS
    records = []

    def record(model):
        st = model.stats
        records.append((model.name, st.status, st.n_vars, st.n_constrs, int(model.has_solution), st.runtime))

    mip.add_solve_listener(record)
    _MIP_RECORDS = records
    return records


_MIP_RECORDS = None


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
        lines.append(f"  {name:<22} {MIP_STATUS.get(status, status):<11} solves={n:<4} "
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


def select_fe_candidates(problem, n):
    """Keep the n stored route sets (problem.heuristics_solutions) with the smallest two-way cost
    (feasible ones only); None keeps all. :return: number of route sets kept"""
    if n is None:
        return len(problem.heuristics_solutions)
    ranked = []
    for sequences in problem.heuristics_solutions:
        routes = [[problem.nodes[x] for x in sequence] for sequence in sequences]
        two_way = RH.two_way_solution_cost(problem, routes)
        if two_way["feasible"]:
            ranked.append((two_way["cost"], len(sequences), sequences))
    ranked.sort(key=lambda x: (x[0], x[1]))
    problem.heuristics_solutions = [sequences for _, _, sequences in ranked[:n]]
    return len(problem.heuristics_solutions)


def solution_figures(problem, solution):
    """SEVs / FEVs / trips, street (distance + detours), water, distance, objective of a two-echelon solution."""
    if solution is None:
        return {"feasible": False}
    trips = len(solution.transshipments)
    street = solution.travelStreet - problem.trip_unit_cost*trips
    return {"feasible": bool(solution.is_feasible), "sevs": solution.nSEVs, "fevs": int(solution.nFEVs),
            "trips": trips, "street": round(street, 2), "water": round(solution.travelWater, 2),
            "distance": round(street + solution.travelWater, 2), "objective": round(solution.objective, 2)}


SUMMARY_COLUMNS = ["instance", "customers", "initial_two_way", "ls_two_way", "its_two_way", "its_rounds",
                   "feasible", "sevs", "fevs", "trips", "street", "water", "distance", "objective",
                   "time_initial", "time_ls", "time_its", "time_fe", "time_total"]


def write_summary(row):
    """Append the row to summary_<date>.csv (header written with the first row)."""
    filename = f"summary_{date.today()}.csv"
    new = not os.path.exists(filename)
    with open(filename, "a") as f:
        if new:
            f.write(",".join(SUMMARY_COLUMNS) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in SUMMARY_COLUMNS) + "\n")


def run_instance(instance_name: str, reports: dict):
    """Solve one instance: initial solution + local search, ITS (ALNS + local search),
    FE solution optimization, report (see the procedure at the top of the file).
    :return: (problem, final two-echelon solution or None, summary row)"""
    import random

    backend = configure_mip_solver()
    mip_log = enable_mip_logging() if LOG_MIP_CALLS else None
    RH.ROUTE_CHECK = ROUTE_CHECK
    RH.REFILL_RULE = REFILL_RULE
    MLD.REPAIR_ALL_STARTS = REPAIR_ALL_STARTS
    out = []

    def emit(text=""):
        print(text, flush=True)
        out.append(text)

    problem = build_problem(instance_name, reports)
    emit(f"=== {instance_name} ===")
    emit(f"customers {len(problem.customers)}, satellites {len(problem.satellites)}, "
         f"SEV capacity {problem.capacity_SEV}, FEV capacity {problem.capacity_FEV}, synchronized {problem.synch_check}")
    emit(f"costs: c^s {problem.cost_unit_travel}, c^w {round(problem.cost_unit_travel_FEV, 4)}, "
         f"beta^s {round(problem.cost_vehicle_use, 2)}, beta^w {round(problem.cost_vehicle_use_FEV, 2)}, "
         f"per transshipment {problem.trip_unit_cost}, transfer duration U {problem.TransferDuration}")
    emit(f"MIP solver {backend} (water MIP time limit {problem.timelimit_FEmodel}s, min vessels bound {problem.minVessel_bound})")
    removed = instanceGenerator.check_feasibility_all_nodes(problem)
    emit(f"customers infeasible on their own: {removed if removed else 'none'}")

    # time required for building the problem is not counted
    problem.modelStartTime = time.time()
    problem.its_iter = 1
    random.seed(RANDOM_SEED + problem.its_iter)
    problem.customers = [node for node in problem.nodes if node.index > 0]
    row = {"instance": instance_name.split(".")[0], "customers": len(problem.customers)}

    # --- step 1: initial solution ------------------------------------------------
    t = time.time()
    best = MLD.multi_start_initial(problem, START_CONS, report_initials=True)
    MLD.set_two_way_cost(problem, best)
    row["time_initial"] = round(time.time() - t, 2)
    row["initial_two_way"] = round(best.two_way_cost, 1)
    emit("")
    emit(f"--- step 1: multi-start construction ({START_CONS} starts): {row['time_initial']:.2f}s")
    for k, sol in enumerate(getattr(problem, "constructed_solutions", [])):
        MLD.set_two_way_cost(problem, sol)
        emit(f"  start {k}: {'feasible' if sol.is_feasible else 'INFEASIBLE'}, {sol.nSEVs} SEVs, "
             f"{len(getattr(sol, 'trips', []))} trips, two-way cost {round(sol.two_way_cost, 1)}")
    emit(describe_solution(problem, best, "initial multi-trip solution (street level)"))
    if not best.is_feasible:
        emit("NOTE: the initial solution is not feasible; local search and ITS are skipped")

    # --- step 1+: local search ---------------------------------------------------
    t = time.time()
    if LOCAL_SEARCH_INITIAL:
        best, report = MLD.local_search_initial(problem, best, seed=RANDOM_SEED)
        emit("")
        emit(f"--- step 1+: local search (relocate + 2-opt*, two-way check): {time.time() - t:.2f}s")
        if report is not None:
            emit("  " + LS.format_report(report))
    row["time_ls"] = round(time.time() - t, 2)
    row["ls_two_way"] = round(best.two_way_cost, 1)

    # --- step 2: ITS (ALNS + local search) ----------------------------------------
    t = time.time()
    rounds, stagnation = 0, 0
    if ITS_ITERATIONS and best.is_feasible:
        emit("")
        emit(f"--- step 2: ITS, at most {ITS_ITERATIONS} rounds of ALNS ({ALNS_ITERATIONS} iterations, "
             f"{ALNS_TIME_LIMIT}s) + local search, stop after {ITS_STAGNATION} rounds without improvement")
    for k in range(1, ITS_ITERATIONS + 1):
        if not best.is_feasible:
            break
        rounds = k
        problem.its_iter = k
        t_round = time.time()
        candidate, alns_report = MLD.alns_improve(problem, best, seed=RANDOM_SEED + k,
                                                  iterations=ALNS_ITERATIONS, time_limit=ALNS_TIME_LIMIT)
        candidate, ls_report = MLD.local_search_initial(problem, candidate, seed=RANDOM_SEED + k)
        improved = candidate.is_feasible and candidate.two_way_cost < best.two_way_cost - LS.MIN_GAIN
        emit(f"  round {k}: two-way cost {round(best.two_way_cost, 1)} -> {round(candidate.two_way_cost, 1)}, "
             f"{candidate.nSEVs} SEVs, {time.time() - t_round:.1f}s{'' if improved else ' (no improvement)'}")
        emit("    ALNS " + MLD.ALNS.format_report(alns_report).replace("\n", "\n    "))
        if ls_report is not None:
            emit("    local search " + LS.format_report(ls_report))
        if improved:
            best, stagnation = candidate, 0
        else:
            stagnation += 1
            if stagnation >= ITS_STAGNATION:
                break
    row["time_its"] = round(time.time() - t, 2)
    row["its_two_way"] = round(best.two_way_cost, 1)
    row["its_rounds"] = rounds
    emit(describe_solution(problem, best, "best multi-trip solution after the ITS (street level)"))

    # --- step 3: FE solution optimization ------------------------------------------
    n_mip = len(mip_log) if mip_log is not None else 0
    t = time.time()
    stored = len(problem.heuristics_solutions)
    kept = select_fe_candidates(problem, FE_CANDIDATES)
    RH.check_found_solutions(problem)
    row["time_fe"] = round(time.time() - t, 2)
    emit("")
    emit(f"--- step 3: FE solution optimization of the {kept} best of {stored} stored route sets "
         f"(water MIP + consolidation): {row['time_fe']:.2f}s")
    emit(f"  two-echelon feasible {len(problem.feas_solutions)}, infeasible {len(problem.inf_solutions)}")
    if mip_log is not None:
        emit("  " + summarize_mip_calls(mip_log, n_mip).replace("\n", "\n  "))
    for k, sol in enumerate(problem.feas_solutions):
        emit(f"  solution {k}: objective {round(sol.objective, 1)}, {sol.nSEVs} SEVs, {sol.nFEVs} FEVs, "
             f"travel total {round(sol.travelTotal, 1)}")

    # --- step 4: report ------------------------------------------------------------
    final = problem.best_found
    row.update(solution_figures(problem, final))
    row["time_total"] = round(time.time() - problem.modelStartTime, 2)
    emit("")
    emit(describe_solution(problem, final, "final two-echelon solution"))
    emit(f"total time {row['time_total']:.2f}s")
    with open(f"run_{row['instance']}_{date.today()}.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    with open(problem.report, "a") as f:
        f.write("\t".join(f"{c}\t{row.get(c, '')}" for c in SUMMARY_COLUMNS) + "\n")
    write_summary(row)
    if final is not None:
        RH.write_out_solution(problem, final)
    return problem, final, row


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # all outputs go to OUTPUT_DIR (data paths are resolved from the module files)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)
    reports = {
        'main': create_report_file("report"),
        'initial': create_report_file("initial"),
        'multi': create_report_file("multi")
    }
    for instance_name in INSTANCES_TO_RUN:
        run_instance(instance_name, reports)
