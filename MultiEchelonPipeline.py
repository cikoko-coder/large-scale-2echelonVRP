import os
import time
import math
import random
import collections
from datetime import date
# The heuristics (Repair_heuristcs, SEVs) build their nodes and read
# closest_satellite_network from the instanceGenerator module, so the pipeline
# generates its problems with it.
import instanceGenerator
import Multi_level_destroy as MLD
import Repair_heuristcs as RH
import local_search as LS

# Solution procedure (run_instance): the Iterated Tabu Search of the manuscript (Section 5, Figure 4)
#   ITS iteration k = 1..N_ITS (Multi_level_destroy.Solve_by_decomposition, its_iter = k):
#     k = 1: Init (multi-start greedy insertion, N_INIT starts) and local search (relocate +
#            crossover, R_LS repetitions) on the best initial solution
#     every k: ALNS (MLD destroy + the single repair RH.k_best_insertion_repair, N_ALNS iterations)
#            from R*, local search on the best ALNS solution (N_LS * k: intensification grows with k),
#            tabu list update + repair of the found schedules; R* = fewest SEVs, then two-way cost
#   FE optimization (Multi_level_destroy.Optimize_FE): all min-fleet SE solutions found (more SEVs
#     discarded) -> every route re-optimized by the fixed-route street MILP (FE_ROUTE_CHECK) -> FE
#     routing MILP + satellite-removal consolidation (check_found_solutions)
#   report: run_<instance>_<date>.txt, a line in the main report, summary_<date>.csv

# --- CONFIGURATION ---
BENCHMARK_MODE = True
DEMAND_SET_INDEX = 1
N_SATELLITES = 8
IS_SYNCHRONIZED = True
TABU_SEARCH_ACTIVE = True  # tabu list T of infeasible routes (route-level exact check)
INSTANCES_TO_RUN = ["c201.txt", "c208.txt", "r201.txt", "r208.txt", "rc201.txt", "rc208.txt"]

# ITS parameters (manuscript Section 6.1)
N_INIT = 5                 # starts of the multi-start initialization
N_ITS = 5                  # ITS iterations
R_LS = 5                   # maximum repetitions of the local search
N_LS = 500                 # local search iterations (times the ITS iteration k)
N_ALNS = 500               # ALNS iterations per ITS iteration
SA_ACCEPTANCE = 0.05       # p_acc: SA temperature = p_acc / k * travel cost of the ALNS start
FE_CANDIDATES = None       # min-fleet SE solutions sent to the FE MILP (None: all of them; more SEVs are discarded)
FE_ROUTE_CHECK = "mip"     # route check before the FE MILP: "mip" = fixed-route street MILP (min SE cost,
#                            no minimum-transshipment bound), "dp" = the two-way DP of the search

RANDOM_SEED = 32           # random.seed(RANDOM_SEED + its_iter)
LOG_MIP_CALLS = True       # count every MIP solve (model name, status) during the run
MIP_SOLVER = None          # None: environment variable MIP_SOLVER, else the free HiGHS solver; "highs", "gurobi" or "scipy"
OUTPUT_DIR = "results"     # reports, solution files and plots are written here
UNBOUNDED_VEHICLE_TIME = True  # vehicles free in time; customer time windows bound the operations
ROUTE_CHECK = "dp"         # route-level exact check: "dp" (two-way refill DP) or "mip" (street MIP of the fixed route)
REFILL_RULE = "two_way"    # route check rule: "two_way" (minimum transshipments K*, then cheapest), "optimal_cost" or "optimal"
REPAIR_ALL_STARTS = True   # initialization: MVRP-repair and store every start, not only the best one

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


def solution_figures(problem, solution):
    """SEVs / FEVs / trips, street (distance + detours), water, distance, objective of a two-echelon solution."""
    if solution is None:
        return {"feasible": False}
    trips = len(solution.transshipments)
    street = solution.travelStreet - problem.trip_unit_cost*trips
    return {"feasible": bool(solution.is_feasible), "sevs": solution.nSEVs, "fevs": int(solution.nFEVs),
            "trips": trips, "street": round(street, 2), "water": round(solution.travelWater, 2),
            "distance": round(street + solution.travelWater, 2), "objective": round(solution.objective, 2)}


SUMMARY_COLUMNS = ["instance", "customers", "init_sevs", "init_two_way", "init_ls_sevs", "init_ls_two_way",
                   "its_sevs", "its_two_way", "fe_candidates", "feasible", "sevs", "fevs", "trips", "street",
                   "water", "distance", "objective", "time_init", "time_its", "time_fe", "time_total"]


def write_summary(row):
    """Append the row to summary_<date>.csv (header written with the first row)."""
    filename = f"summary_{date.today()}.csv"
    new = not os.path.exists(filename)
    with open(filename, "a") as f:
        if new:
            f.write(",".join(SUMMARY_COLUMNS) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in SUMMARY_COLUMNS) + "\n")


def format_key(key):
    return f"{key[0]} SEVs, two-way cost {round(key[1], 1)}" if key else "-"


def format_alns(report):
    used, scores = report["used"], report["scores"]
    levels = "; ".join(f"{level}: " + ", ".join(f"{o} used {used[level][o]} score {round(scores[level][o], 1)}"
                                                for o in used[level]) for level in ("roulette", "removal", "size"))
    return (f"ALNS {report['iterations']} iterations (temperature {round(report['temperature'], 2)}): "
            f"{format_key(report['start_key'])} -> best {format_key(report['best_key'])} at iteration {report['best_iteration']}; "
            f"accepted improving {report['accepted_improving']}, worse (SA) {report['accepted_worse']}, "
            f"aspiration {report['aspiration']}, rejected {report['rejected']}, infeasible {report['infeasible']}\n"
            f"      MLD {levels}")


def run_instance(instance_name: str, reports: dict):
    """Solve one instance by the ITS (procedure at the top of the file).
    :return: (problem, final two-echelon solution or None, summary row)"""
    backend = configure_mip_solver()
    mip_log = enable_mip_logging() if LOG_MIP_CALLS else None
    RH.ROUTE_CHECK = ROUTE_CHECK
    RH.REFILL_RULE = REFILL_RULE
    MLD.REPAIR_ALL_STARTS = REPAIR_ALL_STARTS
    MLD.SA_ACCEPTANCE = SA_ACCEPTANCE
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
    emit(f"ITS: N_init {N_INIT}, N_ITS {N_ITS}, R_LS {R_LS}, N_LS {N_LS} (x k), N_ALNS {N_ALNS}, p_acc {SA_ACCEPTANCE}, "
         f"tabu list {TABU_SEARCH_ACTIVE}, route check {ROUTE_CHECK}")
    emit(f"FE: MILP on {backend} (time limit {problem.timelimit_FEmodel}s), "
         f"{'all' if FE_CANDIDATES is None else f'up to {FE_CANDIDATES}'} min-fleet SE solutions, "
         f"routes re-optimized by {'the fixed-route street MILP' if FE_ROUTE_CHECK == 'mip' else 'the two-way DP'}")
    removed = instanceGenerator.check_feasibility_all_nodes(problem)
    emit(f"customers infeasible on their own: {removed if removed else 'none'}")

    # time required for building the problem is not counted
    problem.modelStartTime = time.time()
    problem.customers = [node for node in problem.nodes if node.index > 0]
    problem.best_SE = None
    problem.its_log = []
    row = {"instance": instance_name.split(".")[0], "customers": len(problem.customers)}

    # --- ITS on the SE level --------------------------------------------------------
    t_its = time.time()
    for k in range(1, N_ITS + 1):
        problem.its_iter = k
        MLD.Solve_by_decomposition(RANDOM_SEED, problem, N_INIT, R_LS, k*N_LS, N_ALNS)
        log = problem.its_log[-1]
        emit("")
        emit(f"--- ITS iteration {k}: {log['time']:.1f}s, R* {format_key(problem.best_SE_key)} "
             f"(from {problem.best_SE_phase}), tabu list {log['tabu_list']} routes")
        if "init" in log:
            emit(f"  init ({N_INIT} starts): {log['init']['time']:.1f}s, best {format_key(log['init']['key'])}")
            emit(f"  local search on the initial solution: {log['init_ls']['time']:.1f}s, "
                 f"R* {format_key(log['init_ls']['key'])}; {LS.format_report(log['init_ls_report'])}")
            row["time_init"] = round(log["init"]["time"] + log["init_ls"]["time"], 2)
            row["init_sevs"], row["init_two_way"] = log["init"]["key"][0], round(log["init"]["key"][1], 1)
            row["init_ls_sevs"], row["init_ls_two_way"] = log["init_ls"]["key"][0], round(log["init_ls"]["key"][1], 1)
        emit(f"  {format_alns(log['alns_report'])}")
        emit(f"  ALNS {log['alns']['time']:.1f}s, {log['alns']['stored']} new solutions stored")
        emit(f"  local search on the best ALNS solution + neighborhood processing: {log['ls']['time']:.1f}s; "
             f"{LS.format_report(log['ls_report'])}")
    row["time_its"] = round(time.time() - t_its - row.get("time_init", 0), 2)
    row["its_sevs"], row["its_two_way"] = problem.best_SE_key[0], round(problem.best_SE_key[1], 1)
    emit("")
    emit(describe_solution(problem, problem.best_SE, "R* after the ITS (multi-trip SE solution)"))

    # --- FE optimization (MILP) ------------------------------------------------------
    n_mip = len(mip_log) if mip_log is not None else 0
    t = time.time()
    stored = len(problem.heuristics_solutions)
    final, n_candidates = MLD.Optimize_FE(problem, FE_CANDIDATES, FE_ROUTE_CHECK)
    row["time_fe"] = round(time.time() - t, 2)
    row["fe_candidates"] = n_candidates
    emit("")
    emit(f"--- FE optimization: {n_candidates} min-fleet SE solutions of {stored} stored (more SEVs discarded) -> "
         f"route check {FE_ROUTE_CHECK} -> FE MILP + satellite-removal consolidation: {row['time_fe']:.2f}s")
    emit(f"  two-echelon feasible {len(problem.feas_solutions)}, not feasible or cut off by the incumbent "
         f"objective {len(problem.inf_solutions)}")
    if mip_log is not None:
        emit("  " + summarize_mip_calls(mip_log, n_mip).replace("\n", "\n  "))
    for k, sol in enumerate(problem.feas_solutions):
        emit(f"  solution {k}: objective {round(sol.objective, 1)}, {sol.nSEVs} SEVs, {sol.nFEVs} FEVs, "
             f"travel total {round(sol.travelTotal, 1)}")

    # --- report ------------------------------------------------------------------------
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
