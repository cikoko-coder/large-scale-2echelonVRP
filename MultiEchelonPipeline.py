import os
import time
import math
from datetime import date
from os import walk # assuming necessary for other files/folders
import ProblemGenerator as instanceGenerator
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
INSTANCES_TO_RUN = ["c208.txt"]

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
        run_instance(instance_name, reports)