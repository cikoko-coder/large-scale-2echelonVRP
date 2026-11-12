# Why street solutions did not reach the FE solver, and the fix

Run: `MultiEchelonPipeline.py`, `STAGE = "initial"`, six Solomon benchmarks (100 customers,
8 satellites, SEV capacity 175, FEV capacity 1400), unbounded vehicle time.

## 1. The chain that lost the solutions

1. **Construction** (`Multi_level_destroy.multi_start_initial`): `START_CONS` seed customers,
   `k_best_insertion_repair` inserts customers with slack-time checks and the makespan bound.
   `problem.mvrp` is False here, so **no refill timing is checked while the routes are built**;
   only the capacity bound `m = ceil(demand / 175)` enters through `get_makespan`.
   Every complete route set is stored in `problem.heuristics_solutions` before any check.
2. **Multi-trip check of the best start** (`check_MVRP_solution`): one street MIP per fixed
   route. An infeasible route loses nodes (`remove_infeasible_nodes`), they are re-inserted
   with `mvrp = True`, and the repaired set is stored too. Result: a feasible MVRP solution
   (c208: 4 SEVs, 14 trips).
3. **Gate of `check_found_solutions`** = `check_VRP_solution` = `is_route_feasible` per route.
   With `REFILL_RULE = "optimal"` this was `bothway_optimal_checking`: a refill plan with
   **exactly m transshipments**. A route whose time windows force m+1 transshipments is
   rejected and the whole route set never reaches `check_multi_trips`, `check_MVRP_solution`
   or the water MIP (`FEVmodels`), although the street MIP in step 2 had already found a
   feasible plan for it.

### The c208 route (32 customers, demand 610, m = 4)

Route `[27, 30, 100, 99, 22, 24, 29, 6, 31, 32, 33, 37, ...]`. With 4 trips the second refill
must lie in the window between the backward and forward greedy positions, positions 9..10
(customers 32 and 33). The closest-satellite detour there is 34.2 / 35.7 time units, the slack
at those customers is about 25, so every 4-trip plan is late by 8.8 (customer 32) or 9.9
(customer 33). With 5 trips a refill fits before customer 29 (position 6, detour 0.0) and
before customer 37 (position 11, where the slack is large), all on time: cost 50045.2, the
same plan the street MIP finds.

Same pattern on r208 (one route needs 4 instead of 3 trips) and rc208
(three routes need 3 instead of 2 or 4 instead of 3). c201, r201 and rc201 were never affected.

## 2. The fix: the minimum is a bound of the DP, not a rule

`Repair_heuristcs.optimal_transshipment_checking(problem, route, extra_trips=None)`:

* `refill_time_bounds`: time relaxation of the plan. Refills only add time, so the fastest
  schedule refills only before the first customer; if it is late, no plan is on time.
  It also gives `can_refill[pos]`: a refill before `pos` is possible only if the earliest
  arrival with the detour is within min(window end, latest start from which the rest of the
  route is still on time). Window positions that fail this are removed.
* K = m first. `fastest_refill_plan` (earliest finish per (trips started, start of the current
  trip), O(n · K · window width)) decides feasibility; the Pareto DP over (finish, cost)
  labels runs only for feasible K and gives the cheapest plan.
* If K = m is feasible its cost is the incumbent and K = m+1, ... are explored only while
  fixed cost + K · trip_unit_cost < incumbent. With `trip_unit_cost = 10000` this stops at
  once. If K = m is infeasible, K grows until a split is on time or no window position can
  take a refill.
* The result says why: `min_trips`, `n_trips`, `attempts[K]` ("feasible", "late at position p
  (customer c) by x", "refill k must be at positions a..b and no position can take a refill on
  time (customer c late by x, ...)"), `blocked`.

`REFILL_RULE = "optimal_cost"` is now the default of `is_route_feasible`; `"optimal"` keeps the
strict minimum rule for comparison.

### Per-route street MIP replaced by the DP

`ROUTE_CHECK = "dp"`: `check_route_transshipments(problem, route)` has the
`street_problem_MIP` interface (feasible, transfer cost, `[[customer, satellite], ...]`, route)
and is used at all 16 per-route call sites (`mip_checking`, `is_route_feasible`,
`remove_infeasible_nodes*`, `check_feasibility`, `check_multi_trips`, `check_MVRP_solution`,
`check_2EVRP_solution`, `check_2EVRP_solution_consolidated`). `ROUTE_CHECK = "mip"` restores
the MIP. `check_MVRP_given` (fixed assignments) still calls the MIP.

Cross-check in step 2a of the pipeline (`CROSS_CHECK_MIP = True`): on the 67 stored routes of
the six instances the DP and the MIP agree on feasibility and on the number of transshipments;
on 4 routes the MIP plan is 0.2 to 2.3 cost units more expensive than the DP plan, within its
`MIPGap = 1e-4` (about 4 units at an objective of 40000). Re-solving those four with a zero gap
gives exactly the DP plan. DP time per instance 0.001-0.012 s against 0.08-0.25 s for the MIPs
of the same routes.

## 3. Results (initial construction, 2 starts)

| instance | before: FE solution | after: route sets passing the gate | FE-feasible | best objective | routes with K > m |
|---|---|---|---|---|---|
| c201 | yes | 2 / 2 | 1 (+1 cut off by the incumbent) | 190657.2 (4 SEV, 2 FEV) | 0 |
| c208 | **none** | 2 / 3 | 2 | 200586.6 (4 SEV, 2 FEV) | 2 |
| r201 | yes | 2 / 2 | 1 (+1 cut off) | 150399.8 (5 SEV, 1 FEV) | 1 |
| r208 | **none** | 2 / 4 | 2 | 110372.6 (3 SEV, 1 FEV) | 1 |
| rc201 | yes | 2 / 2 | 1 (+1 cut off) | 160478.4 (5 SEV, 1 FEV) | 0 |
| rc208 | **none** | 2 / 4 | 1 (+1 cut off) | 140561.7 (3 SEV, 2 FEV) | 3 |

"Infeasible" in `check_found_solutions` counts water MIPs ended by `CUTOFF` (the incumbent),
not real infeasibility. Route sets still rejected by the gate are the ones stored before the
MVRP repair: they contain a route that is infeasible for every refill plan (the street MIP
agrees).

Tests: 22 pass (`tests/test_refill_points.py`, `tests/test_route_check_benchmark.py`): both DPs
and the fast feasibility pass against brute force, pruning never removes a usable refill
position, the crafted m+1 example, the wrapper interface, and the DP against the street MIP on
random sub-sequences of c208.

## 4. Questions from the review

**Is the initial solution created by a MIP with 100 customers?** No. `street_problem_MIP` is
only ever called with one fixed route (`single=route`): every arc of the route gets lower bound
1 (`add_assignment_constraints`), so the MIP decides only where the transshipments are and at
which satellite. The free routing call (`single=None`) is commented out in
`Solve_by_decomposition` and would fail (`len(None)`). The sequences come from the insertion
heuristic; the MIP, and now the DP, are optimal for the transshipment plan of a given sequence,
not for the SE problem. Nothing in the construction optimizes the routing itself; the objective
of the street level (`travelStreet`) is the sum of the per-route transfer costs (detours +
10000 per transshipment) and does not contain the route distances at all, so two route sets
are compared by transshipments only. This is an open modeling item for the local search.

**Multiple solutions from the initial generation.** Before, only the best start (fleet size,
then relaxed cost) was MVRP-checked and repaired; the other starts were stored unchecked and
fell at the gate when a route needed repair. `MLD.REPAIR_ALL_STARTS = True` (set by the
pipeline) repairs and stores every start; `check_found_solutions` then evaluates all of them
with the water MIP and `best_found` is the best **two-echelon** objective. On c208 and r208 the
second start gives the better FE solution (200586.6 vs 200679.5; 110372.6 vs 110413.6). Next
steps for the iterative part: keep `problem.feas_solutions` as a pool of starts for the local
search / ALNS instead of only `best_found`, raise `START_CONS`, and diversify the seeds (the
two starts differ by 0.1 % in cost because `select_seed_customers` takes the customers closest
to a garage-satellite loop).

## 4b. Initial solutions by the street MIP itself (free routing)

The model builders already support free routing (no `solution_in` fixes the arcs); only the
entry point was broken. `MathModels.street_routing_MIP(problem, timelimit, customers=None,
warm_start=None)` now builds the model over all (or the given) customers: the MIP chooses the
routes, the transshipment points and their satellites in one go, objective = vehicles +
distance + detours + 10000 per transshipment. `Multi_level_destroy.mip_initial_solution`
stores its routes and turns them into a multi-trip solution (DP transshipments);
`MultiEchelonPipeline.INITIAL_METHOD = "mip"` or `"heuristic+mip"` (MIP warm-started with the
best heuristic routes, `MIP_INITIAL_TIMELIMIT`) runs it in step 1.

Revisions of the model for the free case: arc membership by set and incoming/outgoing lists
(the builders were O(n^2 * |arcs|) with list lookups), one lead-time constraint per arc
instead of one per arc and satellite (equivalent, 8 times fewer rows), `MIP_GAP` of the
fixed-route model configurable, `street_problem_MIP` raises a clear error instead of crashing
on `single=None`.

Verification in this sandbox (pip Gurobi, size-limited to 2000 variables, so at most about
25 customers): on 10, 15 and 20 customers of c208 the MIP routes pass the DP with exactly the
MIP objective, the warm start is accepted, and the pipeline runs through to the water MIP
(`MIP_INITIAL_CUSTOMERS = 15`). 10 customers solve to optimality in 0.1 s, 15 in 4 s, 20 stop
at the time limit with a 50 % gap (the bound is weak: 10000 per vehicle and per transshipment
dominates the LP relaxation). The 100-customer run needs the full license; expect incumbents
from Gurobi's heuristics rather than proven optima, so the heuristic warm start matters.

Note on the objective: the free MIP optimizes the full street cost including the route
distance, while the heuristic MVRP objective (`check_MVRP_solution`) counts only
transshipments and detours; the pipeline compares both after `check_MVRP_solution`, so the
distance still does not enter the choice (see the open item below).

## 5. Still open

* `run_instance` calls `MLD.Solve_by_iterative`, which does not exist (use `Solve_by_decomposition`).
* Street objective without route distances (see above).
* `Repair_heuristcs.py` builds file paths with Windows backslashes.
* `ProblemGenerator` uses vehicle cost 32400, `instanceGenerator` 10000.
* `twoe.config.resolve_costs` gives an infinite FE shift now that the CDC opens at -inf.
* Street MIP `MIPGap = 1e-4` returns slightly suboptimal transshipment plans; irrelevant with the DP.

## 6. Paper reference: Karademir, Beirigo & Atasoy, EJOR 322 (2025) 480-499

Parameters now follow Section 5.1 (`COST_SETTING = "paper"`, `WLS = 0.1` in the pipeline;
`instance.set_paper_costs(wls)`):

| paper | code |
|---|---|
| c^s = 1, c^w = alpha c^s (WLS 1:10, alpha = 0.1) | `cost_unit_travel`, `cost_unit_travel_FEV` (the water MIP now uses the FE costs) |
| beta = max_{i,p} c^s (t_gi + t_ip + t_pg) + c^w (t_dp + t_pd), eq. (62); beta^w = alpha beta | `twoe.config.fixed_vehicle_cost`, `cost_vehicle_use`, `cost_vehicle_use_FEV` |
| no cost per transshipment | `trip_unit_cost = 0` (the DP then also tries K > m when it saves distance) |
| U = 2 x average service time, rounded up | `set_transfer_duration` (c: 180, r/rc: 20) |
| objective (1): z^s + z^w with travel on both levels | route cost = c^s (distance + detours) + trips; two-echelon objective uses beta^w for vessels |

Still different from the paper: 8 satellites from the data files (paper: 4 at proximity 0.1
or k-means), unbounded vehicle time instead of the latest-return rule, satellites fixed to the
closest one in the water subproblem. Six benchmarks with paper costs: all get a two-echelon
solution (objectives 1897-3282, 3-5 LEFVs, 1-4 vessels), DP and MIP agree on all 101 routes.

LBBD elements already in the heuristic (kept as reference, not implemented as B&C): master =
street routes with the closest-satellite relaxation (M, eq. 42-47), subproblem = water MIP
with synchronization (S), tabu list = no-good feasibility cut (59), cutoff = incumbent bound,
multiple initial solutions tested by S (Sec. 4.3.3). Next: the large-scale iterated heuristic.
