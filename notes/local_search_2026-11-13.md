# Local search on the initial solution: relocate + 2-opt*, two-way check

`local_search.py`, called from `MultiEchelonPipeline.run_initial_construction` (step 1+,
`LOCAL_SEARCH_INITIAL = True`) on the constructed multi-trip solution, before it becomes a
candidate and the MIP warm start; also used by `Multi_level_destroy.Local_search_incumbent`.

## What was wrong with the two operators

* `perform_N_relocation` was called with `N = 0` from `Local_search_incumbent`: it never ran.
  When it ran, the move was applied even when it made the solution worse (only the storage
  was conditional), and `k_best_insertion_repair` could open a new route.
* `one_point_crossover`: `point1 = 0` cut at `list1[-1]`; `while True: ... continue` looped
  forever when no customer of the second route fitted; random cut points, one try per pair.
  `two_point_crossover` swapped segments at the same positions of both routes only.
  `N = N + 1` inside `for _ in range(N)` had no effect; `random.sample(..., 2)` failed with
  fewer than two routes of length > 1.
* Costs from `is_route_feasible` exclude the vehicle cost: emptying a route was not rewarded.
  Every candidate went through the full route check, also the two unchanged base routes,
  recomputed at every iteration.

## Revised operators

* **relocate**: customer u to the position before / after one of its k nearest neighbours,
  any route (the same one included), best move per customer.
* **2-opt\***: tail exchange at a new arc u -> v between two routes (the one-point crossover
  with chosen cut points); an empty head or tail merges the routes.
* **Two-way check**: a move is accepted only if both new routes pass
  `two_way_transshipment_checking` (minimum on-time transshipments K*, then cheapest),
  `get_makespan` and the tabu list; cost = two-way cost incl. vehicle use
  (as `two_way_solution_cost`). Before the DP an O(1) forward / backward time screen:
  earliest end of service of the head in the fastest schedule (refill before the first
  customer only) + travel against the latest start of the tail without refills.

## Data structures

`RouteData` per route (prefix distance and load, forward earliest, backward latest, exact
cost; rebuilt only for the two changed routes), `where` (customer -> route, position),
granular neighbour lists (k = 20), an O(1) cost lower bound (vehicle + c^s(distance + first
detour) + trip cost * ceil(load / Q)) and a cache of exact checks keyed by the sequence.
With the screen and the bound disabled the search makes exactly the same moves
(`tests/test_local_search.py`), with 2-7x more exact checks.

## Results (initial stage, heuristic only, 2 starts, seed 32)

| instance | two-way cost before -> after LS | two-echelon objective without LS | with LS | LS time |
|---|---|---|---|---|
| c201 | 171156.5 -> 150980.3 | 191734.5 | 171496.7 | 1.1 s |
| c208 | 171245.8 -> 151012.8 | 191835.6 | 171564.3 | 2.7 s |
| r201 | 121877.8 -> 121417.8 | 132157.7 | 131720.8 | 0.6 s |
| r208 | 91172.7 -> 91025.8 | 101521.4 | 101374.5 | 1.4 s |
| rc201 | 142293.4 -> 131881.8 | 152635.3 | 152266.8 | 0.3 s |
| rc208 | 131683.8 -> 121088.1 | 152182.9 | 131487.7 (1 FEV instead of 2) | 0.6 s |

## Against the best-known solutions (data/sols)

`scripts/compare_best_known.py`. Fleet = SEVs / FEVs / transshipments. Distance = street
(distance + detours) + water, the paper's cost.

| instance | best known fleet | best known distance | construction fleet | construction distance | + local search fleet | + local search distance |
|---|---|---|---|---|---|---|
| c201 | 3 / 2 / 14 | 1389.34 | 4 / 2 / 13 | 1734.5 (+24.8%) | 4 / 2 / 11 | 1496.7 (+7.7%) |
| c208 | 3 / 2 / 14 | 1278.33 | 4 / 2 / 13 | 1835.6 (+43.6%) | 4 / 2 / 11 | 1564.3 (+22.4%) |
| r201 | 4 / 1 / 8 | 1587.84 | 5 / 1 / 7 | 2157.7 (+35.9%) | 5 / 1 / 7 | 1720.8 (+8.4%) |
| r208 | 2 / 1 / 7 | 1050.15 | 3 / 1 / 6 | 1521.4 (+44.9%) | 3 / 1 / 6 | 1374.5 (+30.9%) |
| rc201 | 4 / 1 / 9 | 1787.57 | 5 / 1 / 9 | 2635.3 (+47.4%) | 5 / 1 / 9 | 2266.8 (+26.8%) |
| rc208 | 3 / 1 / 9 | 1257.35 | 4 / 2 / 9 | 2182.9 (+73.6%) | 4 / 1 / 8 | 1487.7 (+18.3%) |

Our objective (beta = trip cost = 10000 per SEV, FEV and trip, + distance). The best-known SEV
routes are put through the same two-echelon check: our DP re-places the transshipments and our
water MIP builds the FE routes (c201, r201, r208 reproduce the paper distance within 0.7; c208,
rc201, rc208 come out longer than in the paper, rc201 with 2 FEVs).

| instance | best known, our evaluation: fleet | distance | objective | construction | + local search |
|---|---|---|---|---|---|
| c201 | 3 / 2 / 12 | 1389.2 | 171389.2 | 191734.5 (+11.87%) | 171496.7 (+0.06%) |
| c208 | 3 / 2 / 12 | 1370.4 | 171370.4 | 191835.6 (+11.94%) | 171564.3 (+0.11%) |
| r201 | 4 / 1 / 8 | 1587.1 | 131587.1 | 132157.7 (+0.43%) | 131720.8 (+0.10%) |
| r208 | 2 / 1 / 7 | 1050.0 | 101050.0 | 101521.4 (+0.47%) | 101374.5 (+0.32%) |
| rc201 | 4 / 2 / 9 | 2035.1 | 152035.1 | 152635.3 (+0.39%) | 152266.8 (+0.15%) |
| rc208 | 3 / 1 / 8 | 1326.2 | 121326.2 | 152182.9 (+25.43%) | 131487.7 (+8.38%) |

The remaining gap is one SEV on every instance. With SEV and trip both at 10000 our solutions
trade it for fewer trips (c201: 4 SEVs + 11 trips against 3 + 12), so the objective is close,
but the paper ranks the fleet first. Relocate and 2-opt* rarely empty a whole route; a route
elimination move (ejection of a route's customers into the others) is the next step.
