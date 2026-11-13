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
