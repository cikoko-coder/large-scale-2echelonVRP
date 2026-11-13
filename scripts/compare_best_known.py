"""Initial solution (construction, construction + local search) against the best-known solution of
data/sols, the best-known SEV routes evaluated by the same two-echelon check (two-way DP + water MIP).
Usage: python scripts/compare_best_known.py c201.txt <output dir>   -> one JSON line"""
import sys, io, os, contextlib, time, random, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import MultiEchelonPipeline as P, instanceGenerator, Repair_heuristcs as RH
name=sys.argv[1]; os.chdir(sys.argv[2])
P.INITIAL_METHOD="heuristic"; P.CROSS_CHECK_MIP=False
reports={k:P.create_report_file(k) for k in ("main","initial","multi")}
RH.ROUTE_CHECK=P.ROUTE_CHECK; RH.REFILL_RULE=P.REFILL_RULE

def figures(p, sol):
    trips = len(sol.transshipments)
    street = sol.travelStreet - p.trip_unit_cost*trips  # distance + detours (c^s = 1)
    return dict(sevs=sol.nSEVs, fevs=int(sol.nFEVs), trips=trips, street=round(street,2),
                water=round(sol.travelWater,2), dist=round(street+sol.travelWater,2), obj=round(sol.objective,1))

out={}
for ls in (0,1):
    P.LOCAL_SEARCH_INITIAL = bool(ls)
    t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        p,_ = P.run_initial_construction(name, reports)
    out["ls" if ls else "constr"] = dict(figures(p, p.best_found), time=round(time.time()-t,1))

# best known: its SEV routes through the same two-echelon check (transshipments by our DP, water MIP)
with contextlib.redirect_stdout(io.StringIO()):
    p = P.build_problem(name, reports)
    p.modelStartTime=time.time(); p.its_iter=1
    p.customers=[n for n in p.nodes if n.index>0]
    instanceGenerator.read_best_known_solution(p)
    p.heuristics_solutions=[[[n.index for n in r] for r in p.best_known_SL]]
    RH.check_found_solutions(p)
lines=[l.split() for l in open(ROOT + "/data/sols/"+[f for f in os.listdir(ROOT + "/data/sols") if f.split("_")[0]==name.split(".")[0]][0]).read().split("\n")[1:4]]
out["bks_paper"]=dict(sevs=len(p.best_known_SL), fevs=int(lines[1][1]), trips=len(p.best_known_transfers),
                      street=round(p.SAO_cost_SE,2), water=round(p.SOA_cost_FE,2), dist=float(lines[0][1]))
out["bks_ours"]=figures(p, p.best_found) if p.best_found is not None else None
print(json.dumps({name: out}))
