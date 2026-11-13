import os
import io
import numpy
import math 
import time
from twoe.solver import MILP, LinExpr, quicksum
import Repair_heuristcs as RH

bigNumber = 100000

#The street MIPs are solved by the free HiGHS solver by default (no size limit, so the free
#routing MIP runs on 100 customers); Gurobi is the second option. Choose the solver with
#twoe.solver.set_default_backend("gurobi") or the environment variable MIP_SOLVER=gurobi.
GUROBI_HEURISTICS = 0.75        #Gurobi "Heuristics" parameter (share of time in the MIP heuristics)
HIGHS_HEURISTIC_EFFORT = 0.3    #HiGHS "mip_heuristic_effort" of the free routing MIP (default of HiGHS 0.05)

def set_search_parameters(model, free_routing = False):
    """Solver-specific search emphasis: more heuristics, as the original Gurobi setting."""
    model.params["gurobi"]["Heuristics"] = GUROBI_HEURISTICS
    if free_routing:
        model.params["highs"]["mip_heuristic_effort"] = HIGHS_HEURISTIC_EFFORT
MIP_GAP = 0.0001 #relative gap of the fixed-route model (solve_MIP); 0 gives the exact transshipment plan

def get_routes(problem, arcs):

    seen = []
    out = [j for [i,j] in arcs]
    in_flow = [i for [i,j] in arcs]
    out.sort()
    in_flow.sort()
    visited = []
    routes = []
    starts = [j for [i,j] in arcs if i==0] 
    for c in starts:

        seen.append([0, c])
        new_route = []

        current = c
        visited.append(current)        
        new_route.append(problem.nodes[current])
        next_node =  next(j for [i,j] in arcs if i == current)       
        seen.append([current, next_node])

        while next_node != 0:
            current = next_node
            visited.append(current)        
            new_route.append(problem.nodes[current])
            next_node =  next(j for [i,j] in arcs if i == current)       
            seen.append([current, next_node])  

        routes.append(new_route)  

    remaining_arcs = [[i,j] for [i,j] in arcs if [i,j] not in seen]
    unique_visits = list(set(visited)) #checking double visits and missing visits
    if len(remaining_arcs)>0:
        print("Error in street problem MIP: not all nodes visited")
        exit()
    return routes
def generate_arcs(route):
    arcs = []
    arcs.append([0, route[0].index])

    for ii, node_s in enumerate(route):
        if ii < len(route) - 1 :
            arcs.append([node_s.index, route[ii + 1].index])
        else:
            arcs.append([node_s.index, 0])    
    return arcs 
#assignment is fixed over all problems
def generate_triploads(problem, arcList, transfer_points): #for each route
    #get the trip loads
    loads_trips = [] 
    assigned_customers =  []
    
    
    for i in transfer_points:
        assigned_customers.append(i)
        problem.nodes[i].tripAssigned = i
        load_current =  problem.nodes[i].demand
        current = i 
        out_going = next(k for [j,k] in arcList if j == current)
        while  out_going != 0 and out_going not in transfer_points: 
            assigned_customers.append(out_going)                   
            load_current += problem.nodes[out_going].demand
            problem.nodes[out_going].tripAssigned = i
            current = out_going
            out_going = next(k for [j,k] in arcList if j == current)
        
        #mi[i] = load_current
        loads_trips.append(load_current)
    
    
    return loads_trips
def add_assignment_constraints(problem, model, solution_in = None, customers = None):
    """Routing variables x_ij and the assignment / depot flow constraints.
    solution_in: list of routes (node lists) -> only their arcs exist and are fixed to 1
                 (the model then only places the transshipments);
    customers:   nodes to route freely (default: every customer) when solution_in is None."""
    if not problem.nodes:
        print("Error in street problem MIP: no customer to route")
        exit()
    if solution_in:
        model._customers = []
        for route in solution_in:
            for i in route:
                model._customers.append(i.index)
    elif customers is not None:
        model._customers = [i.index for i in customers]
    else:
        model._customers = [i.index for i in problem.nodes if i.index != 0]
    
    #sets
    vertices = []
    vertices.append(0)
    vertices.extend( model._customers)

    #add assignment const
    xij = {} #street route
    nofCars = model.add_var(lb=0,ub=len(model._customers), vtype="I", name = "nofCars")
    arcList = []
    if solution_in:
        arcs_given = []
        for r in solution_in:
            arcs_given.extend(generate_arcs(r))
        #check cycles
        
        if  len(arcs_given) != len(model._customers) + len(solution_in) :
            print("repetitions in arcs")  

    arcs_given_set = set((i, j) for [i, j] in arcs_given) if solution_in else None
    for i in vertices:
        for j in vertices:
            if i==j:
                continue  
            if problem.hard_tw and problem.nodes[i].start_interval + problem.nodes[i].unloading_time + problem.travel_matrix[i][j] > problem.nodes[j].end_interval:
                continue
            if solution_in and (i, j) not in arcs_given_set: 
                continue
            xij[i,j]=model.add_var(vtype="B", name="x."+str(i)+"."+str(j))                
            arcList.append([i,j])
            
            #given solution, provide bounds
            if solution_in:
                xij[i,j].lb = 1


     #---Part 1----- Street level routing Problem Constraints 
    arc_set = set((i, j) for [i, j] in arcList)
    outgoing = {i: [] for i in vertices}
    incoming = {i: [] for i in vertices}
    for [i, j] in arcList:
        outgoing[i].append(j)
        incoming[j].append(i)
    #assignment const
    for i in model._customers:
        model.add_constr(quicksum(xij[i,j] for j in outgoing[i])==1, "outflow."+str(i)) #Const2
        model.add_constr(quicksum(xij[j,i] for j in incoming[i])==1,"inflow."+str(i)) #Const2
    
    #fleet size and  #flow balance at the depot (garage for street vehicles)
    model.add_constr(quicksum(xij[i,0] for i in incoming[0]) == nofCars) #Const3
    leaving = [[i,j] for [i,j] in arcList if i==0] 
    returning = [[i,j] for [i,j] in arcList if j==0]
    model.add_constr(quicksum(xij[i,j] for [i,j] in leaving ) - quicksum(xij[i,j] for [i,j] in returning) == 0) #Const3 

    model._arcList = arcList
    model._arc_set = arc_set
    model._incoming = incoming
    model._outgoing = outgoing
    model._vertices = vertices
    model._xij = xij
    model._nofCars = nofCars


    return

#capcaity constraints change based on solution
def add_capacity_constraints_load(problem, model, solution_in = None, transfers_in = None):
    Customers = model._customers
    arcList = model._arcList
    xij = model._xij
    vertices = model._vertices

    m = {} #load on car
    v = {} #charging at node i   
    for i in Customers:
        for j in problem.satellites:
            if transfers_in and [i,j.index] not in transfers_in:
                v[i,j.index] = model.add_var(lb=0, ub=0, vtype="C", name="v"+str(i)+"."+str(j.index))
                continue
            v[i,j.index] = model.add_var(vtype="B", name="v"+str(i)+"."+str(j.index))       #transfer decision defined just before a customer 
            
    #if solution is given, then the assignments are the closest ones
    if solution_in: 
        for i in Customers:
            continue
            for j in problem.satellites:
                before_node = next(k for [k,m] in arcList if m == i)
                minDist = math.inf
                for p in problem.satellites:
                    cipj = problem.travel_matrix[before_node][p.index] + problem.travel_matrix[p.index][i] - problem.travel_matrix[before_node][i]
                    if cipj < minDist:
                        minDist = cipj
                        selectedSatellite = p.index
                for p in problem.satellites:
                    if p.index == selectedSatellite:
                        v[i,p.index].lb = 1
                    else:
                        v[i,p.index].ub = 0
        
    for i in Customers:
        m[i] = model.add_var(lb=0,ub=problem.capacity_SEV - problem.nodes[i].demand, vtype="C", name="m"+str(i)) #load level at the departure at customer i


    for [i, j] in arcList:
        if i != 0 and j != 0:
            model.add_constr(m[i] >= m[j] + problem.nodes[j].demand - problem.capacity_SEV*(1 - xij[i,j] + quicksum(v[j,k.index] for k in problem.satellites)))
    
    leaving = model._outgoing[0]
    #first transshipment
    for i in leaving:
        model.add_constr(quicksum(v[i,k.index] for k in problem.satellites) >= xij[0, i])
        
    for i in Customers:
        model.add_constr(quicksum(v[i,k.index] for k in problem.satellites) <= 1)
    model._m = m
    model._v = v  

    
    return 

def add_temporal_constraints(problem, model):
    Customers = model._customers
    arcList = model._arcList
    xij = model._xij
    v = model._v
    vertices = model._vertices

    h = {} #service start at wp   
    lead_time = {} #delay at customer j (loading and charging considered together)
    

    #detour of visiting satellite p between i and j (the lead time of j when it is a transfer point)
    def detour(i, p, j):
        return problem.travel_matrix[i][p.index] + problem.travel_matrix[p.index][j] - problem.travel_matrix[i][j]

    incoming = model._incoming
    for i in Customers:
        h[i] = model.add_var(lb=problem.nodes[i].start_interval,ub=problem.nodes[i].end_interval, vtype="C", name="h"+str(i))
        #add lead times, bounded by the largest detour over the arcs entering i
        max_lead = max([detour(k, p, i) for k in incoming[i] for p in problem.satellites] + [0])
        lead_time[i] = model.add_var(lb=0,ub=max_lead, vtype="C", name="lead_time"+str(i))

    #garage departure: free in time between the vehicle start and the horizon end
    h[0] = model.add_var(lb=problem.nodes[0].start_interval,ub=problem.nodes[0].end_interval, vtype="C", name="h"+str(0))    

    #service start times flowing #Const8
    #big-M of arc (i,j) from the variable bounds: the constraint must be slack when x_ij = 0
    #M_ij = ub(h_i) + s_i + t_ij + transfer + ub(lead_j) - lb(h_j)  (customer windows, not the vehicle shift)
    vehicle_start = problem.nodes[0].start_interval
                
    for j in Customers:
        model.add_constr(h[j] >= vehicle_start + problem.travel_matrix[0][j] ) 
        for i in incoming[j]:
            #lead time of j = detour of the satellite chosen for the refill before j on the used arc (i,j)
            #(one constraint per arc: at most one v[j,p] is 1, and the detour bound makes it slack when x_ij = 0)
            max_detour = max(detour(i, p, j) for p in problem.satellites)
            model.add_constr(lead_time[j] >= quicksum(v[j,p.index]*detour(i, p, j) for p in problem.satellites)
                            - max(0, max_detour)*(1 - xij[i,j]))
            bigNumberTime = max(0, h[i].ub + problem.nodes[i].unloading_time + problem.travel_matrix[i][j]
                                + problem.TransferDuration + lead_time[j].ub - h[j].lb)
                    
            model.add_constr(h[i] + problem.nodes[i].unloading_time  + problem.travel_matrix[i][j] #service ends
                            + problem.TransferDuration*quicksum(v[j,p.index] for p in problem.satellites) + lead_time[j]  <= h[j] + bigNumberTime*(1 - xij[i,j])) #transfer ends

    model._h = h
    model._lead_time = lead_time

    return 

def add_objective(problem, model, fleet_level = True):
    """Hierarchical (lexicographic) objective of the street model, no weights between the levels:
    1. fleet size (number of SEVs)                      -- skipped when the routes are fixed
    2. number of transshipments (refill visits)
    3. total distance: arc distances + satellite detours (lead times), times c^s
    The cost parts are kept as expressions (model._fixed_vehicle_cost, ...) for reporting
    the paper cost (beta^s * vehicles + distance + detours + trip_unit_cost * transshipments)."""
    n_transshipments = quicksum(model._v[i,k.index] for i in model._customers for k in problem.satellites)
    distance_travel_cost = quicksum(model._xij[i,j]*problem.DistMatrix[i][j]*problem.cost_unit_travel for [i,j] in model._arcList)
    lead_time_cost = quicksum(model._lead_time[i]*problem.cost_unit_travel for i in model._customers)

    levels = [n_transshipments, distance_travel_cost + lead_time_cost]
    if fleet_level:
        levels.insert(0, LinExpr() + model._nofCars)
    model.set_objectives(levels)

    model.verbose = False
    model.mip_gap = 0.0
    model._n_transshipments = n_transshipments
    model._fixed_vehicle_cost = model._nofCars*problem.cost_vehicle_use
    model._distance_travel_cost = distance_travel_cost
    model._lead_time_cost = lead_time_cost
    return

def paper_cost(problem, model):
    """beta^s * vehicles + distance + detours + trip_unit_cost * transshipments of the
    current solution (the cost reported to the heuristics, not the MIP objective)."""
    ev = model.expr_value
    return (ev(model._fixed_vehicle_cost) + ev(model._distance_travel_cost) + ev(model._lead_time_cost)
            + problem.trip_unit_cost*round(ev(model._n_transshipments)))
#the trip assignment model used for checking a single route or a complete route set: the
#sequences are fixed, the model places the transshipments (see street_routing_MIP for free routing)
def street_problem_MIP(problem, timelimit, single= None, multiple = None, transfers_in = None): 
    solution_in = None 

    routes_in = None
    if single is not None:
        if len(single) == 0:
            return True, 0, [], single 
        solution_in = []
        solution_in.append(single)
        routes_in = single
        
    else:
        if multiple is None:
            raise ValueError("street_problem_MIP needs a route (single) or a route set (multiple); "
                             "use street_routing_MIP to route the customers freely")
        multiple = [x for x in multiple if x != []]
        solution_in = multiple
        routes_in = multiple

    problem.hard_tw = True
    model = MILP("street_problem_MIP")
        
    
    add_assignment_constraints(problem, model, solution_in)
    add_capacity_constraints_load(problem, model, solution_in, transfers_in)   
    add_temporal_constraints(problem, model)    
    add_objective(problem, model, fleet_level = False) #the routes fix the fleet
    
    #do not return any solution/vehicle object, create whenever needed 
    #returns the feasibility, costs, visits, routes  
    return solve_MIP(problem, model,timelimit, solution_in, routes_in)

def street_routing_MIP(problem, timelimit, customers = None, warm_start = None, mip_gap = None, output = False):
    """
    Free routing model: the street problem MIP over all (or the given) customers with the
    routing variables free. The model builds the SEV routes, places the transshipments and
    chooses their satellites; hierarchical objective (add_objective): fleet size, then number
    of transshipments, then distance + detours. "objective" in the result is the paper cost
    beta^s * vehicles + distance + detours + trip_unit_cost * transshipments of that solution,
    "objective_levels" the values of the three levels. Optimal for the street (SE) problem
    when it ends with gap 0; with a time limit the incumbent is a feasible solution.
    warm_start: routes (node lists) given as a MIP start, e.g. the heuristic construction;
                the transshipments of the start come from the refill DP of every route.
    :return: dict feasible, objective, routes (node lists in visiting order), visits
             [[customer, satellite], ...], n_vehicles, gap, status, time, n_vars, n_constrs,
             and the objective parts fixed_vehicle_cost, distance_travel_cost, lead_time_cost,
             transshipment_cost
    """
    start_time = time.time()
    problem.hard_tw = True
    model = MILP("street_routing_MIP")
    add_assignment_constraints(problem, model, None, customers)
    add_capacity_constraints_load(problem, model, None, None)
    add_temporal_constraints(problem, model)
    add_objective(problem, model)
    result = {"feasible": False, "objective": math.inf, "routes": [], "visits": [], "n_vehicles": 0,
              "gap": math.inf, "status": None, "time": 0.0, "n_vars": model.num_vars, "n_constrs": model.num_constrs,
              "backend": model.backend}

    if warm_start:
        for route in warm_start:
            for [i, j] in generate_arcs(route):
                if (i, j) in model._arc_set:
                    model.set_start(model._xij[i, j], 1)
            plan = RH.optimal_transshipment_checking(problem, route)
            if plan["feasible"]:
                for i, p in plan["satellites"].items():
                    model.set_start(model._v[i, p], 1)


    model.mip_gap = 0.0001 if mip_gap is None else mip_gap
    model.verbose = bool(output)
    model.time_limit = timelimit
    set_search_parameters(model, free_routing=True)
    model.optimize()
    result["status"] = model.status
    result["time"] = time.time() - start_time
    if not model.has_solution:
        return result

    val = model.value
    used_arcs = [[i, j] for [i, j] in model._arcList if val(model._xij[i, j]) > 0.5]
    routes = get_routes(problem, used_arcs)
    visits = [[i, p.index] for p in problem.satellites for i in model._customers if val(model._v[i, p.index]) > 0.5]
    gap = model.gap if model.gap is not None and model.gap == model.gap else (0.0 if model.status == "OPTIMAL" else math.inf)
    ev = model.expr_value
    result.update(feasible=True, objective=paper_cost(problem, model), objective_levels=model.obj_values,
                  routes=routes, visits=visits,
                  n_vehicles=int(round(val(model._nofCars))), gap=gap,
                  fixed_vehicle_cost=ev(model._fixed_vehicle_cost), distance_travel_cost=ev(model._distance_travel_cost),
                  lead_time_cost=ev(model._lead_time_cost), transshipment_cost=problem.trip_unit_cost*len(visits))
    return result

def solve_MIP(problem, model,timelimit, solution_in, route_return = False):    
    start_mip = time.time()
    model.mip_gap = MIP_GAP
    model.verbose = False
    model.time_limit = timelimit
    set_search_parameters(model)
    model.optimize()
    solution_time_mip = time.time() - start_mip

    
    #get the optimal if exists
    if model.has_solution:
        val = model.value
        mip_gap = model.gap
        #assignments
        number_of_vehicles_used = round(val(model._nofCars),0)
        used_arcs  = [[i,j] for [i,j] in model._arcList if val(model._xij[i,j])>0.1]
        start_service = [val(model._h[i]) for i in model._customers]
        end_service = [val(model._h[i]) + val(model._lead_time[i]) for i in model._customers]
 
        demand_list = [problem.nodes[i].demand for i in model._customers]
        sum_demand = sum(demand_list)
        
        min_trips = math.ceil(sum_demand/problem.capacity_SEV)
        #satellite visits
        satellite_visits = [[i, p.index] for p in problem.satellites for i in model._customers if val(model._v[i, p.index])>0.1]
        transfer_points_im_solution = [x[0] for x in satellite_visits]
        unique_list = set(transfer_points_im_solution)
        if len(unique_list) != len(transfer_points_im_solution): 
            print("Error in transfer point check: multiple satelllite assignments for the same transfer point") #store it for each transshipment 
            SystemExit(0)
        lead_times = [[i, val(model._lead_time[i])] for i in model._customers]
        fixed_vehicle_cost = model.expr_value(model._fixed_vehicle_cost)
        distance_travel_cost = model.expr_value(model._distance_travel_cost)
        lead_time_cost = model.expr_value(model._lead_time_cost)

        #verification of the solution
        leaving_vehicles = [j for [i,j] in used_arcs if i == 0]
        returning_vehicles = [i for [i,j] in used_arcs if j == 0]
        if len(leaving_vehicles) != len(returning_vehicles):
            print("Error in street problem MIP: vehicles not balanced")
            exit()
        if len(leaving_vehicles) != number_of_vehicles_used:
            print("Error in street problem MIP: number of vehicles used and reported not balanced")
            exit()
        elif len(used_arcs) != len(model._customers) + number_of_vehicles_used: #used arcs
            print("Error in street problem MIP: number of arcs used and reported not balanced wrt number of vehicles")
            exit()

        
        #street travel cost of the route(s): distance + detours (lead times) + transshipments, no vehicle cost
        return True,  distance_travel_cost + lead_time_cost + problem.trip_unit_cost*len(satellite_visits), satellite_visits, route_return 

    else:
        #model.computeIIS()
        #model.write("model.ilp")
        if problem.consolidated: #do not add to the tabu list
            return False, bigNumber, [], []
        if len(solution_in) == 1 and problem.tabu_search_active:
            route = solution_in[0]
            if not is_in_tabu_list(problem, route):
                problem.tabu_list.append([x.index for x in route])
        return False, bigNumber, [], []
        
def is_in_tabu_list(problem, route):
    
    list_in = [x.index for x in route]

    for ii,tabu in enumerate(problem.tabu_list):
        
        it = iter(list_in)
       
        if all(x in it for x in tabu):
            #print("tabu", tabu)
            return True
        
    return False