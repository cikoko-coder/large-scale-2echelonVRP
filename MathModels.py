import os
import io
import numpy
import math 
from gurobipy import *
from gurobipy import GRB
import time
from gurobipy import quicksum
from gurobipy import Model
from gurobipy import LinExpr
import Repair_heuristcs as RH

bigNumber = 100000

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
def add_assignment_constraints(problem, model, solution_in = None):
    if not problem.nodes:
        print("Error in street problem MIP: no customer to route")
        exit()
    if solution_in:
        model._customers = []
        for route in solution_in:
            for i in route:
                model._customers.append(i.index)
    else:
        model._customers = [i.index for i in problem.nodes if i.index != 0]
    
    #sets
    vertices = []
    vertices.append(0)
    vertices.extend( model._customers)

    #add assignment const
    xij = {} #street route
    nofCars = model.addVar(lb=0,ub=len(model._customers), vtype=GRB.INTEGER, name = "nofCars")
    arcList = []
    if solution_in:
        arcs_given = []
        for r in solution_in:
            arcs_given.extend(generate_arcs(r))
        #check cycles
        
        if  len(arcs_given) != len(model._customers) + len(solution_in) :
            print("repetitions in arcs")  

    for i in vertices:
        for j in vertices:
            if i==j:
                continue  
            if problem.hard_tw and problem.nodes[i].start_interval + problem.nodes[i].unloading_time + problem.travel_matrix[i][j] > problem.nodes[j].end_interval:
                continue
            if solution_in and [i,j] not in arcs_given: 
                continue
            xij[i,j]=model.addVar (vtype=GRB.BINARY, name="x."+str(i)+"."+str(j))                
            arcList.append([i,j])
            
            #given solution, provide bounds
            if solution_in:
                xij[i,j].lb = 1
    model.update()         

     #---Part 1----- Street level routing Problem Constraints 
    #assignment const
    for i in model._customers:
        model.addConstr(quicksum(xij[i,j] for j in vertices if [i,j] in arcList)==1, "outflow."+str(i)) #Const2
        model.addConstr(quicksum(xij[j,i] for j in vertices if [j,i] in arcList)==1,"inflow."+str(i)) #Const2
    
    #fleet size and  #flow balance at the depot (garage for street vehicles)
    model.addConstr(quicksum(xij[i,0] for i in model._customers if [i,0] in arcList) == nofCars) #Const3
    leaving = [[i,j] for [i,j] in arcList if i==0] 
    returning = [[i,j] for [i,j] in arcList if j==0]
    model.addConstr(quicksum(xij[i,j] for [i,j] in leaving ) - quicksum(xij[i,j] for [i,j] in returning) == 0) #Const3 

    model._arcList = arcList
    model._vertices = vertices
    model._xij = xij
    model._nofCars = nofCars

    model.update()
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
                v[i,j.index] = model.addVar (lb=0, ub=0, vtype=GRB.CONTINUOUS, name="v"+str(i)+"."+str(j.index))
                continue
            v[i,j.index] = model.addVar (vtype=GRB.BINARY, name="v"+str(i)+"."+str(j.index))       #transfer decision defined just before a customer 
            
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
        m[i] = model.addVar(lb=0,ub=problem.capacity_SEV - problem.nodes[i].demand, vtype=GRB.CONTINUOUS, name="m"+str(i)) #load level at the departure at customer i
    model.update()

    for i in Customers:
        for j in Customers:
            if [i,j] in arcList:
                model.addConstr(m[i] >= m[j] + problem.nodes[j].demand - problem.capacity_SEV*(1 - xij[i,j] + quicksum(v[j,k.index] for k in problem.satellites)))
    
    leaving = [j for [i,j] in arcList if i==0]  
    #first transshipment
    for i in leaving:
        model.addConstr(quicksum(v[i,k.index] for k in problem.satellites) >= xij[0, i])
        
    for i in Customers:
        model.addConstr(quicksum(v[i,k.index] for k in problem.satellites) <= 1)
    model._m = m
    model._v = v  
    model.update()
    
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

    for i in Customers:
        h[i] = model.addVar(lb=problem.nodes[i].start_interval,ub=problem.nodes[i].end_interval, vtype=GRB.CONTINUOUS, name="h"+str(i))
        #add lead times, bounded by the largest detour over the arcs entering i
        max_lead = max([detour(k, p, i) for [k,l] in arcList if l == i for p in problem.satellites] + [0])
        lead_time[i] = model.addVar(lb=0,ub=max_lead, vtype=GRB.CONTINUOUS, name="lead_time"+str(i))

    #garage departure: free in time between the vehicle start and the horizon end
    h[0] = model.addVar(lb=problem.nodes[0].start_interval,ub=problem.nodes[0].end_interval, vtype=GRB.CONTINUOUS, name="h"+str(0))    
    model.update()
    #service start times flowing #Const8
    #big-M of arc (i,j) from the variable bounds: the constraint must be slack when x_ij = 0
    #M_ij = ub(h_i) + s_i + t_ij + transfer + ub(lead_j) - lb(h_j)  (customer windows, not the vehicle shift)
    vehicle_start = problem.nodes[0].start_interval
                
    for j in Customers:
        model.addConstr(h[j] >= vehicle_start + problem.travel_matrix[0][j] ) 
        for [i,k] in arcList:    
            if k != j :
                continue 
            for p in problem.satellites:
                model.addConstr(lead_time[j] >= (xij[i,j] + v[j,p.index] - 1)*detour(i, p, j)) 
            bigNumberTime = max(0, h[i].ub + problem.nodes[i].unloading_time + problem.travel_matrix[i][j]
                                + problem.TransferDuration + lead_time[j].ub - h[j].lb)
                    
            model.addConstr(h[i] + problem.nodes[i].unloading_time  + problem.travel_matrix[i][j] #service ends
                            + problem.TransferDuration*quicksum(v[j,p.index] for p in problem.satellites) + lead_time[j]  <= h[j] + bigNumberTime*(1 - xij[i,j])) #transfer ends

    model._h = h
    model._lead_time = lead_time
    model.update()
    return 

def add_objective(problem, model):
    # Define objective components
    
    fixed_vehicle_cost=model.addVar(lb=0,ub=math.inf, vtype=GRB.CONTINUOUS,name="fixed_vehicle_cost") #number of cars 
    distance_travel_cost=model.addVar(lb=0,ub=math.inf, vtype=GRB.CONTINUOUS,name="distance_travel_cost") 
    lead_time_cost=model.addVar(lb=0,ub=math.inf, vtype=GRB.CONTINUOUS,name="lead_time_cost")
    model.update()

    model.addConstr(fixed_vehicle_cost>=model._nofCars*problem.cost_vehicle_use) #fixed 
    model.addConstr(distance_travel_cost >= quicksum(model._xij[i,j]*problem.DistMatrix[i][j]*problem.cost_unit_travel for [i,j] in model._arcList)) #fixed
    model.addConstr(lead_time_cost>=quicksum(model._lead_time[i]*problem.cost_unit_travel for i in model._customers)) #variable wrt customers
    
    

    model.setObjective(fixed_vehicle_cost + distance_travel_cost + lead_time_cost + problem.trip_unit_cost*quicksum(model._v[i,k.index] for i in model._customers for k in problem.satellites), GRB.MINIMIZE)

    model.setParam('OutputFlag', 0)
    model.setParam("MIPGap", 0.0)
    model._fixed_vehicle_cost = fixed_vehicle_cost
    model._distance_travel_cost = distance_travel_cost
    model._lead_time_cost = lead_time_cost
    model.update()    

    return
#the trip assignment model used for getting initial solution, checking single solution, and checking multiple solutions
def street_problem_MIP(problem, timelimit, single= None, multiple = None, transfers_in = None): 
    solution_in = None 

    routes_in = None
    if single:
        solution_in = []
        solution_in.append(single)
        routes_in = single
        
    else:
        if len(single) == 0:
            return True, 0, [], single 
        multiple = [x for x in multiple if x != []]
        solution_in = multiple
        routes_in = multiple

    problem.hard_tw = True
    model = Model("street_problem_MIP")
        
    
    add_assignment_constraints(problem, model, solution_in)
    add_capacity_constraints_load(problem, model, solution_in, transfers_in)   
    add_temporal_constraints(problem, model)    
    add_objective(problem, model)
    
    #do not return any solution/vehicle object, create whenever needed 
    #returns the feasibility, costs, visits, routes  
    return solve_MIP(problem, model,timelimit, solution_in, routes_in)

def solve_MIP(problem, model,timelimit, solution_in, route_return = False):    
    start_mip = time.time()
    model.setParam("MIPGap", 0.0001)    
    model.Params.OutputFlag = 0
    model.setParam("Timelimit", timelimit)
    model.setParam('Heuristics', 0.75)
    #model.write("street_scheduling.lp")
    #model.setParam("Cutoff",problem.best_solution*1.15)
    model.optimize()
    solution_time_mip = time.time() - start_mip

    
    #get the optimal if exists
    if model.SolCount > 0:
        mip_gap = model.MIPGap
        #assignments
        number_of_vehicles_used = round(model._nofCars.x,0)
        used_arcs  = [[i,j] for [i,j] in model._arcList if model._xij[i,j].x>0.1]
        start_service = [model._h[i].x for i in model._customers]
        end_service = [model._h[i].x + model._lead_time[i].x for i in model._customers]
 
        demand_list = [problem.nodes[i].demand for i in model._customers]
        sum_demand = sum(demand_list)
        
        min_trips = math.ceil(sum_demand/problem.capacity_SEV)
        #satellite visits
        satellite_visits = [[i, p.index] for p in problem.satellites for i in model._customers if model._v[i, p.index].x>0.1]
        transfer_points_im_solution = [x[0] for x in satellite_visits]
        unique_list = set(transfer_points_im_solution)
        if len(unique_list) != len(transfer_points_im_solution): 
            print("Error in transfer point check: multiple satelllite assignments for the same transfer point") #store it for each transshipment 
            SystemExit(0)
        satellite_decisions = [model._v[i, p.index].x for p in problem.satellites for i in model._customers if model._v[i, p.index].x>0]
        lead_times = [[i, model._lead_time[i].x] for i in model._customers]
        obj = model.getObjective()        
        obj_value = obj.getValue()
        fixed_vehicle_cost = model._fixed_vehicle_cost.x 
        distance_travel_cost = model._distance_travel_cost.x
        lead_time_cost = model._lead_time_cost.x

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

        
        #transfer-satellite assignments #returning the travel costs not the objective
        return True,  lead_time_cost + problem.trip_unit_cost*len(satellite_visits), satellite_visits, route_return 

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