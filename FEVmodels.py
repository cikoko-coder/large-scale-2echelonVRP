import os
import io
import numpy
import math 
import time
from twoe.solver import MILP, LinExpr, quicksum
import MathModels

bigNumber = 100000
def get_routes(problem, arcs):

    seen = []
    out = [j for [i,j] in arcs]
    in_flow = [i for [i,j] in arcs]
    out.sort()
    in_flow.sort()
    visited = []
    routes = []
    starts = [j for [i,j] in arcs if i==problem.CDC.index] 
    for c in starts:

        seen.append([problem.CDC.index, c])
        new_route = []

        current = c
        visited.append(current)        
        new_route.append(current)
        next_node =  next(j for [i,j] in arcs if i == current)       
        seen.append([current, next_node])

        while next_node != problem.CDC.index:
            current = next_node
            visited.append(current)        
            new_route.append(current)
            next_node =  next(j for [i,j] in arcs if i == current)       
            seen.append([current, next_node])  

        routes.append(new_route)  

    remaining_arcs = [[i,j] for [i,j] in arcs if [i,j] not in seen]
    unique_visits = list(set(visited)) #checking double visits and missing visits
    if len(remaining_arcs)>0:
        print("Error in street problem MIP: not all nodes visited")
        exit()
    return routes
def water_problem_MIP(problem,tasks, route_list,  SE_cost): 
    model = MILP("water_problem_MIP")

    model._route_given = route_list
    add_assignment_constraints(problem, model, tasks) #VRP
    add_capacity_constraints_load(problem, model, tasks)   #CVRP
    add_temporal_constraints(problem, model, tasks)    #CVRPTW
    if problem.synch_check:
        add_synchronzation_constraints(problem, model, tasks, route_list)
    add_objective(problem, model, tasks)
    return solve_MIP(problem, model, tasks, SE_cost, problem.timelimit_FEmodel)
     
def solve_MIP(problem, model, tasks,SE_cost, timelimit):
    start_mip = time.time()
    model.params["gurobi"]["Heuristics"] = MathModels.GUROBI_HEURISTICS
    model.verbose = False
    model.time_limit = timelimit
    cutoff = problem.cutoff - 1 - SE_cost
    model.cutoff = cutoff if math.isfinite(cutoff) else None
    model.optimize()
    solution_time_mip = time.time() - start_mip
    #model.write("FEV_MIP.lp")
    
    #get the optimal if exists
    if model.has_solution:
        val = model.value
        mip_gap = model.gap
        #assignments
        fleet = round(val(model._nofVessels),0)
        used_arcs  = [[i,j] for [i,j] in model._arcList if val(model._xij[i,j])>0.1]
        
        locations_arcs = []
        for [i,j] in used_arcs:
            if i == problem.CDC.index:
                taskj_satellite = [task for task in tasks if task.transfer_point == j][0]
                loc_j = taskj_satellite.index
                locations_arcs.append([i, loc_j])
            elif j == problem.CDC.index:
                taski_satellite = [task for task in tasks if task.transfer_point == i][0]
                loc_i = taski_satellite.index
                locations_arcs.append([loc_i, j])
            else:
                taski_satellite = [task for task in tasks if task.transfer_point == i][0]
                loc_i = taski_satellite.index
                taskj_satellite = [task for task in tasks if task.transfer_point == j][0]
                loc_j = taskj_satellite.index
                locations_arcs.append([loc_i, loc_j])
        obj_value = model.obj_value
        fixed_vehicle_cost = val(model._fixed_vehicle_cost)
        distance_travel_cost = val(model._distance_travel_cost)

        return True, obj_value, fleet, distance_travel_cost, used_arcs, locations_arcs

    else:
        #model.computeIIS()
        #model.write("model.ilp")
        #SystemExit()
        return  False, None, None, None, None, None
    
def get_temporal_network(problem, model, tasks):
    return
    arcs_feasible = []
    arcList_SEVs = []

    transshipment_points = [x.transfer_point for x in tasks]
    trans_satellites = [i.index for i in transshipment_points ]
    for r in model._route_given:
        for node in r:
            print("cc")

    return arcs_feasible
def add_synchronzation_constraints(problem, model, tasks, route_list):
    u = model._u 
    transshipment_points = [x.transfer_point for x in tasks]
    h = {}
    arcList_SEVs = []
    for r in route_list:
        arcList_SEVs.extend(MathModels.generate_arcs(r))
    seen_list = []
    for [i,j] in arcList_SEVs:
        if i in seen_list:
            continue
        seen_list.append(i)
        h[i] = model.add_var(lb=problem.nodes[i].start_interval,ub=problem.nodes[i].end_interval, vtype="C", name="h"+str(i))
          


    #adjusting time for SEVs should comply with tws
    for [i,j] in arcList_SEVs:
        if j not in transshipment_points:
            if j == 0:
                model.add_constr(h[i] + problem.nodes[i].unloading_time +problem.travel_matrix[i][j]  <= problem.nodes[0].end_interval)
            else:
                model.add_constr(h[i] + problem.nodes[i].unloading_time + problem.travel_matrix[i][j]  <= h[j])
        else:            
            satellite_chosen = [task for task in tasks if task.transfer_point == j][0]
            delay_j = problem.travel_matrix[i][satellite_chosen.index] + problem.travel_matrix[satellite_chosen.index][j]
            model.add_constr(h[i] + problem.nodes[i].unloading_time +problem.TransferDuration + delay_j  <= h[j])
            

    for j in transshipment_points:
        satellite_chosen = [task for task in tasks if task.transfer_point == j][0]  
        prev_node = next(ii for [ii,jj] in arcList_SEVs if jj == j) 
       
        model.add_constr(u[j] >= h[prev_node] + problem.nodes[prev_node].unloading_time + problem.travel_matrix[prev_node][satellite_chosen.index], name = "lower_bound"+str(satellite_chosen.index)+"."+str(j))
        model.add_constr(u[j] +problem.TransferDuration +problem.travel_matrix[satellite_chosen.index][j] <= h[j], name = "upper_bound"+str(satellite_chosen.index)+"."+str(j))
            
    model._h = h

    return 
def add_objective(problem, model,tasks):
    fixed_vehicle_cost=model.add_var(lb=0,ub=math.inf, vtype="C",name="fixed_vehicle_cost") #number of cars 
    distance_travel_cost=model.add_var(lb=0,ub=math.inf, vtype="C",name="distance_travel_cost") 


    model.add_constr(fixed_vehicle_cost>=model._nofVessels*problem.cost_vehicle_use_FEV) #fixed, beta^w 
    
    exp = LinExpr()
    
    for taski in tasks:
        i = taski.transfer_point
        loc_i = taski.index
        if [problem.CDC.index,i] in model._arcList:
            exp.add_term(model._xij[problem.CDC.index,i], problem.cost_unit_travel_FEV*problem.DistMatrix[problem.CDC.index][loc_i])
        if [i,problem.CDC.index] in model._arcList:
            exp.add_term(model._xij[i,problem.CDC.index], problem.cost_unit_travel_FEV*problem.DistMatrix[loc_i][problem.CDC.index])
        
        for taskj in tasks:
            j = taskj.transfer_point
            loc_j = taskj.index
            if [i,j] in model._arcList:
                exp.add_term(model._xij[i,j], problem.cost_unit_travel_FEV*problem.DistMatrix[loc_i][loc_j])
            if loc_i != loc_j:
                model.set_start(model._xij[i,j], 0)

    model.add_constr(distance_travel_cost >= exp) #fixed
    

    model.set_objective(fixed_vehicle_cost + distance_travel_cost)

    model.mip_gap = 0.0
    model._fixed_vehicle_cost = fixed_vehicle_cost
    model._distance_travel_cost = distance_travel_cost


    return
def add_temporal_constraints(problem, model, tasks):
    arcList = model._arcList
    xij = model._xij
    vertices = model._vertices

    u = {} #service start at wp   
    

    #bounds of the transfer times u (vessel times are free; the customer windows bound them)
    #synchronized: u_j >= h_prev + s_prev + t(prev, p)  and  u_j + transfer + t(p, j) <= h_j,
    #              so  e_prev + s_prev + t(prev, p) <= u_j <= l_j - transfer - t(p, j)
    #asynchronous: no link with the street routes, u lies in the vehicle horizon
    predecessor = {}
    for r in getattr(model, "_route_given", None) or []:
        for [a,b] in MathModels.generate_arcs(r):
            predecessor[b] = a
    horizon = problem.CDC.end_interval
    for taski in tasks:
        i = taski.transfer_point
        p = taski.index
        if problem.synch_check and i in predecessor:
            prev = predecessor[i]
            lb_u = problem.nodes[prev].start_interval + problem.nodes[prev].unloading_time + problem.travel_matrix[prev][p]
            ub_u = problem.nodes[i].end_interval - problem.TransferDuration - problem.travel_matrix[p][i]
        else:
            lb_u = problem.CDC.start_interval if problem.CDC.start_interval > -math.inf else -horizon
            ub_u = horizon
        u[i] = model.add_var(lb=lb_u,ub=ub_u, vtype="C", name="u"+str(i))
        

    #service start times flowing #Const8
    #big-M of arc (i,j) from the bounds of u: M_ij = ub(u_i) + s_i + t_ij + transfer - lb(u_j)
                
    for taski in tasks:
        i = taski.index
        transfer_i = taski.transfer_point
        #model.add_constr(u[transfer_i] >= problem.travel_matrix[problem.CDC.index][i] ) 
        for taskj in tasks:
            j = taskj.index
            transfer_j = taskj.transfer_point
            if [transfer_i,transfer_j] in arcList:                    
                bigNumberTime = max(0, u[transfer_i].ub + problem.network[i].unloading_time + problem.travel_matrix[i][j]
                                    + problem.TransferDuration - u[transfer_j].lb)
                model.add_constr(u[transfer_i] + problem.network[i].unloading_time  + problem.travel_matrix[i][j]*xij[transfer_i,transfer_j] #service ends
                                + problem.TransferDuration  <= u[transfer_j] + bigNumberTime*(1 - xij[transfer_i,transfer_j])) #transfer ends

    model._u = u


    return
def  add_capacity_constraints_load(problem, model, tasks) :
    arcList = model._arcList
    xij = model._xij
    vertices = model._vertices

    m = {} #load on car
    for task in tasks:
        i = task.transfer_point
        m[i] = model.add_var(lb=0,ub=problem.capacity_FEV - task.demand, vtype="C", name="m"+str(i)) #load level at the departure at customer i


    for taski in tasks:
        i = taski.transfer_point
        for taskj in tasks:
            j = taskj.transfer_point
            if [i,j] in arcList:
                model.add_constr(m[i] >= m[j] + taskj.demand - problem.capacity_FEV*(1 - xij[i,j] ))
    model._m = m

    return
def add_assignment_constraints(problem, model, tasks):
    if len(tasks) == 0:
        print("Error in water problem MIP: no task to route")
        exit()
    
    #sets
    vertices = []
    vertices.append(problem.CDC.index)
    for task in tasks:
        vertices.append(task.transfer_point)

    #add assignment const
    xij = {} #street route
    nofCars = model.add_var(lb=1,ub=len(tasks), vtype="I", name = "nofCars")
    arcList = []
    
    kk = False
    if kk:
        arcList = get_temporal_network(problem, model, tasks)  

    else:
        for i in vertices:
            for j in vertices:
                if i==j:
                    continue              
                        
                xij[i,j]=model.add_var(vtype="B", name="x."+str(i)+"."+str(j))                
                arcList.append([i,j])


     #---Part 1----- Street level routing Problem Constraints 
    #assignment const
    for taski in tasks:
        i = taski.transfer_point
        model.add_constr(quicksum(xij[i,j] for j in vertices if [i,j] in arcList)==1, "outflow."+str(i)) #Const2
        model.add_constr(quicksum(xij[j,i] for j in vertices if [j,i] in arcList)==1,"inflow."+str(i)) #Const2
    
    #fleet size and  #flow balance at the depot (garage for street vehicles)
    model.add_constr(quicksum(xij[taski.transfer_point, problem.CDC.index] for taski in tasks if [taski.transfer_point,problem.CDC.index] in arcList) == nofCars) #Const3
    min_nofCars = math.ceil(sum(taski.demand for taski in tasks)/ problem.capacity_FEV)
    model.add_constr(nofCars >= min_nofCars)

    leaving = [[i,j] for [i,j] in arcList if i==problem.CDC.index] 
    returning = [[i,j] for [i,j] in arcList if j==problem.CDC.index]
    model.add_constr(quicksum(xij[i,j] for [i,j] in leaving ) - quicksum(xij[i,j] for [i,j] in returning) == 0) #Const3 

    model._arcList = arcList
    model._vertices = vertices
    model._xij = xij
    model._nofVessels = nofCars


    return