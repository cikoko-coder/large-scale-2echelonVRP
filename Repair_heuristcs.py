import copy
import math
from multiprocessing import pool
from operator import index
import SEVs as sev
import MathModels
import instanceGenerator
import FEVmodels
import matplotlib.pyplot as plt
import time
import os

BigNumber = 1000000
def bothway_checking(problem_instance, route):
    """
    Finds node indices where a refill is required.
    :param demands: List of loads for each node (e.g., [2, 5, 8, 3])
    :param capacity: Maximum vehicle capacity Q
    :return: List of indices where the vehicle takes a break to refill
    """
    demands = [node.demand for node in route]
    capacity = problem_instance.capacity_SEV
    
    
    total_demand = sum(demands)
    min_refills = math.ceil(total_demand / capacity)    

    list_ind = [node.index for node in route]
    first_node = route[0].index

    if min_refills > 1:
        for check in range(2):
            if check == 0:
                refill_indices = []
                forward = refill_indices
            else:
                demands.reverse()
                list_ind.reverse()
                refill_indices = []
                backward = refill_indices
            current_load =  0    
            for i, demand in enumerate(demands):            
                # If adding this node exceeds current capacity, refill first
                if current_load + demand > capacity:
                    refill_indices.append(list_ind[i])
                    # After refilling, the load is just the current node's demand
                    current_load = demand
                else:
                    # Continue adding to the current "bin"
                    current_load += demand
            if first_node not in refill_indices:
                refill_indices.insert(0, first_node) #first node is always a refill  
    else:
        forward, backward = [route[0].index], [route[0].index] #first node is always a refill and the only one if needed

    return forward, backward

def forward_checking(problem_instance, route):
    """
    Finds node indices where a refill is required.
    :param demands: List of loads for each node (e.g., [2, 5, 8, 3])
    :param capacity: Maximum vehicle capacity Q
    :return: List of indices where the vehicle takes a break to refill
    """
    demands = [node.demand for node in route]
    capacity = problem_instance.capacity_SEV   

    
    total_demand = sum(demands)
    min_refills = math.ceil(total_demand / capacity)    

    if min_refills > 1:
        refill_indices = []
        current_load =  0    
        for i, demand in enumerate(demands):            
            # If adding this node exceeds current capacity, refill first
            if current_load + demand > capacity:
                refill_indices.append(route[i].index)
                # After refilling, the load is just the current node's demand
                current_load = demand
            else:
                # Continue adding to the current "bin"
                current_load += demand
          

        if route[0].index not in refill_indices:
            refill_indices.insert(0, route[0].index) #first node is always a refill
    else: 
        refill_indices = [route[0].index] #first node is always a refill
    return refill_indices

def backward_checking(problem_instance, route):
    demands = [node.demand for node in route]
    capacity = problem_instance.capacity_SEV
    # Reverse the list to process backward
    rev_demands = demands[::-1]
    n = len(demands)
    
    refill_indices_rev = []
    current_load = 0
    
    for i, demand in enumerate(rev_demands):
        
        if current_load + demand > capacity:
            # This node triggers a 'refill' in the backward flow
            # The actual node index in the original list is (n - 1 - i)
            refill_indices_rev.append(route[n - 1 - i].index)
            current_load = demand
        else:
            current_load += demand
            
    # Reverse the results to keep them in chronological order (0 -> N)
    refill_indices_rev.reverse()
    if route[0].index not in refill_indices_rev:
        refill_indices_rev.insert(0, route[0].index) #first node is always a refill
    return refill_indices_rev

def mip_checking(problem_instance, route):

    feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, route)
     #visit [i,p]
    if feasibility:
        transfer_points = [visit[0] for visit in visits ]   
    else:
        transfer_points = []
    
    return transfer_points, cost

def comparison(problem_instance, route):
    vrp_cost = problem_instance.cost_unit_travel*sum(problem_instance.distance_matrix[route[i].index][route[i+1].index] for i in range(len(route)-1))
    vrp_cost += problem_instance.cost_unit_travel*problem_instance.distance_matrix[route[-1].index][0] 
    vrp_cost += problem_instance.cost_unit_travel*problem_instance.distance_matrix[0][route[0].index] #return to depot and departure from depot
    #transfer costs
    #sets of solutions
    #forward = forward_checking(problem_instance, route) 
    forward, backward = bothway_checking(problem_instance, route)
    
    #distance increase to visit the closest satellite before the forward checking points
    forward_cost = check_cost_VRP_given_transfer_points(problem_instance, route, forward)
    backward_cost = check_cost_VRP_given_transfer_points(problem_instance, route, backward)

    if forward_cost < backward_cost: #solution is the same for routing but different for the transfer points
        
        #if feasible return the forward solution
        trips = forward
        #feasibility = check_tw(problem_instance, route, forward)
    
    return vrp_cost, forward_cost, backward_cost

def check_cost_VRP_given_transfer_points(problem_instance, route, transfer_points):
    #visit [i,p]
    transfer_cost = problem_instance.trip_unit_cost*len(transfer_points)
    for i in range(len(route)):
        index_out = route[i].index
        if index_out in transfer_points:
            if i == 0:
                index_in = 0
            else:
                index_in = route[i-1].index
                
            transfer_cost += problem_instance.closest_satellite_network.check_arc_cost(index_in, index_out)

    return transfer_cost 

def categorize_orders(problem_instance, solution, pool_nodes, partial = False):
    #separate node ordering function
    workloads = []
    satellite_assignements = []
    for s in problem_instance.satellites:
        workloads.append([])
        satellite_assignements.append([])

    
    for node in pool_nodes:
        closest = [problem_instance.distance_matrix[node.index][satellite.index] for satellite in problem_instance.satellites]
        index_close = closest.index(min(closest))
        workloads[index_close].append(node)    
        satellite_assignements[index_close].append(node)
    
    workloads = [x for x in workloads if len(x) > 0]
    workloads.sort(key=lambda x: len(x), reverse = True)
    
    #workloads = []
    #workloads.append(pool_nodes) 
    #max_number =min(problem_instance.removal_max, len(workloads[0])) 
    max_number =min(1000, len(workloads[0]))
    
    pool_nodes = workloads[0][:max_number]
    if max_number == len(workloads[0]):
        workloads.pop(0)
    else:
        workloads[0] = [node for node in workloads[0] if node not in pool_nodes]

    w = sum([len(list) for list in workloads])
    if not partial and sum(len(x) for x in solution) + w + len(pool_nodes) != len(problem_instance.customers) :
        print(" more nodes inserted than customers")    

    return pool_nodes, workloads



def k_best_insertion_repair(problem_instance, solution, pool_nodes, partial = False): 
    
    pool_nodes, workloads = categorize_orders(problem_instance, solution, pool_nodes, partial)    
    
    while len(pool_nodes) > 0:
        update_intervals(problem_instance, solution)
        
        before = [node.index for node in pool_nodes]
        node = pool_nodes[0] 
        last_node = node

        if node.index == "67":
            print("here")
            
        # Calculate insertion costs for the current node into the route's sequence
        insertion_costs = k_best_insertion_costs_per_route(problem_instance, solution, node )
        # Populate insertion_costs based on the calculation
        # Find the best insertion cost for the current node and add it to the result
        if not insertion_costs: # If a feasible solution was found                
            #worst case scenario is to open a new route    
            new_route = [node]
            pool_nodes.remove(node)
            solution.append(new_route)   

        else: #feasible insertion exist    
            
            # Insertion costs is sorted so start with best and see if there is a better one battery wise
            insertion_costs.sort(key=lambda x: x[2])

            best_insertion = insertion_costs[0] 
                
            # Now take the best insertion and add it to the routes
            route_new = best_insertion[1]
            insertion_index = route_new.index(node)
            route_new.remove(node)   
            solution.remove(route_new)
            
            #append update route(important for list based references with addres pointer information)
            route_new_s = route_new[:insertion_index] + [node] + route_new[insertion_index:]                        
            solution.append(route_new_s)
            pool_nodes.remove(node)
                
        
         
        after = [node.index for node in pool_nodes]
        
        if before == after:
            print("no insertion")   
        if sum( x.demand for x in pool_nodes) < problem_instance.capacity_SEV:
            if len(workloads) > 0:     
                selected = [problem_instance.distance_matrix[last_node.index][x[0].index] for x in workloads ]           
                
                index_chosen = selected.index(min(selected))
                chosen = workloads[index_chosen]                                

                max_number =min(problem_instance.removal_max, len(chosen)) 
    
                new_pool_nodes = chosen[0:max_number]
                if max_number == len(chosen):
                    workloads.remove(chosen)
                else:
                    workloads[index_chosen] = [node for node in workloads[index_chosen] if node not in new_pool_nodes]

                pool_nodes.extend(new_pool_nodes) # workloads[0]
                #workloads.pop(0)
                terminated = True

        w = sum([len(list) for list in workloads]) 
        if not partial and sum(len(x) for x in solution) + w + len(pool_nodes) != len(problem_instance.customers) :
            print(" more nodes inserted than customers")   

        if not pool_nodes:
            break

    if sum(len(x) for x in solution) == len(problem_instance.customers):
        partial = False

    if not pool_nodes and not partial: #important to track and store all possible solutions in the feasible solution list for the future reference and to avoid infeasible insertions in the future
        add_to_feasible_solution(problem_instance, solution)

    if len(pool_nodes)>0: #complete the solution all the time
        for node in pool_nodes:
            xx = [node]
            solution.append(xx)
        pool_nodes.clear()
    
    return solution, pool_nodes


def k_best_insertion_costs_per_route(problem_instance, solution, customer_node):
        
    k_best_costs = []
    
    for route in solution:
        #get sorted insertions
        #use slack times AND relaxed feasibility conditions
        insertions = insert_into_every_index_slack(problem_instance, route, customer_node) 
        
        k_best_costs.extend(insertions)  

    status2 = problem_instance.mvrp    
    if problem_instance.mvrp:
        feasible_solutions = k_best_costs
        feasible_solutions.sort(key=lambda x: x[2])
        mvrp_feas = []        
        for x in feasible_solutions:
            result = is_route_feasible(problem_instance, x[1])
            if result[0]: #return only the cheapest feasible solution with the insertion
                mvrp_feas.append(result)
                break
        k_best_costs = mvrp_feas

    problem_instance.mvrp = status2

    
        
    return k_best_costs

def insert_into_every_index_slack(problem_instance, input_list, item):
    #problem_instance.earliest_times_feasible[node.index] = service_starts[i]
    #problem_instance.latest_times_feasible[node.index] = latest_times_feasible[i]
    # Add possibility of route with one location
    feasible_solutions = [] 
    initial_cost = get_min_multi_trip_cost(problem_instance, input_list)
    
    for i in range(len(input_list) + 1):
        candidate = []
        if i == 0:
            prev_node = problem_instance.depot
            next_node = input_list[0]
        elif i == len(input_list):
            prev_node = input_list[-1]
            next_node = problem_instance.depot
        else:   
            prev_node = input_list[i-1]
            next_node = input_list[i]
        service_start = max(item.start_interval, problem_instance.earliest_times_feasible[prev_node.index] + prev_node.unloading_time + problem_instance.distance_matrix[prev_node.index][item.index])
        if service_start > item.end_interval:
            continue
        next_arrival = service_start + item.unloading_time + problem_instance.distance_matrix[item.index][next_node.index]
        #makespan_cost =  problem_instance.latest_times_feasible[next_node.index] - next_arrival 
        
        #if makespan_cost < 0:
            #print("infeasible due to time window violation detected in slack time based insertion list generation")
        #    continue
            
        if next_arrival <= problem_instance.latest_times_feasible[next_node.index]:
            candidate = input_list[:i] + [item] + input_list[i:]

            if get_makespan(problem_instance, candidate) > 1:
                continue            
            
            makespan_cost =  max(0,problem_instance.latest_times_feasible[next_node.index] - next_arrival )
            
            distance_cost = get_min_multi_trip_cost(problem_instance, candidate) - initial_cost            
            
            feasible_solutions.append([True, candidate, distance_cost]) 
   
    #insertions represent the candidate route first, then objective change, then makespan change, then the cost of the route with the insertion, then the cost of the route without the insertion, then the index of insertion
    #AT INDEX one THE ROUTE MODIFIED BY THE INSERTION
    #AT INDEX two THE COST OF THE ROUTE WITH THE INSERTION
    return feasible_solutions

def get_min_multi_trip_cost(problem_instance, route):
    
    arcs_no_trip = MathModels.generate_arcs(route)

    arcs_with_trip = [min(problem_instance.distance_matrix[i][satellite.index]
                       + problem_instance.distance_matrix[satellite.index][j]
                       - problem_instance.distance_matrix[i][j] for satellite in problem_instance.satellites) 
                       for [i,j] in arcs_no_trip if j!=0]

    min_transfer = math.ceil((sum([node.demand for node in route]) )/problem_instance.capacity_SEV) 
    arcs_with_trip.sort()

    #num_bins = min_transfer + problem_instance.min_relaxed_transfer_num
    #num_bins = min(len(route), num_bins)
    #bin_size = len(route) // num_bins
    #cutting_points = [i * bin_size for i in range(0, num_bins)]    

    min_satellite_visit_cost =  sum(arcs_with_trip[:min_transfer]) 

    #min_satellite_visit_cost =  sum(arcs_with_trip[x] for x in cutting_points) 

    
    arc_cost = sum(problem_instance.distance_matrix[i][j] for [i,j] in arcs_no_trip)
    cost_v = problem_instance.cost_unit_travel*(min_satellite_visit_cost + arc_cost)
    cost_trips =  problem_instance.trip_unit_cost*min_transfer
    
    return cost_v   + cost_trips #+ problem_instance.cost_vehicle_use

def get_makespan(problem_instance, route): #check feasibility in time, use the lowet bound 
    
    arcs_no_trip = MathModels.generate_arcs(route)

    arcs_with_trip = [min(problem_instance.distance_matrix[i][satellite.index]
                       + problem_instance.distance_matrix[satellite.index][j]
                       - problem_instance.distance_matrix[i][j] for satellite in problem_instance.satellites) 
                       for [i,j] in arcs_no_trip if j!=0]

    min_transfer = math.ceil((sum([node.demand for node in route]) )/problem_instance.capacity_SEV)
    arcs_with_trip.sort()
 
    #num_bins = min_transfer
    #bin_size = len(route) // num_bins
    #cutting_points = [i * bin_size for i in range(0, num_bins)]    

    min_satellite_visit_cost =  sum(arcs_with_trip[:min_transfer]) 

    #min_satellite_visit_cost =  sum(arcs_with_trip[x] for x in cutting_points) 

    arc_cost = sum(problem_instance.distance_matrix[i][j] for [i,j] in arcs_no_trip)
    service_sum = sum(x.unloading_time for x in route) + min_transfer*problem_instance.TransferDuration
    make = min_satellite_visit_cost + arc_cost + service_sum

    minSEvs = math.ceil(make/problem_instance.nodes[0].end_interval)
    
    return minSEvs

def get_makespan2(problem_instance, route):
    
    distances = [problem_instance.distance_matrix[route[i].index][route[i + 1].index] for i in range(len(route) - 1)]
    distances.append(problem_instance.distance_matrix[route[-1].index][0])  
    distances.append(problem_instance.distance_matrix[0][route[0].index])  
    dist_sum = sum(distances)
    service_sum = sum(x.unloading_time for x in route)
    make = dist_sum + service_sum
    
    minSEvs = math.ceil(make/problem_instance.nodes[0].end_interval)

    return minSEvs

def divide_routes(problem_instance, route, minSEVS):
    routes = []
    node_list = [copy.deepcopy(x) for x in route]
    for i in range(minSEVS):
        route = []
        for node in node_list:
            route.append(node)
            nof_sev = get_makespan(problem_instance, route)
            if nof_sev >1:
                route.remove(node)
                routes.append(copy.deepcopy(route))
                break
            else:
                node_list.remove(node)
                
    return routes

def update_intervals(problem_instance, routes):
    problem_instance.earliest_times_feasible = [problem_instance.depot.start_interval]*(len(problem_instance.nodes))
    problem_instance.latest_times_feasible = [problem_instance.depot.end_interval]*(len(problem_instance.nodes))
    problem_instance.arrivals = [problem_instance.depot.end_interval]*(len(problem_instance.nodes))
    for route in routes:
        service_starts, latest_times_feasible,arrivals = get_latest_times(problem_instance, route)
        for i, node in enumerate(route):
            problem_instance.earliest_times_feasible[node.index] = service_starts[i]
            problem_instance.latest_times_feasible[node.index] = latest_times_feasible[i]
            problem_instance.arrivals[node.index] = arrivals[i]
    return
def get_latest_times(problem_instance, route):    
    #departure and return
    indices = [node.index for node in route]
    indices.insert(0,0)
    indices.append(0)
    size = len(indices) #- 1

    arrivals = [0 for i in range(len(indices) )]

    service_starts = [problem_instance.nodes[ind].start_interval for ind in indices]       
    unloading_times = [problem_instance.nodes[ind].unloading_time for ind in indices]
    latest_times_feasible = [problem_instance.nodes[ind].end_interval for ind in indices] 
    distances = [problem_instance.distance_matrix[indices[i]][indices[i+1]] for i in range(size-1)]
    
    for i in range(1,size):
        arrivals[i] = service_starts[i - 1] + distances[i - 1] + unloading_times[i - 1]
        service_starts[i] = max(service_starts[i - 1] + distances[i - 1] + unloading_times[i - 1], service_starts[i])
    
    for i in range(size-1, 0, -1):
        latest_times_feasible[i - 1] = min(latest_times_feasible[i] - distances[i-1] - unloading_times[i - 1], latest_times_feasible[i -1])
    
    slack_times = [latest_times_feasible[i] - service_starts[i] for i in range(size)]

    if min(slack_times) < -0.001:
        return False, route, BigNumber, BigNumber, BigNumber   
        
    service_starts.pop(0)
    latest_times_feasible.pop(0)
    service_starts.pop()
    latest_times_feasible.pop()
    arrivals.pop(0)
    arrivals.pop()

    return service_starts, latest_times_feasible, arrivals

def contains_subset(main_array, sub_array):
    it = iter(main_array)
    return 
def is_feasible_list(problem_instance, route):
    #return False
    list_in = [x.index for x in route]
    cost = None
    list_nn = [x for x in problem_instance.feasible_list if x [0] == list_in]
    if list_nn:            
        return True, list_nn[0][1]
        
    return False, cost

def is_in_tabu_list(problem_instance, route):
    
    list_in = [x.index for x in route]

    for ii,tabu in enumerate(problem_instance.tabu_list):
        
        it = iter(list_in)
       
        if all(x in it for x in tabu):            
            return True
    return False

def is_route_feasible(problem_instance, route):      
           
    
    if get_makespan(problem_instance, route) > 1:
        return False, route, BigNumber
    if is_in_tabu_list(problem_instance, route):
        return False, route, BigNumber
    
    vrp_cost = problem_instance.cost_unit_travel*sum(problem_instance.distance_matrix[route[i].index][route[i+1].index] for i in range(len(route)-1))
    vrp_cost += problem_instance.cost_unit_travel*problem_instance.distance_matrix[route[-1].index][0] 
    vrp_cost += problem_instance.cost_unit_travel*problem_instance.distance_matrix[0][route[0].index] #return to depot and departure from depot

    trans_points = forward_checking(problem_instance, route) 
    
    current_time = 0    
    prev_node_index = 0
    # Calculate penalties for arriving late
    for ind, node in enumerate(route):
        #first node is always a transfer point, so add transfer time if it's the first node
        current_time += problem_instance.closest_satellite_network.distance_matrix[prev_node_index][node.index]            
        if node.index in trans_points: #transfer point
            current_time += problem_instance.closest_satellite_network.check_arc_cost(prev_node_index, node.index)
            current_time += problem_instance.TransferDuration  # Add transfer time if it's a transfer point
        
        
        if current_time < node.start_interval:
            current_time = node.start_interval  # Wait until the start_interval time
        
        elif current_time > node.end_interval: # If hard TW is violated
            return False, route, BigNumber
        
        current_time += node.unloading_time  # Add service time
        prev_node_index = node.index       
    
    
    end = current_time
    feas = True
    if end > problem_instance.nodes[0].end_interval:
        #return False, route, BigNumber
        opt_tri = mip_checking(problem_instance, route)[0]
        if len(opt_tri) == 0:            
            return False, route, BigNumber
    # Get the distances for each route segment, starting and ending at 0
    
    
    if problem_instance.mvrp:        
        feas, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, route)
        
    else:
        cost = get_min_multi_trip_cost(problem_instance, route)

    
    return feas, route, cost 

def remove_infeasible_nodes_slack(problem_instance, route):
    pool = []    
    initial_served = len(route)
    

    removed = True
    while removed:
        service_starts, latest_times_feasible, arrivals = get_latest_times(problem_instance, route)
        slacks = [[route[i].index, latest_times_feasible[i] - arrivals[i]] for i in range(len(service_starts))]
        slacks.sort(key=lambda x: x[1], reverse = False)
        
        removed = False
        if len(pool) + len(route) != initial_served:
            print("stop")
        feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, route)
        if feasibility:
            if not is_feasible_list(problem_instance, route):
                problem_instance.feasible_list.append([x.index for x in route])
            break
        
        if len(pool) + len(route) != initial_served:
            print("stop")
        ii = slacks[0][0]    
        node =[x for x in route if x.index == ii][0]       
        sequence_init = [problem_instance.customers[x.index] for x in route if x.index != ii]
        #sequence_init.pop(ii)
        feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, sequence_init)
            
        if not feasibility:            
            route.remove(node)
            pool.append(node)     
            removed = True
            if len(pool) + len(route) != initial_served:
                print("stop")                      
    
    
    return pool
def remove_infeasible_nodes_binary(problem_instance, route, pool = []):
    #loop breaking
    removed_bef = [x.index for x in pool]
    route = [x for x in route if x.index not in removed_bef] #remove the nodes removed before
    feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, route)
    if feasibility:
        return pool #breaking point for the recurive function
    
    start = 1
    
    if len(route) > 2:
        half = len(route)//2    
        removed_right =  route[:half]  #remove left part
        removed_left =  route[half:] #remove right part

        before = len(pool)
        remove_infeasible_nodes(problem_instance, removed_right, pool)   

             
        start = len(removed_right) - 1 #index first removed
        remove_infeasible_nodes(problem_instance, removed_left, pool)     

        if len(pool) > before: #any change?

            removed_bef = [x.index for x in pool] #remove
            
            route = [x for x in route if x.index not in removed_bef]

            
            feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, route)
            if feasibility:
                return pool #breaking point for the recurive function
        
    
    initial_served = len(route) 
        
    for ii, node in enumerate(route):
        if ii < start:
            continue
        sequence_init = copy.deepcopy(route[:ii+1])
        #sequence_init.pop(ii)
        feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, sequence_init)
        if not feasibility:                
            route.remove(node)
            pool.append(node)                
            
            if len(pool) + len(route) != initial_served:
                print("stop")
            return remove_infeasible_nodes(problem_instance, route, pool)   
    
    return pool

def remove_infeasible_nodes(problem_instance, route):
    pool = []
    
    initial_served = len(route)
    start = 0
    
    if len(route) > 4:
        divisions = [0.25, 0.5, 0.75, 0.9, 0.95]

        divisions = [ int(len(route)*x) for x in divisions]

        divisions = set(divisions)
        for length in divisions:
            seq_left = route[:length]
            feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, seq_left)
            if feasibility:
                start = length
            else:
                break
    
    removed = True
    while removed:
        removed = False
        if len(pool) + len(route) != initial_served:
            print("stop")
        feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, route)
        if feasibility:
            if not is_feasible_list(problem_instance, route):
                problem_instance.feasible_list.append([x.index for x in route])
            break
        
        if len(pool) + len(route) != initial_served:
            print("stop")
        for ii, node in enumerate(route):
            if ii < start:
                continue
            sequence_init = copy.deepcopy(route[:ii+1])
            #sequence_init.pop(ii)
            feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, sequence_init)
            if not feasibility:                
                route.remove(node)
                pool.append(node)                
                start = ii
                removed = True
                if len(pool) + len(route) != initial_served:
                    print("stop")
                break          
    
    
    return pool

def plot_found_solutions(problem_instance):

    mvrp_stats = problem_instance.mvrp
    problem_instance.mvrp = False

    found_unique = []

    for time, sequence in problem_instance.points_found:
        routes = []
        for seq in sequence:
            r = [[node for node in problem_instance.nodes if node.index == ind][0] for ind in seq]
            routes.append(r)
        
        sol = check_VRP_solution(problem_instance, routes)
        found_unique.append([time, sol.travelStreet, sol.nSEVs,  sol.is_feasible])

    problem_instance.mvrp = mvrp_stats

    problem_instance.points_found = [x for x in found_unique if x[3] == True]

    fleet = [x[2] for x in problem_instance.points_found]
    distance = [x[1] for x in problem_instance.points_found]
    time = [x[0] for x in problem_instance.points_found]
    plt.scatter(y=distance,x=time, c = "black", marker='o' )
    plt.title("relaxed cost of the solutions found over time")
    #plt.scatter(time, fleet, c = "black", marker='X', label='fleet size of the solutions found over time')
    #plt.legend(loc="upper right")
    plt.xlabel('time')
    plt.ylabel('relaxed cost')
    
    plt.savefig(problem_instance.name.split(".")[0]+"_distance_solutions_over.png")
    plt.close()

    plt.scatter( y=fleet,x=time, c = "black", marker='o', label='fleet size of the solutions found over time')
    plt.title("fleet size of the solutions found over time")
    #plt.legend(loc="upper right")
    plt.xlabel('time')
    plt.ylabel('fleet size')
    
    plt.savefig(problem_instance.name.split(".")[0]+"_fleet_solutions_over.png")
    plt.close()

    return
def add_feasible_points(problem_instance, solution_indices):
    found_time = round(time.time() - problem_instance.modelStartTime,2)  
    problem_instance.points_found.append([found_time, solution_indices])
    return

def add_to_feasible_solution(problem_instance,solution, tabu_search = False): 

    if sum(len(x) for x in solution) != len(problem_instance.customers): #partial solution
        return False

    list_of_found =  problem_instance.heuristics_solutions  
    if tabu_search:
        list_of_found =  problem_instance.tabu_list
    if not list_of_found:
        sequence_current = [[x.index for x in route] for route in solution]
        sequence_current.sort(key=lambda x: x[0])
        list_of_found.append(sequence_current)
        add_feasible_points(problem_instance, sequence_current)
        return True
    # Assuming we have a list of lists with each inner list containing an object with an attribute called 'attribute_to_sort_by'
    # Sorting the list of lists by the value of the attribute for the first element
    list_of_found.sort(key=lambda x: len(x))
    
    sequence_current = [[x.index for x in route] for route in solution]
    sequence_current.sort(key=lambda x: x[0])
    
    for sequence_found in list_of_found:
        found = True
        if len(sequence_current) != len(sequence_current):
            found = False
            continue       
        
        if sequence_current != sequence_found:
            found = False
        else:
            break

    if not found:
        list_of_found.append(sequence_current)
        
        add_feasible_points(problem_instance, sequence_current)
        return True
    else:
        return False
def check_found_solutions(problem_instance):     
    solutions_to_scheck = []

    for solution in problem_instance.heuristics_solutions:
        sequence_current = solution
        sequence_current.sort(key=lambda x: x[0])
        if sequence_current not in problem_instance.checked_solutions:
            problem_instance.checked_solutions.append(copy.deepcopy(sequence_current))        
            routes = [[problem_instance.nodes[x] for x in sequence] for sequence in sequence_current]
            current_solution = check_VRP_solution(problem_instance, routes) #relaxed VRP costs
            if current_solution.is_feasible:
                solutions_to_scheck.append([current_solution.objective, routes])

    best_street = math.inf
    best_FEV_number = math.inf
    best_SEV_feasible = math.inf

    if len(solutions_to_scheck) > 0:
        solutions_to_scheck.sort(key=lambda x: len(x[1]))
        
        if problem_instance.best_found != None:
            best_FEV_number = problem_instance.best_found.nFEVs
            best_SEV_feasible = problem_instance.best_found.nSEVs    
    
    multi_trips = []
    for solution in solutions_to_scheck:
        solution_s = check_multi_trips(problem_instance, solution[1])
        multi_trips.append(solution_s)
    solutions_to_scheck = multi_trips
    solutions_to_scheck.sort(key=lambda x: x.objective)
    min_check = max(50, len(solutions_to_scheck)) 
    for solution_mt in solutions_to_scheck[:min_check]:         
        routes = solution_mt.routes
        if problem_instance.minVessel_bound == best_FEV_number and len(routes) > best_SEV_feasible:
            continue

        if problem_instance.minVessel_bound == best_FEV_number and len(routes) > best_SEV_feasible:
            continue
        solution_two_echelon = check_MVRP_solution(problem_instance, routes)  #MVRP costs
        
        if solution_two_echelon.is_feasible:
            obj, routes, solution_two_echelon = solution_two_echelon.objective, solution_two_echelon.routes_sevs, solution_two_echelon
    
            if obj > problem_instance.cutoff - 1:
                solution_two_echelon.checked_optimality = True
                continue
            if problem_instance.minVessel_bound == best_FEV_number and len(routes) > best_SEV_feasible:
                solution_two_echelon.checked_optimality = True
                continue
            solution_two_echelon = check_2EVRP_solution(problem_instance, solution_two_echelon) #two echelon costs
            
            solution_two_echelon.checked_optimality = True
            if solution_two_echelon.is_feasible:      
                before = 1000
                after = 0
                status_before = problem_instance.consolidated
                problem_instance.consolidated = True
                while abs(after - before) > 1 :
                    before = round ( solution_two_echelon.travelTotal,1)
                
                    solution_two_echelon = check_2EVRP_solution_consolidated(problem_instance, solution_two_echelon)
                    
                    after = round ( solution_two_echelon.travelTotal,1) #if new solution is provided

                problem_instance.feas_solutions.append(solution_two_echelon)
                problem_instance.consolidated = status_before
                
                if solution_two_echelon.nFEVs < best_FEV_number:
                    best_FEV_number = solution_two_echelon.nFEVs
                if solution_two_echelon.objective < best_street:
                    best_street =  solution_two_echelon.travelStreet  + problem_instance.cost_vehicle_use*solution_two_echelon.nSEVs
                    if solution_two_echelon.nSEVs < best_SEV_feasible:
                        best_SEV_feasible = solution_two_echelon.nSEVs
            else:
                problem_instance.inf_solutions.append(solution_two_echelon)   
        else:
            problem_instance.inf_solutions.append(solution_two_echelon)   
        
    return

#can be called for VRP, MVRP, and 2e-MVRP
def check_feasibility(problem_instance,  routes,  VRP_only = True, multi_trip = False, single_route_MVRP = False, multi_trip_solution = None): #integrated solution or single

    if VRP_only:
        #add transfers here
        return check_VRP_solution(problem_instance, routes)
       
    elif multi_trip:
        if single_route_MVRP: #SINGLE - PARTIAL MVRP
            feasibility, cost, visits, routes = MathModels.street_problem_MIP(problem_instance, problem_instance.timelimit_SEmodel, routes) #single array is sent
            return feasibility, cost
        else: #COMPLETE MVRP
            return check_MVRP_solution(problem_instance, routes)
    
    else:#two echelon
        if not multi_trip_solution.is_feasible:
            return multi_trip_solution
        return check_2EVRP_solution(problem_instance, multi_trip_solution)   
      
def check_multi_trips(problem_instance, routes):   
    current_solution =  sev.two_echelon_solution() 
    for route in routes: 
        newRoute = sev.street_vehicle()
        newRoute.sequence.extend(route)
        feasibility, cost, visits, route = MathModels.street_problem_MIP(problem_instance, problem_instance.timelimit_SEmodel, route) #single array is sent
        newRoute.cost = cost
        newRoute.visits = visits
        newRoute.is_feasible = feasibility
        current_solution.routes_sevs.append(newRoute)

    current_solution.routes = routes    
    resultss = [is_route_feasible(problem_instance, route)[0:] for route in routes]
    current_solution.objective =  round(sum(result[2] for result in resultss),2)
    current_solution.travelStreet =  round(sum(result[2] for result in resultss),2)
    current_solution.is_feasible =  all(result[0] for result in resultss)
    current_solution.nSEVs = len(routes)

    return current_solution
  
def check_VRP_solution(problem_instance, routes):   
    current_solution =  sev.two_echelon_solution() 
    for route in routes: 
        newRoute = sev.street_vehicle()
        newRoute.sequence.extend(route)
        current_solution.routes_sevs.append(newRoute)
    current_solution.routes = routes    
    resultss = [is_route_feasible(problem_instance, route)[0:] for route in routes]
    current_solution.objective =  round(sum(result[2] for result in resultss),2)
    current_solution.travelStreet =  round(sum(result[2] for result in resultss),2)
    current_solution.is_feasible =  all(result[0] for result in resultss)
    current_solution.nSEVs = len(routes)

    return current_solution
def check_2EVRP_solution_consolidated(problem_instance, solution_mvrp): 
    if len(problem_instance.satellites) == 1:
        return solution_mvrp
    
    routes = [copy.deepcopy(route.sequence) for route in solution_mvrp.routes_sevs]

    satellites_before = [satelitte for satelitte in problem_instance.satellites]

    satellites_used = [satellite for satellite in problem_instance.satellites if satellite.index in solution_mvrp.satellites_used]
    satellites_consolidated = [satellite for satellite in satellites_before if satellite.index != solution_mvrp.satellite_least_used]

    problem_instance.satellites = satellites_consolidated
    solution = check_multi_trips(problem_instance, routes)
    if not solution.is_feasible:
        problem_instance.satellites = satellites_before
        return solution_mvrp
    solution = check_MVRP_solution(problem_instance, routes)
    if not solution.is_feasible:
        problem_instance.satellites = satellites_before
        return solution_mvrp
    solution_two_echelon = check_2EVRP_solution(problem_instance, solution)

    problem_instance.satellites = satellites_before

    if not solution_two_echelon.is_feasible:
        return solution_mvrp
    else:    
        if solution_two_echelon.objective > solution_mvrp.objective-0.1:    
            return solution_mvrp
        else:
            return solution_two_echelon

def check_MVRP_solution(problem_instance, routes):
    final_solution = sev.two_echelon_solution() #for single routes, no need to generate vehicles every time
    final_solution.objective = BigNumber    
    final_solution.is_feasible = True  

    #check street solution
    #trip generation, street schedules feasibility check spatio-temporal
    assignments = []
    demand_to_supply = 0
    infeasible_nodes = []
    
    #remove infeasible nodes
    for route in routes:
        demand_to_supply += sum([x.demand for x in route])
        feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance, problem_instance.timelimit_SEmodel, route)
        if not feasibility:            
            pool = remove_infeasible_nodes(problem_instance, route)
            infeasible_nodes.extend(pool)
            route = [x for x in route if x not in pool]

            if route:                
                feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance, problem_instance.timelimit_SEmodel, route)
                if feasibility:
                    newRoute = sev.street_vehicle()
                    newRoute.sequence.extend(route) 
                    newRoute.cost = cost
                    newRoute.visits = visits
                    assignments.extend(visits)
                    final_solution.routes_sevs.append(newRoute)
                else:
                    final_solution.is_feasible = False
                    return final_solution

        else:
            if not is_feasible_list(problem_instance, route):
                problem_instance.feasible_list.append([x.index for x in route])
            newRoute = sev.street_vehicle()
            newRoute.sequence.extend(route) 
            newRoute.cost = cost
            newRoute.visits = visits
            assignments.extend(visits)
            final_solution.routes_sevs.append(newRoute)
    
    #re-insert unserved and repair the solution
    routes = [x.sequence for x in final_solution.routes_sevs]
    if infeasible_nodes:
        print("infeasible nodes", len(infeasible_nodes))
        print(len(infeasible_nodes) + sum(len(x) for x in routes))
        final_solution = sev.two_echelon_solution() #reinitilize
        final_solution.is_feasible = True
        mvrp_status_before = problem_instance.mvrp
        problem_instance.mvrp = True
        routes, infeasible_nodes = k_best_insertion_repair(problem_instance, routes, infeasible_nodes, 3)
        problem_instance.mvrp = mvrp_status_before
        print(len(infeasible_nodes) + sum(len(x) for x in routes))
        if not infeasible_nodes: #feasible
            assignments = []    
            for route in routes:
                feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance,  problem_instance.timelimit_SEmodel, route)
                if not feasibility:
                    final_solution.is_feasible = False
                    return final_solution
                else:
                    newRoute = sev.street_vehicle()
                    newRoute.sequence.extend(route) 
                    newRoute.cost = cost
                    newRoute.visits = visits
                    assignments.extend(visits)
                    final_solution.routes_sevs.append(newRoute)
            
            routes = [x.sequence for x in final_solution.routes_sevs] #update routes
            add_to_feasible_solution(problem_instance, routes)
        else:
            final_solution.is_feasible = False
            return final_solution
    
    #routes = [x.sequence for x in final_solution.routes_sevs]
    #solution is feasible, now generate water level problem
    arcList = []    
    for r in routes:
        arcList.extend(MathModels.generate_arcs(r))
    final_solution.arcList = arcList
    transfer_points = [i for [i,p] in assignments]
    satellites = [p for [i,p] in assignments]
    #get the trip loads
    loads = MathModels.generate_triploads(problem_instance, arcList, transfer_points)
    for ii, trip in enumerate(assignments):
        trip.append(loads[ii])
    sum_trips = sum([x[2] for x in assignments])

    if sum_trips != demand_to_supply:
        print("Error in demand supply check", sum_trips, demand_to_supply)
    final_solution.nSEVs= len(final_solution.routes_sevs)
    final_solution.objective = sum([x.cost for x in final_solution.routes_sevs]) + problem_instance.cost_vehicle_use*final_solution.nSEVs
    final_solution.travelStreet = sum([x.cost for x in final_solution.routes_sevs])
    final_solution.trips = assignments
    final_solution.transfer_points = transfer_points
    final_solution.satellites_used = satellites
    routes = [route.sequence for route in final_solution.routes_sevs] #= routes
    
    served = sum(len(route) for route in routes)
    if served != len(problem_instance.customers):
        print ("stop")
    #place, time, load is found
    tasks = []    
    for ii, trip in enumerate(assignments):
        satellite = next(x for x in problem_instance.satellites if x.index == trip[1]) 
        task = instanceGenerator.Node("S", (satellite.index), satellite.xcoordinate, satellite.ycoordinate, trip[2], satellite.start_interval,satellite.end_interval,
                            satellite.unloading_time)
        task.transfer_point = trip[0]
        task.load = trip[2] #used one
        task.demand = trip[2] #might be needed for simulating operations and also math formulations
        tasks.append(task)
        task.name = satellite.name
    final_solution.transshipments = tasks

    used = [0 for _ in final_solution.satellites]
    used = [sum(x.demand for x in final_solution.transshipments if x.index == satellite.index) for satellite in  problem_instance.satellites]
    
    min_load = min(x for x in used if x > 0)
    index_least = used.index(min_load)
    final_solution.satellite_least_used = problem_instance.satellites[index_least].index
    routes = [route.sequence for route in final_solution.routes_sevs]    

    return final_solution
def check_2EVRP_solution(problem_instance, multi_trip_solution, remove_satellite = False):
    final_solution = multi_trip_solution

    routes = [x.sequence for x in final_solution.routes_sevs]
    #remove below after verifying feasibility of multi trip solution any time called
    hasks = []
    for route in routes:
        feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance, problem_instance.timelimit_SEmodel, route)
        hasks.append(visits)
        if not feasibility:
            print("infeasible solution for multi-trip") #normally should not happen since it is checked before
            final_solution.is_feasible = False
            return final_solution

    tasks = final_solution.transshipments
    for task in tasks:
        print(task.name, task.transfer_point, task.load)

    if sum(x.demand for x in tasks) != sum(x.demand for x in problem_instance.nodes if x.index != 0):
        print("Error in demand supply check", sum(x.demand for x in tasks), sum(x.demand for x in problem_instance.nodes if x.index != 0))
    
    #check water solution
    SE_cost = final_solution.travelStreet + problem_instance.cost_vehicle_use*len(final_solution.routes_sevs)
    feas, obj_value, fleet, distance_travel_cost, used_arcs, locations_arcs = FEVmodels.water_problem_MIP(problem_instance,tasks, routes,  SE_cost)
    final_solution.is_feasible = feas
    if not feas:
        
        return final_solution
    
    final_solution.water_objective = obj_value
    final_solution.water_fleet = fleet
    final_solution.nFEVs = fleet
    final_solution.travelWater = distance_travel_cost
    final_solution.travelTotal = round(final_solution.travelStreet + final_solution.travelWater, 2)
    final_solution.water_used_arcs = used_arcs #operation sequence (temporal location)
    final_solution.routes_fevs = locations_arcs #satellite sequence (spatial location)
    final_solution.objective = final_solution.travelStreet + final_solution.travelWater + problem_instance.cost_vehicle_use*(final_solution.nSEVs + final_solution.nFEVs)
    final_solution.checked_optimality = True

    routes =  FEVmodels.get_routes(problem_instance, used_arcs)
    sequences  = [[[task for task in tasks if task.transfer_point == i][0] for i in route_curr] for route_curr in routes]
    loads = [sum([task.load for task in sequence]) for sequence in sequences]
    sum_loads = sum(loads)
    sum_tasks = sum([task.demand for task in tasks])
    sum_demand = sum([node.demand for node in problem_instance.customers])
    served_tasks = sum([len(sequence) for sequence in sequences])
    total_tasks = len(tasks)
    
    final_solution.operation_sequence_fevs = sequences
    
    #number of satellites visited
    list_locs = []
    for [i,j] in final_solution.routes_fevs:
        if i not in list_locs:
            list_locs.append(i)

    final_solution.nVisited_locations = len(list_locs) - 1
    
    #update incumbent
    
    if problem_instance.cutoff > final_solution.objective + 0.0001:
        problem_instance.cutoff = final_solution.objective
        updated = False
        if problem_instance.best_found is None:
            problem_instance.best_found = final_solution
        else:
            if final_solution.nFEVs <= problem_instance.best_found.nFEVs:
                 if final_solution.nFEVs < problem_instance.best_found.nFEVs:
                     updated = True
                 elif final_solution.nSEVs < problem_instance.best_found.nSEVs:
                     updated = True
                 elif final_solution.nSEVs == problem_instance.best_found.nSEVs and final_solution.nFEVs == problem_instance.best_found.nFEVs:
                     if final_solution.travelTotal < problem_instance.best_found.travelTotal :
                         updated = True
                 if updated:
                    problem_instance.best_found = final_solution
                    
            
    return final_solution

def check_MVRP_given(problem_instance, routes, assignments):
    final_solution = sev.two_echelon_solution() #reinitilize
    final_solution.is_feasible = True
    for route in routes:        
        feasibility, cost, visits, route_return = MathModels.street_problem_MIP(problem_instance, problem_instance.timelimit_SEmodel, route,None, assignments )
        newRoute = sev.street_vehicle()
        newRoute.sequence.extend(route) 
        newRoute.cost = cost
        newRoute.visits = visits
        final_solution.routes_sevs.append(newRoute)

    routes = [x.sequence for x in final_solution.routes_sevs]

    #solution is feasible, now generate water level problem
    arcList = []    
    for r in routes:
        arcList.extend(MathModels.generate_arcs(r))
    final_solution.arcList = arcList
    transfer_points = [i for [i,p] in assignments]
    #get the trip loads
    loads = MathModels.generate_triploads(problem_instance, arcList, transfer_points)
    for ii, trip in enumerate(assignments):
        trip.append(loads[ii])
    
    final_solution.nSEVs= len(final_solution.routes_sevs)
    final_solution.objective = sum([x.cost for x in final_solution.routes_sevs]) + problem_instance.cost_vehicle_use*final_solution.nSEVs
    final_solution.travelStreet = sum([x.cost for x in final_solution.routes_sevs])
    final_solution.trips = assignments
    routes = [route.sequence for route in final_solution.routes_sevs] #= routes
    
    served = sum(len(route) for route in routes)
    if served != len(problem_instance.customers):
        print ("stop")
    #place, time, load is found
    tasks = []    
    for ii, trip in enumerate(assignments):
        satellite = next(x for x in problem_instance.satellites if x.index == trip[1]) 
        task = instanceGenerator.Node("S", (satellite.index), satellite.xcoordinate, satellite.ycoordinate, trip[2], satellite.start_interval,satellite.end_interval,
                            satellite.unloading_time)
        task.transfer_point = trip[0]
        task.load = trip[2] #used one
        task.demand = trip[2] #might be needed for simulating operations
        tasks.append(task)
        task.name = satellite.name
    final_solution.transshipments = tasks
    routes = [route.sequence for route in final_solution.routes_sevs]

    final_solution = check_2EVRP_solution(problem_instance, final_solution)
    return final_solution
def read_initial_SEV_routes_use_case(problem_instance, file_content): #single uncapacitated VRP route
    with open(file_content, 'r') as file:
        line = file.readline()
        
        while line:
            indices = line.strip().split()
            break
    routes = []
    route = []
    for ind in indices:
        node = [node for node in problem_instance.nodes if node.index == int(ind)][0]
        route.append(node)
    routes.append(route)
    return routes
def read_initial_SEV_routes_benchmark(problem_instance, demand_set_given,  usecase = True):
    
    if usecase:
        projectPath = os.path.dirname(os.path.abspath(__file__))
        files_use_case = projectPath+"/usecase/initials/"
        text = "init_demand_set"+str(demand_set_given)+"_"+str(len(problem_instance.customers))+"_best.txt"

        file_content = files_use_case + text
    else:
        file_content = "initials_multi\init_demand_set" + str(problem_instance.name) +"_"+str(len(problem_instance.customers))+ ".txt"
    with open(file_content, 'r') as file:
        line = file.readline() #"problem demand set\t"+ str(problem_instance.name)
        line = file.readline() # number of customers\t"+ str(len(problem_instance.customers)) 
        line = file.readline() #"SEV capacity\t"+ str(problem_instance.capacity_SEV) 
        line = file.readline() # file.write("vehicle capacity\t"+ str(problem_instance.capacity_FEV) + "\n")        
        line = file.readline() # file.write("runtime\t"+ str(runtime) + "\n")
        
        #the routes with customer indices
        line = file.readline()
        sequences_given = line.strip().split()

        routes_given = []
        visited = []
        route_current = []

        for item in sequences_given:
            if item != "-":
                ind = int(item)
                visited.append(ind)
                route_current.append(ind)
            else:
                routes_given.append(route_current)
                route_current = []
    
    initial_routes_asynch = [[problem_instance.nodes[i] for i in route] for route in routes_given]
    add_to_feasible_solution(problem_instance, initial_routes_asynch)
    check_found_solutions(problem_instance)
    return problem_instance.best_found



def write_out_solution(problem_instance, final_solution):
    
    with open("init_demand_set" + str(problem_instance.name) +"_"+str(len(problem_instance.customers))+"_"+str(problem_instance.synch_check)+"_"+str(len(problem_instance.satellites))+ ".txt", 'w') as file:
        file.write("problem demand set\t"+ str(problem_instance.name) + "\n")
        file.write("number of customers\t"+ str(len(problem_instance.customers)) + "\n")
        file.write("SEV capacity\t"+ str(problem_instance.capacity_SEV) + "\n")
        file.write("vehicle capacity\t"+ str(problem_instance.capacity_FEV) + "\n")
        
        runtime = round(time.time() - problem_instance.modelStartTime, 0)
        file.write("runtime\t"+ str(runtime) + "\n")
        
        for route in final_solution.routes_sevs:
            for node in route.sequence:
                file.write(str(node.index) + "\t")
            file.write( "-"+ "\t")
        file.write("\n")
        
        #total demand
        demands = [x.demand for x in problem_instance.customers]
        file.write("total demand\t"+ str(sum(demands)) + "\n")
        #minimum number of trips
        file.write("minimum number of trips\t"+ str(math.ceil(sum(demands)/problem_instance.capacity_SEV)) + "\n")
        file.write("minimum number of vessels\t"+ str(math.ceil(sum(demands)/problem_instance.capacity_FEV)) + "\n")
        loads_trips = [x.load for x in final_solution.transshipments]
        loads_satellites = [sum(x.load for x in final_solution.transshipments if x.index == satellite.index) for satellite in  problem_instance.satellites]
        used_satellites = [sum(1 for x in final_solution.transshipments if x.index == satellite.index) for satellite in  problem_instance.satellites]

        used_satellites = [x for x in used_satellites if x > 0]
        loads_satellites = [x for x in loads_satellites if x > 0]
        file.write("nof trips\t"+ str(len( final_solution.transshipments)) + "\n")
        file.write("load per trip\t"+ str(loads_trips)+ "\n")
        file.write("used satellites\t"+ str(used_satellites)+ "\n") 
        file.write("loads_satellites\t"+ str(loads_satellites)+ "\n") 
        file.write("objective\t"+ str(final_solution.objective)+ "\n") 
        file.write("is feasible\t"+ str(final_solution.is_feasible)+ "\n") 
        file.write("nSEVs\t"+ str(final_solution.nSEVs)+ "\n") 
        file.write("nFEVs\t"+ str(final_solution.nFEVs)+ "\n") 
        file.write("travel street\t"+ str(final_solution.travelStreet)+ "\n") 
        file.write("travel water\t"+ str(final_solution.travelWater)+ "\n") 
        if final_solution.is_feasible:
            file.write("n visited locations\t"+ str(final_solution.nVisited_locations)+ "\n") 

            cumulative = 0
            for route in final_solution.routes_sevs:
                for node in route.sequence:
                    cumulative+=node.demand
                    file.write(str(cumulative) + "\t")
                file.write( "-"+ "\t")
            file.write("\n")

            file.write("demand to supply "+ str(sum(node.demand for node in problem_instance.customers))+ "\n") 
            file.write("supplied demand by SEVs " + str(cumulative)+ "\n") 
            file.write("supplied demand by FEVs " + str(sum(x.demand for x in final_solution.transshipments))+ "\n") 
        
            file.write("n visited satellites by name")
            chosen_id = []
            for id in final_solution.satellites_used:
                if id not in chosen_id:
                    chosen_id.append(id)
                    file.write("-"+ str(problem_instance.network[id].name))
        

    return 