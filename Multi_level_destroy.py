import random 
import numpy
import SEVs 
import math 
import time
import copy
import MathModels 
import Repair_heuristcs as RH
import local_search as LS
import alns as ALNS
import SEVs
BigNumber = 1000000
#define set up for ALNS operators instead of creating at each time required such as probs, scores, used times and etc.

def Solve_by_decomposition(random_num, problem_instance, nstart = 10, repLS = 5, iterLS = 50, iterALNS = 500):
    random.seed(random_num+problem_instance.its_iter) 

    problem_instance.customers = [node for node in problem_instance.nodes if node.index>0]

    start_time = time.time()   
    if problem_instance.best_found is None:
        construction_two_echelon = multi_start_initial( problem_instance, nstart )       #returns a two-echelon solution    
        #construction_two_echelon = MathModels.street_problem_MIP(problem_instance, 100, single= None, multiple = None, transfers_in = None)
        RH.check_found_solutions(problem_instance)
    
    
    construction_two_echelon = problem_instance.best_found

    percentage_imp1, routes1 = Local_search_incumbent(problem_instance, construction_two_echelon, replications = repLS, iters = iterLS) 
    RH.check_found_solutions(problem_instance)
    
    improved_inital_solution = problem_instance.best_found
    
    if construction_two_echelon == None:
        percentage_imp1 = 100
    else:
        percentage_imp1 = 100*(improved_inital_solution.travelTotal - construction_two_echelon.travelTotal)/construction_two_echelon.travelTotal
    #local search
    solution, time2, incumbent = Solve_by_multi_level_ALNS(problem_instance, improved_inital_solution, max_iterations=iterALNS) 
    alns_final_solution = problem_instance.best_found

    if alns_final_solution == None:
        percentage_imp1 = 100
    else:
        percentage_imp2 = 100*(alns_final_solution.travelTotal - improved_inital_solution.travelTotal)/improved_inital_solution.travelTotal
    
    percentage_imp2, routes1 = Local_search_incumbent(problem_instance, alns_final_solution, replications = repLS, iters = iterLS) 
    RH.check_found_solutions(problem_instance)  
    final_solution = problem_instance.best_found

    if final_solution == None:
        percentage_imp1 = 100
    else:
        percentage_imp3 = 100*(final_solution.travelTotal - alns_final_solution.travelTotal)/alns_final_solution.travelTotal
    
    
    problem = problem_instance
    
    with open(problem.report , 'a') as file:
            file.write("problem demand set\t"+ str(problem.name) + "\t")
            file.write("number of customers\t"+ str(len(problem.customers)) + "\t")
            file.write("run time secs\t"+ str(time.time() - start_time) + "\t")
            #file.write("FINAL SOLUTION STATS relax, alns, and two echelon objective\t"+ str(relaxed_cost_final)+ "\t"+ str(relaxed_cost_alns)+ "\t"+ str(final_solution.objective)+ "\t") 
            file.write("is feasible\t"+ str(final_solution.is_feasible)+ "\t") 
            file.write("nSEVs\t"+ str(final_solution.nSEVs)+ "\t") 
            file.write("nFEVs\t"+ str(final_solution.nFEVs)+ "\t") 
            file.write("total travel\t"+ str(final_solution.travelStreet +final_solution.travelWater)+ "\t") 
            file.write("street travel\t"+ str(final_solution.travelStreet)+ "\t") 
            file.write("water travel\t"+ str(final_solution.travelWater)+ "\t" ) 
            file.write("nof transfers\t"+ str(len(final_solution.transshipments))+ "\t") 
    
    final_solution = alns_final_solution
    with open(problem.report , 'a') as file:
            file.write("improvement final LS and alns solution\t"+ str(percentage_imp3) + "\t")
            file.write("is feasible\t"+ str(final_solution.is_feasible)+ "\t") 
            file.write("nSEVs\t"+ str(final_solution.nSEVs)+ "\t") 
            file.write("nFEVs\t"+ str(final_solution.nFEVs)+ "\t") 
            file.write("total travel\t"+ str(final_solution.travelStreet +final_solution.travelWater)+ "\t") 
            file.write("street travel\t"+ str(final_solution.travelStreet)+ "\t") 
            file.write("water travel\t"+ str(final_solution.travelWater)+ "\t" ) 
            file.write("nof transfers\t"+ str(len(final_solution.transshipments))+ "\t") 
    final_solution = improved_inital_solution
    with open(problem.report , 'a') as file:
            file.write("improvement in ALNS and LS solution\t"+ str(percentage_imp2) + "\t")
            file.write("is feasible\t"+ str(final_solution.is_feasible)+ "\t") 
            file.write("nSEVs\t"+ str(final_solution.nSEVs)+ "\t") 
            file.write("nFEVs\t"+ str(final_solution.nFEVs)+ "\t") 
            file.write("total travel\t"+ str(final_solution.travelStreet +final_solution.travelWater)+ "\t") 
            file.write("street travel\t"+ str(final_solution.travelStreet)+ "\t") 
            file.write("water travel\t"+ str(final_solution.travelWater)+ "\t" ) 
            file.write("nof transfers\t"+ str(len(final_solution.transshipments))+ "\t"  ) 
    final_solution = construction_two_echelon
    with open(problem.report , 'a') as file:
            file.write("improvement first LS and inital solution \t"+ str(percentage_imp1) + "\t")
            file.write("is feasible\t"+ str(final_solution.is_feasible)+ "\t") 
            file.write("nSEVs\t"+ str(final_solution.nSEVs)+ "\t") 
            file.write("nFEVs\t"+ str(final_solution.nFEVs)+ "\t") 
            file.write("total travel\t"+ str(final_solution.travelStreet +final_solution.travelWater)+ "\t") 
            file.write("street travel\t"+ str(final_solution.travelStreet)+ "\t") 
            file.write("water travel\t"+ str(final_solution.travelWater)+ "\t" ) 
            file.write("nof transfers\t"+ str(len(final_solution.transshipments))+ "\t"  + "\n") 

    return problem_instance.best_found

def Solve_by_multi_level_ALNS(problem_instance, start_solution, max_iterations=1500):    
    start = time.time()

    if type(start_solution) != list: # a two-echelon object
        current_solution = [copy.deepcopy(route.sequence) for route in start_solution.routes_sevs]
    else:        
        current_solution = copy.deepcopy(start_solution)

        
    generate_operators_statistics(problem_instance, max_iterations)
    pool = []
    temperature = problem_instance.cost_vehicle_use
    iteration = 0
    incumbent_iteration = 0

    feasible_solutions = {}
    feasible_solutions[iteration] = (len(current_solution), problem_instance.best_cost, current_solution)
    nof_accepted_worse_solutions = 0
    problem_instance.accepted_solutions = []   
    
    nonimproving = 0
    
    integrated_start= RH.check_feasibility(problem_instance, current_solution)
    
    deviations = []
    print("------------------start solution----------------"+"\t, fleet\t"+str(len(current_solution))+"\t, cost street\t"+str(integrated_start.travelStreet))
                
    while iteration < max_iterations - 1:   
        feasibility = True 
        iteration += 1   

        solution_start = copy.deepcopy(current_solution)

        roulette_type, removal_type, removal_number, repair_type = Select_operators(problem_instance)    
    
        partial_solution, pool = multi_direction_random_destroy_repair(problem_instance, current_solution, roulette_type, removal_type, removal_number)

        current_solution, pool = RH.k_best_insertion_repair(problem_instance, partial_solution, pool)
        score = calculate_score(problem_instance, solution_start, current_solution)
        update_scores(problem_instance,roulette_type, removal_type, removal_number, repair_type, score)

        
        if feasibility == False:
            current_solution = copy.deepcopy(solution_start)
            print( )
            with open(problem_instance.initial_report, 'a') as file:
                file.write("iteration " + str(iteration)+"\t------------------infeasible solution regarding MVRP----------------"+"\n")
    
            continue            
                
        #print in single line for better readability
        #print("iteration ", iteration, "\tVRP cost ", vrp_cost, "\tforward ", forward_cost, "\tbackward ", backward_cost, "\toptimal ", optimal_cost)
        with open(problem_instance.initial_report, 'a') as file:
            file.write("iteration " + str(iteration) +"\tfleet: " + str(len(current_solution)) + "\n")

                      
        integrated_solution = RH.check_feasibility(problem_instance,  current_solution)

        if len(pool) == 0 and integrated_solution.is_feasible and len(current_solution) <= len(solution_start):
            if type(current_solution) != list: # a two-echelon object
                current_solution = [copy.deepcopy(route.sequence) for route in integrated_solution.routes_sevs]         
                        
            cost_VRP_start = integrated_start.travelStreet# sum(RH.is_route_feasible(problem_instance, item)[4] for item in solution_start)
            cost_VRP_new = integrated_solution.travelStreet #sum(RH.is_route_feasible(problem_instance, item)[4] for item in current_solution)

            deviations.insert(0,abs(cost_VRP_new - cost_VRP_start))

            feasible_solutions[iteration] = (len(current_solution), cost_VRP_new, current_solution)
            
            print("feasible solution found at \t"+str(iteration)+"\twith cost\t"+str(cost_VRP_new)+"\t, fleet\t"+str(len(current_solution))+"\t, cost street\t"+str(integrated_solution.travelStreet))
           
            
            if len(current_solution) < len(solution_start):
                integrated_start = integrated_solution
                print("---------ASPIRATION---"+"\t, fleet\t"+str(len(current_solution))+"\t, cost street\t"+str(integrated_solution.travelStreet))
                problem_instance.accepted_solutions.append([cost_VRP_start, cost_VRP_new, current_solution, integrated_solution])
            elif len(current_solution) > len(solution_start)  or not simulated_annealing_acceptance(cost_VRP_new,cost_VRP_start, problem_instance):
                current_solution = copy.deepcopy(solution_start)
            else:
                print("------------------ACCEPTED----------------"+"\t, fleet\t"+str(len(current_solution))+"\t, cost street\t"+str(integrated_solution.travelStreet))
                integrated_start = integrated_solution
                nof_accepted_worse_solutions += 1
                problem_instance.accepted_solutions.append([cost_VRP_start, cost_VRP_new, current_solution, integrated_solution])
            
            if nonimproving > 500:
                break
        else:
            current_solution = copy.deepcopy(solution_start)

    
    problem_instance.accepted_solutions.sort(key = lambda x: len(x[2]))
    problem_instance.accepted_solutions.sort(key = lambda x: x[1])
    
    if len(problem_instance.accepted_solutions) > 0:
        best_integrated_solution = problem_instance.accepted_solutions[0][3]
    else:
        best_integrated_solution = integrated_start
    #best_integrated_solution = problem_instance.accepted_solutions[0][2]

    return best_integrated_solution, time.time() - start, incumbent_iteration

def simulated_annealing_acceptance(current_solution, solution_start, problem_instance):
    change = round(current_solution - solution_start,1)
    
    temperature = (solution_start)*(problem_instance.SAA_rate / problem_instance.its_iter )
    #if problem_instance.best_found is not None: 
    #    temperature = (problem_instance.best_found.travelStreet)*problem_instance.SAA_rate#(1 - iteration/max_iterations)    

    probability = math.exp(-change / temperature)

    return random.random() < probability
   
def multi_direction_random_destroy_repair(problem_instance, solution, roulette_type, removal_type, removal_number):
    
    pool = []
    new_solution = [copy.deepcopy(route) for route in solution]  # Create a copy of the original solution
    
    customers_to_select = []
    [customers_to_select.extend(route) for route in new_solution]  #node objects in routes are added, no pointer needed
#region destroy
    if removal_type == "route" and len(new_solution) != 1:        
        if roulette_type == "equal":
            weights = [1 for route in new_solution]
        else:
            #weights = [float(RH.is_route_feasible(problem_instance, route)[4]/len(route)) for route in new_solution]
            weights = [float(1/len(route)) for route in new_solution]
        route_removed = random.choices(new_solution, weights=weights, k=1)[0]
        pool.extend(route_removed) 
        new_solution.remove(route_removed)
        [customers_to_select.remove(node) for node in route_removed]
    else:
        if len(customers_to_select) > 1:
            if roulette_type == "equal":
                weights = [1 for _ in range(len(customers_to_select))]
            else:
                weights = get_weights_for_node_removal(problem_instance, customers_to_select, new_solution) 
            node_removed = random.choices(customers_to_select, weights=weights, k=1)[0]
            pool.append(node_removed)
            customers_to_select.remove(node_removed)

    
    #remove customers
    num_customers_to_remove =  int(len(customers_to_select)  * removal_number)
    num_customers_to_remove = max(num_customers_to_remove, 1)

    if roulette_type == "equal":
        weights = [1 for _ in range(len(customers_to_select))]
    else:
        weights = get_weights_for_nodes_removal(customers_to_select, pool) 

    #do it without replications
    for _ in range(num_customers_to_remove): 
        if len(customers_to_select) == 1: #leave at least one customer in the remaining routes
            break
        node_removed = random.choices(customers_to_select, weights=weights, k=1)[0]
        pool.append(node_removed)

        index_removed = customers_to_select.index(node_removed)   
        customers_to_select.remove(node_removed)   
        if roulette_type == "equal":      
            weights.remove(weights[index_removed])
        else:
            weights = get_weights_for_nodes_removal(customers_to_select, pool)        #update new weights since a customer updates the pool
            
            #re-updating pool each time after an item is added might reduce the chance of inserting the initiallyu removed ones but cretaes a local neighbourhood
            #alternative is just to remove the node, no uodate on the weights
    #remove customers

    #check the pointers of the nodes, missing a customer in removal part
    routes_new = []
    for route in new_solution:
        updated_route = [node for node in route if node not in pool]
        routes_new.append(updated_route)
    
    new_solution = [route for route in routes_new if route] 
    #check the operation
    x = 0
    for route in new_solution:
        x+=len(route) 
    if x + len(pool) != len(problem_instance.customers) :
        print(" more nodes are requested than customers")   
        #endregion  
    
    return new_solution, pool
def generate_operators_statistics(problem_instance, destroy_in = None):
    problem_instance.selection_counts = [1]*len(problem_instance.customers)
    problem_instance.selection_counts.insert(0, 0) #history based removal
    
    #roulette selection operators
    problem_instance.operators_roulette_selection_type = ["equal", "weighted"]
    problem_instance.scores_roulette_selection_type = [1]*len(problem_instance.operators_roulette_selection_type)
    problem_instance.scores_roulette_selection_type[0] = (10 / problem_instance.its_iter )*(problem_instance.scores_roulette_selection_type[1])
    problem_instance.nof_used_roulette_selection_type =  [0]*len(problem_instance.operators_roulette_selection_type)    
    problem_instance.nof_success_roulette_selection_type =  [0]*len(problem_instance.operators_roulette_selection_type)

    #removal type
    problem_instance.operators_removal_type = ["route", "node"]
    problem_instance.scores_removal_type = [1]*len(problem_instance.operators_removal_type)
    problem_instance.scores_removal_type[0] = (10 / problem_instance.its_iter)*(problem_instance.scores_removal_type[1]) #warm up
    problem_instance.nof_used_removal_type =  [0]*len(problem_instance.operators_removal_type)
    problem_instance.nof_success_removal_type =  [0]*len(problem_instance.operators_removal_type)

    removal_types = [0.05, 0.15, 0.25, 0.35]
    #if destroy_in is not None:
    #    removal_types = [destroy_in]
    #number of removals 
    problem_instance.operators_removal_number = removal_types#[ (0.05*i) + 0.05 for i in range(problem_instance.nof_destroy_operators)] 
    problem_instance.scores_removal_number  = [1 ]*len(problem_instance.operators_removal_number) #generate weights same size as operators
    problem_instance.nof_used_removal_number = [0]*len(problem_instance.operators_removal_number)
    problem_instance.nof_success_removal_number = [0]*len(problem_instance.operators_removal_number)

    #repair_operators - only single repair operator used 
    repair_types = [1,2,3,4]
    problem_instance.operators_repair_type = repair_types # [ i+1 for i in range(problem_instance.nof_destroy_operators)]     
    problem_instance.scores_repair_type  = [1 ]*len(problem_instance.operators_repair_type) #generate weights same size as operators
    problem_instance.nof_used_repair_type = [0]*len(problem_instance.operators_repair_type)
    problem_instance.nof_success_repair_type = [0]*len(problem_instance.operators_repair_type)
    return

def Select_operators(problem_instance):
    #select randomly    
    roulette_type = random.choices(problem_instance.operators_roulette_selection_type, problem_instance.scores_roulette_selection_type, k=1)[0] 
    removal_type = random.choices(problem_instance.operators_removal_type, problem_instance.scores_removal_type, k=1)[0] 
    removal_number = random.choices(problem_instance.operators_removal_number, problem_instance.scores_removal_number, k=1)[0] 
    repair_type = random.choices(problem_instance.operators_repair_type, problem_instance.scores_repair_type, k=1)[0] 

    #update use
    problem_instance.nof_used_roulette_selection_type[problem_instance.operators_roulette_selection_type.index(roulette_type)] +=1
    problem_instance.nof_used_removal_type[problem_instance.operators_removal_type.index(removal_type)] +=1
    problem_instance.nof_used_removal_number[problem_instance.operators_removal_number.index(removal_number)] +=1
    problem_instance.nof_used_repair_type[problem_instance.operators_repair_type.index(repair_type)] +=1

    return roulette_type, removal_type, removal_number, repair_type
def calculate_score(problem_instance, solution, updated_solution):
    score = 0
    energy_old = sum(RH.is_route_feasible(problem_instance, route)[2] for route in solution)
    energy_new = sum(RH.is_route_feasible(problem_instance, route)[2] for route in updated_solution)

    if abs(energy_new  - energy_old) < 1:
        score = 0.2
    elif energy_new > energy_old :
        score = 0
    else:
        score = 1

    return score

def update_scores(problem_instance, roulette_type, removal_type, removal_number, repair_type, score):
    problem_instance.scores_roulette_selection_type[problem_instance.operators_roulette_selection_type.index(roulette_type)] += score
    problem_instance.scores_removal_type[problem_instance.operators_removal_type.index(removal_type)] += score
    problem_instance.scores_removal_number[problem_instance.operators_removal_number.index(removal_number)] += score
    problem_instance.scores_repair_type[problem_instance.operators_repair_type.index(repair_type)] += score

    if score > 0:
        problem_instance.nof_success_roulette_selection_type[problem_instance.operators_roulette_selection_type.index(roulette_type)] +=1
        problem_instance.nof_success_removal_type[problem_instance.operators_removal_type.index(removal_type)] +=1
        problem_instance.nof_success_removal_number[problem_instance.operators_removal_number.index(removal_number)] +=1
        problem_instance.nof_success_repair_type[problem_instance.operators_repair_type.index(repair_type)] +=1
    return
def get_weights_for_removal(problem_instance, customers_to_select, pool):
   
    if pool:
        centerX = sum(node.xcoordinate for node in pool)/ len(pool)
        centerY = sum(node.ycoordinate for node in pool)/ len(pool)        
    else:
        centerX = sum(node.xcoordinate for node in problem_instance.customer_nodes)/ len(problem_instance.customer_nodes)
        centerY = sum(node.ycoordinate for node in problem_instance.customer_nodes)/ len(problem_instance.customer_nodes)
        centroid = min(problem_instance.customer_nodes, key=lambda x: math.sqrt((x.xcoordinate - centerX)**2 + (x.ycoordinate - centerY)**2))

    
    weights = [math.sqrt((x.xcoordinate - centerX)**2 + (x.ycoordinate - centerY)**2) for x in customers_to_select]
    weights = [ (x / sum(weights)) for x in weights]
    return weights
def get_weights_for_node_removal(problem_instance, customers_to_select, solution):
    RH.update_intervals(problem_instance, solution)
    weights = []

    indices = [[node.index for node in route] for route in solution] #required since the copies do not match with the original nodes
        
    for node in customers_to_select:
        route_index = indices.index([x for x in indices if node.index in x][0])
        route = solution[route_index]
        index = route.index([node_cur for node_cur in route if node_cur.index == node.index][0])

        #weights.append(problem_instance.latest_times_feasible[node.index] - problem_instance.arrivals[node.index])    
        if index == len(route) - 1:
            weights.append(problem_instance.distance_matrix[node.index][0])
        else:
            weights.append(problem_instance.distance_matrix[node.index][route[index + 1].index])

    return weights
def get_weights_for_nodes_removal(customers_to_select, pool): #ordered by the customers given lists
    weights = []

    for node_p in customers_to_select:
        weights.append(sum((math.sqrt((node_p.xcoordinate - node_s.xcoordinate)**2 + (node_p.ycoordinate - node_s.ycoordinate)**2)  for node_s in pool ))/len(pool))
        
    weights = [ 1/x for x in weights]
    sumx = sum(weights)
    weights = [x/sumx for x in weights]
    
    return weights
# Local search: the two operators (relocate, 2-opt* = tail exchange, the revised one-point
# crossover) live in local_search.py; every move is decided by the two-way check with the
# forward / backward time screen, the cost lower bound, granular neighbour lists and a route cache.
def _routes_of(solution):
    if type(solution) != list: # a two-echelon / multi-trip object
        return [list(route.sequence) for route in solution.routes_sevs]
    return [list(route) for route in solution]

def perform_N_two_point_crossover(problem_instance, routes_in, N, points = 1):
    """2-opt* (tail exchange) descent, at most N passes. :return: (relative change of the two-way cost, routes)"""
    return _run_operators(problem_instance, routes_in, ("two_opt_star",), N)

def perform_N_relocation(problem_instance, current_solution, N):
    """Relocate descent, at most N passes. :return: (relative change of the two-way cost, routes)"""
    return _run_operators(problem_instance, current_solution, ("relocate",), N)

def _run_operators(problem_instance, routes_in, operators, passes):
    routes = _routes_of(routes_in)
    if len(routes) < 2 or passes < 1:
        return 0, routes
    routes, report = LS.local_search(problem_instance, routes, operators = operators, max_rounds = 1,
                                     max_passes = passes, seed = random.random())
    return (report["final_cost"] - report["start_cost"])/report["start_cost"], routes

def Local_search_incumbent(problem_instance, current_solution_in, replications = 5,  iters = 50):
    """
    Relocate + 2-opt* descent (local_search.local_search) on the SEV routes of the incumbent:
    rounds of both operators until neither improves the two-way cost (at most `replications`
    rounds, `iters` passes per operator and round). The improved route set is stored for the
    two-echelon check. :return: (relative change of the two-way cost, routes)
    """
    routes = _routes_of(current_solution_in)
    routes, report = LS.local_search(problem_instance, routes, max_rounds = max(1, replications),
                                     max_passes = max(1, iters), seed = random.random())
    problem_instance.local_search_report = report
    if not report["start_cost"]:
        return 0, routes
    return (report["final_cost"] - report["start_cost"])/report["start_cost"], routes

def local_search_initial(problem_instance, solution, seed = None, max_rounds = 20):
    """
    Improve an initial multi-trip solution by the relocate + 2-opt* descent (local_search.py,
    two-way check). The improved route set is stored for the two-echelon check and checked again
    as a multi-trip solution (transshipments by the DP), with its two-way cost set.
    :return: (improved multi-trip solution, or the given one when nothing improved or the start
             is not feasible, local search report)
    """
    if solution is None or not solution.is_feasible:
        return solution, None
    routes, report = LS.local_search(problem_instance, _routes_of(solution), seed = seed, max_rounds = max_rounds)
    if report["improvement"] <= LS.MIN_GAIN:
        return solution, report
    improved = RH.check_feasibility(problem_instance, routes, VRP_only = False, multi_trip = True, single_route_MVRP = False)
    set_two_way_cost(problem_instance, improved)
    if not improved.is_feasible:
        return solution, report
    return improved, report

def alns_initial(problem_instance, solution, seed = None, iterations = None, time_limit = None, log = None):
    """
    Improve an initial multi-trip solution by the ALNS of alns.py (destroy / repair on the SEV
    routes, two-way check, simulated annealing, relocate + 2-opt* on every new best). The
    ALNS.ALNS_ELITE best route sets are stored for the two-echelon check; the best is checked again
    as a multi-trip solution (transshipments by the DP), with its two-way cost set.
    iterations / time_limit: None -> ALNS.ALNS_ITERATIONS / ALNS.ALNS_TIME_LIMIT.
    :return: (improved multi-trip solution, or the given one when nothing improved or the start
             is not feasible, ALNS report)
    """
    if solution is None or not solution.is_feasible:
        return solution, None
    routes, report = ALNS.alns(problem_instance, _routes_of(solution), seed = seed,
                               iterations = ALNS.ALNS_ITERATIONS if iterations is None else iterations,
                               time_limit = ALNS.ALNS_TIME_LIMIT if time_limit is None else time_limit,
                               log = log)
    if report["improvement"] <= LS.MIN_GAIN:
        return solution, report
    improved = RH.check_feasibility(problem_instance, routes, VRP_only = False, multi_trip = True, single_route_MVRP = False)
    set_two_way_cost(problem_instance, improved)
    if not improved.is_feasible:
        return solution, report
    return improved, report

def select_seed_customers(problem_instance, nstart = 5):
    distances = [[x, min(problem_instance.distance_matrix[problem_instance.depot.index][satellite.index]
                         #transferduration
                         +problem_instance.distance_matrix[x.index][problem_instance.depot.index]
                         +problem_instance.distance_matrix[satellite.index][x.index] for satellite in problem_instance.satellites)] for x in problem_instance.customers]
    distances.sort(key=lambda x: x[1], reverse=False) #sort by waiting times

    nstart = min(len(distances), nstart)  #order for the minimum waiting
    #first half
    seed_customer_list = [x[0] for x in distances[:nstart]] #+ [x[0] for x in distances[len(distances)-half:]]

    return seed_customer_list

#multi_start_initial: repair + store every constructed start (True) or only the best one (False)
REPAIR_ALL_STARTS = False

def multi_start_initial(problem_instance, nstart = 5, report_initials = False, last_route = []):
        
    start_time = time.time()
    
    
    initial_solutions = []       
    
    seed_customers = select_seed_customers(problem_instance, nstart)
    nstart = min(len(seed_customers), nstart) 

    for start in range(nstart):

        seed_customer =  seed_customers[start]  #seed_customers[start]
        new_solution = [[seed_customer]]

        pool = [node for node in problem_instance.customers if node != seed_customer]
        
        #after each repair we check and store the solution
        constructed_solution, pool = RH.k_best_insertion_repair(problem_instance, new_solution, pool)    

        cost = sum(RH.is_route_feasible(problem_instance, route)[2] for route in constructed_solution)
        initial_solutions.append([start, 
                                   len(constructed_solution), cost, constructed_solution])
    
    runtime_all = round(time.time() - start_time, 2)

    initial_solutions.sort(key=lambda x: x[2])   #sort by dist cost 
    initial_solutions.sort(key=lambda x: x[1])  #sort by fleet size
         
         
    if report_initials:
        report_initial_solutions_stats(problem_instance, initial_solutions, runtime_all)
    
    #multi-trip (MVRP) check with repair of the infeasible routes: the best constructed route set
    #only, or every start (REPAIR_ALL_STARTS): each repaired set is stored in
    #problem_instance.heuristics_solutions, so check_found_solutions evaluates all of them at the
    #two-echelon level and the best FE objective wins, not the best street objective
    checked = []
    for start, n_routes, cost, routes in (initial_solutions if REPAIR_ALL_STARTS else initial_solutions[:1]):
        solution_mvrp = RH.check_feasibility(problem_instance, routes, VRP_only = False, multi_trip=True, single_route_MVRP = False)
        checked.append(solution_mvrp)
    problem_instance.constructed_solutions = checked
    feasible = [s for s in checked if s.is_feasible]
    if feasible:
        feasible.sort(key=lambda s: (s.nSEVs, s.objective))
        return feasible[0]
    return checked[0]

def mip_initial_solution(problem_instance, timelimit, warm_start = None, customers = None, output = False):
    """
    Initial solution by the free street routing MIP (MathModels.street_routing_MIP): the model
    routes the customers, places the transshipments and chooses the satellites in one go,
    objective = vehicles + distance + detours + transshipments (optimal for the street problem
    at gap 0; with a time limit the incumbent). warm_start = routes of a heuristic solution;
    None (the free initializer) passes no routes, the MIP starts from scratch.
    The routes are stored in problem_instance.heuristics_solutions (complete solutions only)
    and checked as a multi-trip solution (check_MVRP_solution, transshipments by the DP).
    The cost of the solution is decided by the two-way check (RH.two_way_solution_cost of the
    MVRP routes), not by the MIP objective: result["two_way"] and solution.two_way_cost.
    :return: (multi-trip solution object or None, MIP result dict)
    """
    result = MathModels.street_routing_MIP(problem_instance, timelimit, customers = customers,
                                           warm_start = warm_start, output = output)
    result["warm_started"] = bool(warm_start)
    if not result["feasible"]:
        return None, result
    routes = result["routes"]
    result["two_way_mip_routes"] = RH.two_way_solution_cost(problem_instance, routes)
    RH.add_to_feasible_solution(problem_instance, routes)
    solution = RH.check_feasibility(problem_instance, routes, VRP_only = False, multi_trip = True, single_route_MVRP = False)
    solution.mip_result = result
    set_two_way_cost(problem_instance, solution)
    result["two_way"] = solution.two_way
    return solution, result

def set_two_way_cost(problem_instance, solution):
    """solution.two_way = RH.two_way_solution_cost of its SEV routes, solution.two_way_cost = its cost."""
    if solution is None:
        return None
    if solution.is_feasible:
        solution.two_way = RH.two_way_solution_cost(problem_instance, [route.sequence for route in solution.routes_sevs])
    else:
        solution.two_way = {"feasible": False, "cost": BigNumber, "n_vehicles": 0, "n_trips": 0, "served": 0,
                            "route_costs": [], "route_trips": [], "infeasible_routes": []}
    solution.two_way_cost = solution.two_way["cost"]
    return solution.two_way

def select_initial_by_two_way(problem_instance, candidates):
    """
    Pick the initial solution among candidates [(method name, multi-trip solution or None), ...]
    by the two-way cost: feasible solutions first, then the smallest two-way cost (vehicles +
    distance + detours + trips with the minimum on-time transshipments). Ties keep the earlier
    candidate. :return: (method name, solution) or (None, None)
    """
    best = (None, None)
    best_key = None
    for name, solution in candidates:
        if solution is None:
            continue
        if not hasattr(solution, "two_way"):
            set_two_way_cost(problem_instance, solution)
        key = (not (solution.is_feasible and solution.two_way["feasible"]), solution.two_way_cost)
        if best_key is None or key < best_key:
            best, best_key = (name, solution), key
    return best

def get_initial_solution(problem_instance, nof_initializations = 1, read_initials = False):    

      
    if read_initials: 
        best_initial_routes = RH.read_initial_SEV_routes_benchmark(problem_instance)
    else:
        best_initial_routes = multi_start_initial(problem_instance,nof_initializations, True)  #generate nof_initializations many initial solutions and True write out initials

    relaxed_initial = RH.check_feasibility(problem_instance, best_initial_routes)
    relaxed_cost_initial = relaxed_initial.objective
    RH.check_found_solutions(problem_instance)

    runtime = round(time.time() - problem_instance.modelStartTime,1)
    write_out_all_initials(problem_instance, runtime)

    return relaxed_initial, relaxed_cost_initial

def write_out_all_initials(problem_instance, runtime):
    #write out all initial solutions
    path = "init_solutions_makespan_" + str(problem_instance.name) +"_"+str(len(problem_instance.customers))+ ".txt"

    with open(path, 'w') as file:
        for sol_array in problem_instance.heuristics_solutions:
            for route in sol_array:
                for ind in route:
                    file.write(str(ind) + "\t")
                file.write( "-"+ "\t")
            file.write("\n")
    return
def report_initial_solutions_stats(problem_instance, initial_solutions, runtime):  
    
    average_fleet_size = sum([x[1] for x in initial_solutions])/len(initial_solutions)
    average_cost = sum([x[2] for x in initial_solutions])/len(initial_solutions)
    best_solution = initial_solutions[0]

    with open(problem_instance.initial_report, 'a') as file:
        file.write("problem_num\t"+str(problem_instance.name) +"\tnumber_of_customers\t"+str(len(problem_instance.nodes)-1) 
                +"\tBest_initial_solution_fleet_size\t" + str(best_solution[1])+ "\ttotal_distance\t"+str(best_solution[2])+ "\tsolution_time_secs\t"+str(runtime)
                +"\tnofinitial_solutions\t" + str(len(initial_solutions))
                +"\tAverage_initials_fleet_size\t" + str(average_fleet_size) + "\tAverage_initials_cost\t" + str(average_cost)+ 
                "\n")
    
    return 

