import random 
import numpy
import SEVs 
import math 
import time
import copy
import MathModels 
import Repair_heuristcs as RH
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
def two_point_crossover(list1, list2, problem_instance = None):
    if len(list1) == 1 or len(list2) == 1:
        return list1, list2
    """
    Perform a two-point crossover operation on two lists.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        tuple: A tuple containing the two offspring lists and a boolean indicating whether the lists can be combined.
    """
    # Get the length of the lists
    length1 = len(list1)
    length2 = len(list2)
    
    if length1 > length2:
        list1, list2 = list2, list1
        length1, length2 = length2, length1
    
    # Choose two random crossover points
    while True:
        try:
            point1 = random.randint(0, length1 - 1)
            point2 = random.randint(point1 + 1, length1 - 1)
        except ValueError:
            continue
        break

    # Perform the crossover
    offspring1 = list1[:point1] + list2[point1:point2] + list1[point2:]
    offspring2 = list2[:point1] + list1[point1:point2] + list2[point2:]    

    return offspring1, offspring2
def one_point_crossover(list1, list2, problem_instance = None):
    """
    Perform a two-point crossover operation on two lists.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        tuple: A tuple containing the two offspring lists and a boolean indicating whether the lists can be combined.
    """
    # Get the length of the lists
    length1 = len(list1)
    length2 = len(list2)
    if len(list1) == 1 or len(list2) == 1:
        return list1, list2
    
    if length1 > length2:
        list1, list2 = list2, list1
        length1, length2 = length2, length1
    # Choose two random crossover points
    while True:
        try:
            point1 = random.randint(0, length1 - 1) 
            chosen_index =  list1[point1-1].index 
            ee = problem_instance.earliest_times_feasible 
            ll = problem_instance.latest_times_feasible 
            x = ([node for node in list2 
                  if ee[chosen_index] +list1[point1-1].unloading_time+problem_instance.distance_matrix[chosen_index][node.index] < ll[node.index]])
            if x:
                point2 = list2.index(x[0])
            else:
                continue
            #point2 = random.randint(point1 + 1, length1 - 1)
            
        except ValueError:
            continue
        break

    # Perform the crossover
    offspring1 = list1[:point1] +  list2[point2:]
    offspring2 = list2[:point2] + list1[point1:] 

    return offspring1, offspring2

def perform_N_two_point_crossover(problem_instance,routes_in, N, points = 1):
    routes = copy.deepcopy(routes_in)
    if len(routes) <2:
        return 0, routes
    
    initial_route_cost = sum([RH.is_route_feasible(problem_instance, route)[2] for route in routes])

    for _ in range(N):
        RH.update_intervals(problem_instance, routes)
    
        #get two random routes
        routes_to_choose = [route for route in routes if len(route) > 1]
        routes_chosen = random.sample(routes_to_choose, 2)
        route1 = routes_chosen[0]
        route2 = routes_chosen[1]
        if route1 == route2 or min(len(route1), len(route2)) == 1:
            N = N +1
            continue
        
        list1 = [copy.deepcopy(node) for node in route1]
        list2 = [copy.deepcopy(node) for node in route2]
        if (len(list1) != 1 and len(list2) != 1) and points == 1:
            offspring1, offspring2 = one_point_crossover(list1, list2, problem_instance)
        else:
            offspring1, offspring2 = two_point_crossover(list1, list2,problem_instance)  

        improving_move = False     
        total_cost = 0         
        
        result2 = RH.is_route_feasible(problem_instance, offspring2)
        if not result2[0]:
            continue
        total_cost += result2[2]

        result1 = RH.is_route_feasible(problem_instance, offspring1)
        if not result1[0]:
            continue
        
        total_cost += result1[2]
        initial_total_cost = RH.is_route_feasible(problem_instance, route1)[2] + RH.is_route_feasible(problem_instance, route2)[2]
        
        if total_cost < initial_total_cost  and abs(initial_total_cost - total_cost) > 0.5:
            improving_move = True

        if improving_move:
            RH.add_to_feasible_solution(problem_instance, routes)
            routes[routes.index(route1)] = offspring1
            routes[routes.index(route2)] = offspring2 
            sum2 = sum([len(x) for x in routes]) 
            if sum2 != len(problem_instance.customers):
                print("Error in crossover move: not all nodes visited")
    final_routes_cost = sum([RH.is_route_feasible(problem_instance, route)[2] for route in routes])
    improvement_percentage = (final_routes_cost - initial_route_cost)/initial_route_cost
        
    return improvement_percentage, routes

def perform_N_relocation(problem_instance,current_solution, N):
    routes = copy.deepcopy(current_solution)

    if len(routes) <2:
        return 0, routes

    initial_route_cost = sum([RH.is_route_feasible(problem_instance, route)[2] for route in routes])
    
    for _ in range(N):
        initial_cost = sum([RH.is_route_feasible(problem_instance, route)[2] for route in routes])

        routes_chosen = random.sample(routes, 2)
        node_chosen = random.choice(routes_chosen[0])
        
        routes.remove(routes_chosen[0])
        routes.remove(routes_chosen[1])

        routes_chosen[0].remove(node_chosen)
        pool = [node_chosen]
        routes_chosen = [x for x in routes_chosen if len(x) > 0]

        routes_modified, pool = RH.k_best_insertion_repair(problem_instance, routes_chosen, pool, 1, True)
        
        routes.extend(routes_modified)
        final_cost = sum([RH.is_route_feasible(problem_instance, route)[2] for route in routes])
        if final_cost < initial_cost and abs(initial_cost - final_cost) > 0.5:
            RH.add_to_feasible_solution(problem_instance, routes)
        if len(pool)>0:
            print("partial solution found in repairing")
        sum_served = sum(len(route) for route in routes)
        if sum_served != len(problem_instance.customers):
            print("customer node loss/addition")


    final_routes_cost = sum([RH.is_route_feasible(problem_instance, route)[2] for route in routes])
    improvement_percentage = (final_routes_cost - initial_route_cost )/initial_route_cost
        
    return improvement_percentage, routes

def Local_search_incumbent(problem_instance, current_solution_in, replications = 5,  iters = 50):

    #mvrp_status = problem_instance.mvrp
    #problem_instance.mvrp = True
    if type(current_solution_in) != list: # a two-echelon object
        routes1 = [copy.deepcopy(route.sequence) for route in current_solution_in.routes_sevs]

    else:
        routes1 = copy.deepcopy(current_solution_in) #do not change given solution just find better to store if possible
    
    initial_cost = sum([RH.is_route_feasible(problem_instance, route)[2] for route in routes1])
    improvement_percentage1 = 5
    for _ in range(replications):
        improvement_percentage2, routes2 = perform_N_relocation(problem_instance,routes1, 0)
        if abs(improvement_percentage1 + improvement_percentage2) < 0.001:
            break
        improvement_percentage1, routes1 = perform_N_two_point_crossover(problem_instance,routes2, iters)
        if abs(improvement_percentage1 + improvement_percentage2) < 0.001:
            break
    
    
    
    final_cost = sum([RH.is_route_feasible(problem_instance, route)[2] for route in routes1])
    percentage_imp = (final_cost - initial_cost)/initial_cost

    #problem_instance.mvrp = mvrp_status
    return percentage_imp, routes1

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
    
    
    solution_best_constructed =  RH.check_feasibility(problem_instance, initial_solutions[0][3],VRP_only = False, multi_trip=True, single_route_MVRP = False)

    return solution_best_constructed #, initial_solutions

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

