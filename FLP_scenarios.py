import os
import io
import numpy
import math 
import time
from twoe.solver import MILP, quicksum
import Satellite_scenarios_generation
import matplotlib.pyplot as plt

def get_distance_matrix(nodes):    
    #onstreet operations, check SE complete matrix for satellites and horecas
    for ii, node in enumerate(nodes):
        node.dist_ind = ii
    distance_matrix = numpy.zeros((len(nodes), len(nodes)))
    
    #map names to connect -> index naming is the same
    #use given SE network 
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i <= j: #symmetric
                distance_matrix[i,j] = math.sqrt((nodes[i].xcoordinate - nodes[j].xcoordinate)**2 + (nodes[i].ycoordinate - nodes[j].ycoordinate)**2)
            else:
                distance_matrix[i,j] = distance_matrix[j,i]

    return distance_matrix

def get_cost_per_satellite(road_nodes, satellites, nominal_category= False, equal = False, closest_road_segments = 5):
    if equal:
        for satellite in satellites:
            satellite.unit_cost = 1
        return [satellite.unit_cost for satellite in satellites]
    average_traffic = sum(node.average_traffic for node in road_nodes) / len(road_nodes)
    min_traffic = min(node.average_traffic for node in road_nodes) 
    max_traffic = max(node.average_traffic for node in road_nodes)

    print(average_traffic, min_traffic, max_traffic)

    all_nodes = satellites + road_nodes
    distance_matrix = get_distance_matrix(all_nodes)
    
    for satellite in satellites:
        satellite.unit_cost = 0
        if not nominal_category:
            min_distances = [[road_node.average_traffic,  distance_matrix[satellite.dist_ind, road_node.dist_ind]] for road_node in road_nodes]
            min_distances.sort(key = lambda x: x[1])
            min_distances = min_distances[:closest_road_segments]
            satellite.unit_cost = sum(x[0] for x in min_distances) /  len(min_distances) 
        else:
            min_distances = [[road_node.scenario,  distance_matrix[satellite.dist_ind, road_node.dist_ind]] for road_node in road_nodes]
            min_distances.sort(key = lambda x: x[1])
            min_distances = min_distances[:closest_road_segments]
            degrees = [x[0] for x in min_distances]

            satellite.unit_cost += sum(1 for x in degrees if x == "L")
            satellite.unit_cost += sum(2 for x in degrees if x == "M")
            satellite.unit_cost += sum(4 for x in degrees if x == "H")
            satellite.unit_cost = satellite.unit_cost / len(degrees)
    
    
    min_val = min([satellite.unit_cost for satellite in satellites])
    max_val = max([satellite.unit_cost for satellite in satellites])

    for satellite in satellites:
        satellite.unit_cost =  (satellite.unit_cost  - min_val) / (max_val - min_val) + 1

    unit_costs = [satellite.unit_cost for satellite in satellites]
    
    return unit_costs
def plot_scenarios_cost(road_nodes, satellites):
    equal_case = get_cost_per_satellite(road_nodes, satellites, nominal_category = False, equal = True)
    numeric_case = get_cost_per_satellite(road_nodes, satellites)
    nominal_case = get_cost_per_satellite(road_nodes, satellites, nominal_category = True, equal = False)

    satellite_names = [ii for ii,satellite in enumerate(satellites)]
    plt.scatter(satellite_names, equal_case, c = "green", marker='X', label='equal')
    plt.scatter(satellite_names, numeric_case, c = "blue", marker='x', label='numeric')
    plt.scatter(satellite_names, nominal_case, c = "purple", marker='.', label='nominal')
    plt.legend(loc="upper right")
    plt.xlabel('satellite_names')
    plt.ylabel('unit_cost')
    plt.title("Satellite unit costs per scenario")
    plt.savefig("satellite_scenarios_all.png")
    plt.close()
    return
def plot_satellite_locations_optimized(scenario, N):
    ids_chosen,selected_satellites, satellites, horecas, roads = facility_location_weighted(scenario, N)
    Satellite_scenarios_generation.generate_scenarios(roads, selected_satellites, horecas, plot_satellites = True, classify_network_wise = True, within_demand = True)
    return
def optimize_FLP(scenario, satellites, horecas, N, printing = False, dist_in = None):
    optmal_locations = []

    all_nodes = satellites + horecas
    #if dist_in is  None:
    #    return satellites
    distance_matrix = get_distance_matrix(all_nodes)

    #build model
    model = MILP("facility_location")

    #decision variables
    x = {}
    y = {}

    for i in range(len(horecas)):
        for j in range(len(satellites)):
            x[i,j] = model.add_var(vtype = "B", name = "x"+str(i)+"."+str(j)) #1 if horeca i is assigned to satellite j, 0 otherwise

    for i in range(len(satellites)):
        y[i] = model.add_var(vtype = "B", name = "y"+str(i)) #1 if satellite i is open, 0 otherwise
    
    #assign each horeca to only a satellite
    for i in range(len(horecas)):
        model.add_constr(quicksum(x[i,j] for j in range(len(satellites))) == 1)
        #assign a horeca i to a satellite only if satellite j is open
        for j in range(len(satellites)):
            model.add_constr(x[i,j]  <= y[j])

    #open at most N satellites
    model.add_constr(quicksum(y[i] for i in range(len(satellites))) <= N)

    #set objective
    obj_FLP = quicksum(x[i,j]*satellite.unit_cost * distance_matrix[horeca.dist_ind, satellite.dist_ind] for i, horeca in enumerate(horecas) for j,satellite in enumerate(satellites)) 
    
    model.set_objective(obj_FLP)
    model.verbose = bool(printing)

    model.optimize()

    if model.has_solution:
        for i in range(len(horecas)):
            for j in range(len(satellites)):
                if model.value(x[i,j]) > 0.5:
                    horecas[i].satellite = satellites[j]

        for i in range(len(satellites)):
            if model.value(y[i]) > 0.5:
                satellites[i].is_open = True
                optmal_locations.append(satellites[i])
    else:
        print("No solution", model.status)
        optmal_locations = None
    return optmal_locations

def facility_location_weighted(demand_set, scenario = "numeric", N = 10):
    road_nodes, satellites, horecas = Satellite_scenarios_generation.get_network_and_scenarios(demand_set, scenario)
    if scenario == "equal":
        unit_costs = get_cost_per_satellite(road_nodes, satellites, nominal_category = False, equal = True)
    elif scenario == "numeric":
        unit_costs = get_cost_per_satellite(road_nodes, satellites)
    elif scenario == "nominal":
        unit_costs = get_cost_per_satellite(road_nodes, satellites, nominal_category = True, equal = False)

    selected_satellites = optimize_FLP(scenario, satellites, horecas, N, False)
    ids_chosen = [satellite.index for satellite in selected_satellites]

    return ids_chosen,selected_satellites, satellites, horecas, road_nodes



