from json import load, dump
import os
import io
import numpy
import math 
import time
from pathlib import Path
from os import walk
import csv
import pandas as pd
# NOTE: FLP_scenarios and Repair_heuristcs are imported lazily inside the
# use-case functions that need them, so that benchmark loading does not pull in
# the heuristics (and their solver dependencies).

bigNumber = 100000000

class Node:
    def __init__(self, t, index, xcor,ycor,demand,earliest,latest,serviceTime):
        self.type = t
        self.index = index
        self.xcoordinate = xcor
        self.ycoordinate = ycor
        self.demand = demand 
        self.start_interval = earliest
        self.end_interval = latest
        self.unloading_time = serviceTime

def ReadSolomonInstance(instance_name, N):
    #based on Solomon test data for VRPTW stored in json files in data folder
    #Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling
    #problems with time window constraints. Operations research, 35(2), 254-265.
    
    projectPath = os.path.dirname(os.path.abspath(__file__))
    json_file = projectPath+"/data/json/"+instance_name

    with io.open(json_file, 'rt', encoding='utf-8', newline='') as file_object:
            instance = load(file_object)    
    
    nodes = []
    for i in range(N):
        ss = instance[f'customer_{i+1}']
        nodes.append(Node("C", (i+1), ss['coordinates']['x'], ss['coordinates']['y'], ss['demand'], ss['ready_time'],ss['due_time'],
                         ss['service_time']))  
    
    ss = instance['depart']
    nodes.insert(0,Node("D", 0, ss['coordinates']['x'], ss['coordinates']['y'], ss['demand'], ss['ready_time'],ss['due_time'],
                         ss['service_time'])) 
    distMatrix = instance['distance_matrix']
    Capacity = instance['vehicle_capacity']
    
    return nodes, distMatrix, Capacity
def read_vrp_data(instance_name, N):
        nodes = []
        capacities = []
        projectPath = os.path.dirname(os.path.abspath(__file__))
        filename = projectPath+"/data/dataset/"+instance_name
        # Read the entire content of the file into a string variable
        with open(filename, "r") as file:
            data = file.read()

        # Split the content into lines and remove leading/trailing whitespace
        lines = [line.strip() for line in data.split("\n")]

        # Initialize dictionaries to store the data
        
        # Iterate through each line and identify sections
        current_section = None
        is_header_row = False  # Flag to track header rows
        for line in lines:
            if line in ["VEHICLE", "POINT", "CUSTOMER","SATELLITE"]:
                current_section = line
                if current_section == "CUSTOMER": #no header after customer data
                    continue
                is_header_row = True  # Next line will be a header
                continue

            if is_header_row:
                is_header_row = False  # Reset the flag after skipping the header
                continue

            # Extract data based on the current section
            if current_section == "VEHICLE" and line:
                number, capacity = line.split()
                capacities.append(int(capacity))

            elif current_section == "POINT" and line and not line.startswith("CDC") and not line.startswith("SATELLITE"):
                if  line.startswith("SLDEPOT"):
                    continue
                label, xcoord, ycoord, demand, ready_time, due_date, service_time = line.split()                
                nodes.append(Node("C", int(label), int(xcoord), int(ycoord), int(demand), int(ready_time),int(due_date),
                            int(service_time)))
                nodes[-1].name = label

            elif current_section == "CUSTOMER" and line:
                customer, xcoord, ycoord, demand, ready_time, due_date, service_time = line.split()                
                nodes.append(Node("C", int(customer), int(xcoord), int(ycoord), int(demand), int(ready_time),int(due_date),
                            int(service_time)))
                nodes[-1].name = customer
        
        nodes = [node for node in nodes if node.index <= N]

        return nodes, capacities[0], capacities[1]
    
def read_best_known_solution(problem):    
    instance_name = problem.name
    path = os.path.dirname(os.path.abspath(__file__))+"/data/sols"
    sol_files = next(walk(path), (None, None, []))[2]

    name_ins = instance_name.split(".")[0]
    sol_file = [filename for filename in sol_files if filename.split("_")[0] == name_ins][0]

    projectPath = os.path.dirname(os.path.abspath(__file__))
    sol_file = projectPath+"/data/sols/"+sol_file

    with open(sol_file, "r") as file:
            data = file.read()

    # Split the content into lines and remove leading/trailing whitespace
    lines = [line.strip() for line in data.split("\n")]
    lines = [line for line in lines if line]

    #retrieve the solution
    name = lines[0].split(" ")[2]
    cost = float(lines[1].split(" ")[1])
    nof_FEVs = int(lines[2].split(" ")[1])
    nof_SEVs = int(lines[3].split(" ")[1])

    FEVs_headers = ["FL"+str(i) for i in range(nof_FEVs)]
    SEVs_headers = ["SL"+str(i) for i in range(nof_SEVs)]
    #empty lines at 4,5,6,
    started_FE = False
    sequences_FE = []
    new_sequence_FE = []
    for line in lines:
        if line.split(" ")[0] in FEVs_headers:
            started_FE = True
            if new_sequence_FE:
                sequences_FE.append(new_sequence_FE)
            new_sequence_FE = []
            continue
        elif line.split(" ")[0] in SEVs_headers:
            sequences_FE.append(new_sequence_FE)
            break
        elif started_FE:
            new_sequence_FE.append(line.split(" ")[1])
    
    #READ SEVs
    started = False
    sequences = []
    new_sequence = []    
    for line in lines:
        if line.split(" ")[0] in SEVs_headers:
            started = True
            if new_sequence:
                sequences.append(new_sequence)                
            new_sequence = []
            continue
        elif started:            
            new_sequence.append(line.split(" ")[1])
                        
    sequences.append(new_sequence) 

    not_customers = ["SAT"+str(i) for i in range(8)]
    not_customers.append("0")
    routes = []
    transfers = []
    for list in sequences:
        new_route = []
        for ii,item in enumerate(list):
            if item not in not_customers:
                new_route.append(int(item))
            else:
                if item != "0":
                    transfers.append([int(list[ii+1]), item ]) #next point transfer point, satellite id by the authors            
            
        routes.append(new_route)
    
    problem.best_known_SL = [[problem.nodes[ind] for ind in list_r ] for list_r in routes]     
    
    for ii, item in enumerate(transfers):
        item[1] = [satellite.index for satellite in problem.satellites if str(satellite.name) == item[1]][0]

    problem.best_known_transfers = transfers

    #calculate the cost on both levels for the given problem   
    sequences.extend(sequences_FE) 
    for list_SE in sequences:
        for ii,item in enumerate(list_SE):
            list_SE[ii] = [node for node in problem.network if str(node.name) == item][0]
    
    travel_street = 0
    travel_water = 0
    for route in sequences:
        distances = [problem.distance_matrix[route[i].index][route[i + 1].index] for i in range(len(route) - 1)]        
            
        if route[0] == problem.depot:
            travel_street+= sum(distances)   
        else:
            travel_water += sum(distances)

    problem.SOA_cost = travel_water + travel_street
    problem.SOA_cost_FE = travel_water
    problem.SAO_cost_SE = travel_street
    
    return 

class SatelliteNetwork:
    def __init__(self, satellite_nodes_indices = [], distmatrix = []):
        # The graph: { node_id: [arc1, arc2, ...] }
        self.graph = {}
        self.satellite_inds = satellite_nodes_indices #indices
        self.distance_matrix = distmatrix

    def get_closest_satellite_metrics(self, start_node, end_node):        
        mindist = 100000000
        closest_satellite_id = None

        for satellite in self.satellite_nodes:
            dist = self.distance_matrix[start_node][satellite] + self.distance_matrix[satellite][end_node]
            if dist < mindist:
                mindist = dist
                closest_satellite_id = satellite
        
        extra_distance = mindist - self.distance_matrix[start_node][end_node]
        
        return extra_distance, closest_satellite_id    

    def check_arc_cost(self, start_node, end_node):
        """
        Adds an arc with satellite metrics.
        """
        if start_node not in self.graph:
            self.graph[start_node] = []            
        else:
            #check if the arc already exists and if so, skip the calculation
            existing_arcs = self.graph[start_node]
            for arc in existing_arcs:
                if arc["to"] == end_node:
                    return arc["extra_distance"]
        # Store metadata in a dictionary for O(1) access per arc
        extra_dist, sat_id = self.get_closest_satellite_metrics(start_node, end_node)
        arc_data = {
            "to": end_node,
            "closest_satellite": sat_id,
            "extra_distance": extra_dist
        }
        
        self.graph[start_node].append(arc_data)
        return extra_dist

    def get_arcs(self, node_id):
        return self.graph.get(node_id, [])
    
class instance:
    def __init__(self, name, nodes):
        
        self.name = name
        self.nodes = nodes
        self.satellites = []
        self.customers = []
        self.network = []
        self.singleTransfer = False   #we assume it is true for all problems solved in the paper but if False, it removes one-to-one transfers at the satellites for alternatives 
        self.report = "report.txt"
        self.stats = "stats.txt"        
        self.bestObjectve = 10000000
        self.bestSolution = None       
        self.hard_tw = True
        self.cost_unit_travel = 1
        self.cost_unit_travel_FEV = 1
        self.cost_vehicle_use = 10000#ensure bigness for the case
        self.cost_vehicle_use_FEV = 10000
        self.trip_unit_cost = 10000 
        self.TransferDuration = 0
        #settings for solving models
        self.timelimit_SEmodel = 50
        self.timelimit_FEmodel = 1000
        self.cutoff = math.inf #does not let any solution be feasible if it is suboptimal, check for intermediate steps for ALNS
        self.vrp = True
        self.mvrp = False
        self.twovrp = False
        self.heuristics_solutions = []
        self.feas_solutions = []
        self.inf_solutions = [] 
        self.checked_solutions = [] 
        self.tabu_list = []   
        self.feasible_list = []
        self.earliest_times_feasible = []
        self.latest_times_feasible  = []
        self.best_found = None 
        self.best_cost = math.inf
        self.SAA_rate = 0.01  
        self.removal_max = 300  
        self.tabu_search_active = True  
        self.consolidated = False
        #vehicle timing (see set_unbounded_vehicle_time)
        self.unbounded_vehicle_time = False
        self.vehicle_start = 0
        self.time_horizon = None
    

    def AddCDC(self):
    #The layout used is 50/150 for CDC
    #based on Grangier et al (2016) CDC definition
    #Grangier, P., Gendreau, M., Lehuédé, F., & Rousseau, L. M. (2016).
    #An adaptive large neighborhood search for the two-echelon multiple-trip
    #vehicle routing problem with satellite synchronization. European journal of operational research, 254(1), 80-91.
        xd = self.depot.xcoordinate
        yd = self.depot.ycoordinate
        distance = math.ceil(math.sqrt((yd-150)*(yd-150) + (xd-50)*(xd-50)))

        self.CDC = Node("CDC", len(self.network), 50, 150, 0, self.depot.start_interval,
                                  self.depot.end_interval + 4*distance, 0)
        self.depot.end_interval = self.depot.end_interval + 4*distance
        self.network.append(self.CDC)
        self.CDC.name = "CDC"
        return
        
    def AddSatellites(self):
        #satellites are located at the city center based on benchmark in the literature by Grangier et al
        #transfer durations or any handling time is assumed zero in Grangier paper/reference paper.
        #in total 8 satellites are located for 100 customers
       
        coordinates = [[25,25], [50,25], [75,25], [25,50], [75,50], [25,75],  [50,75],   [75,75]]

        for i in range(8):
            xCor = coordinates[i][0]
            yCor = coordinates[i][1]
            new_satellite = Node("S", len(self.network), xCor, yCor, 0,self.depot.start_interval,
                                self.depot.end_interval, 0)
            self.satellites.append(new_satellite)  
            self.network.append(new_satellite)
            new_satellite.name = "SAT" + str(i)
            new_satellite.unloading_time = self.TransferDuration
        return
    def get_WL(self):
        fev_profile = 5  # THERE ARE NINE FEV PROFILES [1,2,3,4,5,6,7,8,9] CORRESPONDING TO ['E_R', E', 'D', 'C', 'B', A2', 'A2+', 'A1', 'A1+']

        projectPath = os.path.dirname(os.path.abspath(__file__))
        folder_matrices = projectPath+"/usecase/"
        pickle_off = open(folder_matrices + 'dict_FE_shortest_dist_directed-True_nodes_all.pickle', 'rb')
        dict_FE_shortest_dist_directed = pd.read_pickle(pickle_off)
        df_fe_distance_matrix = dict_FE_shortest_dist_directed[f'vessel_profile_{fev_profile}']
        
        indices = [satellite.index for satellite in self.satellites]
        indices.append(self.CDC.index)
        ind_interest = [satellite.canal_node for satellite in self.satellites]
        ind_interest.append( self.CDC.canal_node)
        dist_matrix = []
        for i in range(len(ind_interest)):
            dist_matrix.append([0]*(len(ind_interest)))

        for indexi,i in enumerate(ind_interest):
            for indexj,j in enumerate(ind_interest):
                distance = df_fe_distance_matrix.at[i, j] 
                if math.isnan(distance):
                    distance  = 1000000
                dist_matrix[indexi][indexj] = distance/self.vessel_driving_speed      
        for i in range(len(dist_matrix)):
            for k in range(len(dist_matrix)):
                for j in range(len(dist_matrix)):
                    if dist_matrix[i][j] > dist_matrix[i][k]+dist_matrix[k][j]:
                        dist_matrix[i][j] = dist_matrix[i][k]+dist_matrix[k][j]
        return dist_matrix, indices 
    def get_SL(self):
        projectPath = os.path.dirname(os.path.abspath(__file__))
        folder_matrices = projectPath+"/usecase/"
        pickle_off = open(folder_matrices + 'df_SE_shortest_dist_directed-True_nodes_all.pickle', 'rb')
        df_fe_distance_matrix = pd.read_pickle(pickle_off)       
        
        indices = [customer.index for customer in self.customers]
        indices.insert(0, self.depot.index)
        indices.extend([satellite.index for satellite in self.satellites])

        ind_interest = [customer.road_node for customer in self.customers]
        ind_interest.insert(0, self.depot.road_node)
        ind_interest.extend([satellite.road_node for satellite in self.satellites])

        dist_matrix = []
        for i in range(len(ind_interest)):
            dist_matrix.append([0]*(len(ind_interest)))

        for indexi,i in enumerate(ind_interest):
            for indexj,j in enumerate(ind_interest):
                distance = df_fe_distance_matrix.at[i, j] 
                if math.isnan(distance):
                    distance  = 1000000
                dist_matrix[indexi][indexj] = distance/self.driving_speed
                    
        for i in range(len(dist_matrix)):
            for k in range(len(dist_matrix)):
                for j in range(len(dist_matrix)):
                    if dist_matrix[i][j] > dist_matrix[i][k]+dist_matrix[k][j]:
                        dist_matrix[i][j] = dist_matrix[i][k]+dist_matrix[k][j]
        return dist_matrix, indices 
    def get_matrix(self):        
        dist_matrix_SL, indices_SL   = self.get_SL()
        dist_matrix_WL, indices_WL  = self.get_WL()

        return dist_matrix_SL, indices_SL,dist_matrix_WL, indices_WL   
    def CreateDistMatrix_use_case(self, triangluality_check = True):  
        dist_matrix_SL, indices_SL,dist_matrix_WL, indices_WL  = self.get_matrix()   
        
        #connect nodes on the same networks
        #  here you start       
        self.DistMatrix = numpy.zeros( (len(self.network), len(self.network)) ) 
        self.travel_matrix = [[bigNumber for l in self.network] for l in self.network]
        
        for i in range(len(self.network)):
            for j in range(len(self.network)):                
                ii = self.network[i]
                jj = self.network[j]
                #types = C, D, S, CDC 
                
                if ii.type == "D" or ii.type == "C":
                    if jj.type == "CDC":
                        dist = bigNumber #there is no direct link between customers and garage nodes                        
                    else:
                        index_jj = indices_SL.index(jj.index)
                        index_ii = indices_SL.index(ii.index)

                        dist = dist_matrix_SL[index_ii][index_jj]      

                elif ii.type == "S" or ii.type == "CDC":
                    if jj.type == "S" or jj.type == "CDC":
                        index_jj = indices_WL.index(jj.index)
                        index_ii = indices_WL.index(ii.index)

                        dist = dist_matrix_WL[index_ii][index_jj]
                        
                       
                    if jj.type == "D" or jj.type == "C":
                        if ii.type == "CDC":
                            dist = bigNumber #there is no direct link between customers and garage nodes
                        else:
                            index_jj = indices_SL.index(jj.index)
                            index_ii = indices_SL.index(ii.index)

                            dist = dist_matrix_SL[index_ii][index_jj]
                            
                if dist == "nan" or dist == bigNumber:
                    self.DistMatrix[i][j] = 10*self.depot.end_interval
                    self.travel_matrix[i][j] =  10*self.depot.end_interval 
                else:
                    dist = float(dist)
                    self.DistMatrix[i][j] =  dist
                    self.travel_matrix[i][j] =  dist  #in seconds 
        
        return 
   
    
    def CreateDistMatrix_coords(self, unit_coord = 1, triangluality_check = True): #use Euclidean distance to calculate distances
        ii = self.CDC
        jj = self.depot
        cdc_to_depot = round( math.sqrt((ii.xcoordinate - jj.xcoordinate)*(ii.xcoordinate - jj.xcoordinate) 
                                       + (ii.ycoordinate - jj.ycoordinate)*(ii.ycoordinate - jj.ycoordinate)), 1)
        
        self.DistMatrix = numpy.zeros( (len(self.network), len(self.network)) ) 
        for i in range(len(self.network)):
            for j in range(len(self.network)):
                ii = self.network[i]
                jj = self.network[j]
                cost =round( math.sqrt((ii.xcoordinate - jj.xcoordinate)*(ii.xcoordinate - jj.xcoordinate) 
                                       + (ii.ycoordinate - jj.ycoordinate)*(ii.ycoordinate - jj.ycoordinate)), 1)
                self.DistMatrix[i][j] = unit_coord*cost
        if not triangluality_check: #to reduce time in initial experiments
            return
        for i in range(len(self.network)): #triangle inequality 
            for j in range(len(self.network)):
                for k in range(len(self.network)):
                    cc = self.DistMatrix[i][k] + self.DistMatrix[k][j] 
                    if cc<=self.DistMatrix[i][j]:
                        self.DistMatrix[i][j] = cc
        
    def check_travel_matrix(self):
        self.travel_matrix = [[math.inf for l in self.network] for l in self.network]
        for i in range(len(self.network)):
            for j in range(len(self.network)):
                dist = self.DistMatrix[i][j]
                self.travel_matrix[i][j] = round(dist/self.driving_speed,1)
                self.travel_matrix[j][i] = round(dist/self.driving_speed,1)

        if not self.travel_matrix:
            print("Error in street problem MIP: travel Matrix not created")
            exit()
    def UpdateTWs(self):
    #moving time windows to accomodate extra time needed to supply vehicles from CDC and vessels to return 
        addition = math.ceil(self.DistMatrix[self.CDC.index][self.depot.index])  + self.TransferDuration    #the best is assumed that vessels bring goods to the depot and transfer them  
        for i in range(len(self.network)):
            current = self.network[i]
            if current.type == "C" :           
                current.end_interval = current.end_interval +  addition
                current.start_interval = current.start_interval + addition
        #update depot, CDC and satellites for feasibly serving given demand        
        #max_return = float(max(customer_i.end_interval +customer_i.unloading_time +self.DistMatrix[customer_i.index][satellite_p.index] 
        #                        +self.TransferDuration
        #                        + self.DistMatrix[satellite_p.index][self.CDC.index] for customer_i in self.customers for satellite_p in self.satellites ))
        max_return = self.depot.end_interval + 2*addition
        for i in range(len(self.network)):
            current = self.network[i]
            if current.type != "C" :                
                current.end_interval = max_return
                    
        return
        
    def set_transfer_duration(self, duration):
        """Transfer (transshipment) duration U at the satellites, also their service time."""
        self.TransferDuration = duration
        for satellite in self.satellites:
            satellite.unloading_time = duration
        return

    def set_unbounded_vehicle_time(self, horizon_factor = 2.0, vehicle_start = None):
        """Vehicle timing is left free; only the customer time windows bound the operations.

        The big-M values in the models are derived from the customer windows, so the
        vehicle horizon only has to be "long enough":
        * horizon end  T = horizon_factor * latest customer end   (garage, satellites, CDC)
        * garage start = vehicle_start, by default the earliest departure that still lets a
          vehicle reach any customer through any satellite before that customer opens
          (-max_{p,j}(t_0p + t_pj)); the first stop of a street vehicle is a satellite.
        * satellites and CDC open at -infinity (vessel times are free; synchronisation with
          the street routes, i.e. the customer windows, bounds them).
        """
        customers = [node for node in self.network if node.type == "C"]
        latest = max(node.end_interval for node in customers)
        horizon = horizon_factor*latest
        if vehicle_start is None:
            first_leg = max(self.travel_matrix[self.depot.index][p.index] + self.travel_matrix[p.index][j.index]
                            for p in self.satellites for j in customers)
            vehicle_start = -math.ceil(first_leg)

        self.depot.start_interval = vehicle_start
        self.depot.end_interval = horizon
        for satellite in self.satellites:
            satellite.start_interval = -math.inf
            satellite.end_interval = horizon
        self.CDC.start_interval = -math.inf
        self.CDC.end_interval = horizon

        self.unbounded_vehicle_time = True
        self.vehicle_start = vehicle_start
        self.time_horizon = horizon
        return

    def GenerateProblem(self, use_case = False):      
        
        #as in benchmark 50/150/3/3  
        self.AddCDC()
        #as in benchmark 3/3 and satellites have transfer duration
        self.AddSatellites()
        if use_case:
            self.CreateDistMatrix_use_case()
        else:
            self.CreateDistMatrix_coords()
        self.check_travel_matrix()
        
        self.distance_matrix = self.DistMatrix          
        self.closest_satellite_network = SatelliteNetwork()
        self.closest_satellite_network.satellite_nodes = [satellite.index for satellite in self.satellites]
        self.closest_satellite_network.distance_matrix = self.DistMatrix  

        return
def read_initial_case(problem):
    import Repair_heuristcs as RH
    best_initial_routes = RH.read_initial_SEV_routes_use_case(problem, "initial_300.txt")
    min_SEVS = RH.get_makespan(problem, best_initial_routes[0])
    best_initial_routes = RH.divide_routes(problem, best_initial_routes[0], min_SEVS)
    RH.add_to_feasible_solution(problem, best_initial_routes)

    initial_solution = RH.check_feasibility(problem, best_initial_routes)
    RH.check_found_solutions(problem)

    if problem.best_found is not None:
        initial_solution = problem.best_found

    return initial_solution
def read_satellites(file_content):
# Usage
    #type0,index1,name2,xcor3,ycor4,road_node5,canal_node6  0-S,1-P010,2-TF_10,3-4.89616373994886,4-52.3622724117054,5-102271,6-222
    unique_links_ids = []
    satellites = []

    with open(file_content, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for ii, row in enumerate(reader):
            if ii==0:
                continue
            row = [x for x in row if x]
            float_params = [3,4]
            int_params = [5,6]
            for i in range(len(row)):
                if i in float_params:
                    row[i] = float(row[i])
                if i in int_params:
                    row[i] = int(row[i])
            unique_links_ids.append(row[1])        
            satellite_n = Node("S", row[1], row[3], row[4], 0, 0, 28800, 0) 
            satellite_n.name = row[2]
            satellite_n.canal_node = row[6]
            satellite_n.road_node = row[5]
            satellites.append(satellite_n)

    return satellites
def read_horeca_data(file_content):
#type,index,xcoordinate,ycoordinate,demand,start_interval,end_interval,unloading_time,road_node,name,addres
    un_reach = [2632, 3564, 3566, 6756, 12553, 8812, 15971, 16643, 16911, 17462, 14214, 17767, 14693, 14921, 14723]
    
    unique_links_ids = []
    satellites = []

    with open(file_content, 'r' , encoding='ISO-8859-1') as file:
        reader = csv.reader(file, delimiter=',')
        for ii, row in enumerate(reader):
            if ii==0:
                continue
            row = [x for x in row if x]
            if int(row[1]) in un_reach:
                continue
            float_params = [2,3, 5,6,7]
            int_params = [1,4,8]
            for i in range(len(row)):
                if i in float_params:
                    row[i] = float(row[i])
                if i in int_params:
                    row[i] = int(row[i])
            unique_links_ids.append(row[1])        
            satellite_n =  Node("C", row[1], row[2], row[3], row[4], row[5], row[6], row[7]) 
            satellite_n.addres = row[10]
            satellite_n.name = row[9]
            satellite_n.road_node = row[8]
            satellites.append(satellite_n)

    return satellites

def readGenerate_benchmark(instance_name, size, unbounded_vehicle_time = True):
    """Costs: the constants of instance() (1 per travel unit, 10000 per vehicle and per
    transshipment); transfer duration U = 0 as in the benchmark."""
    nodes, capacity_FEV, capacity_SEV = read_vrp_data(instance_name, size)
    problem = instance(instance_name, nodes)
    problem.depot = problem.nodes[0]
    problem.depot.type = "D"
    problem.depot.name = "0"
    problem.customers = [x for x in problem.nodes if x.type == "C"]    
    
    problem.capacity_SEV = capacity_SEV    #class 1 ratio  4/0.5 and class 2 ratio 2/0.25
    problem.capacity_FEV = capacity_FEV
    problem.driving_speed = 1
    problem.TransferDuration = 0 #as in benchmark, fit something for the use case
    problem.network.extend(problem.nodes)

    problem.GenerateProblem(use_case = False)
    
    problem.depot.end_interval = 1.5*problem.depot.end_interval
    problem.CDC.end_interval = 1.5*problem.CDC.end_interval
    if unbounded_vehicle_time: #customer time windows define the timing, vehicles are free in time
        problem.set_unbounded_vehicle_time()
    
    problem.distance_matrix = problem.DistMatrix


    #read_best_known_solution(problem)
    total_demand = sum(x.demand for x in problem.nodes if x in problem.customers)
    problem.minVessel_bound = math.ceil(sum(x.demand for x in problem.nodes if x in problem.customers)/problem.capacity_FEV)   
    print(total_demand)
    
    problem.case_study = False
    problem.hard_tw = True
    return problem 

def read_nodes(nof_horeca, nof_satellites, scenario, demand_set=1):
    import FLP_scenarios as FLP
    projectPath = os.path.dirname(os.path.abspath(__file__))
    files_use_case = projectPath+"/usecase/"

    file_satellites = files_use_case + "satellite_nodes_list.txt"
    file_horecas = files_use_case+ "daily_horeca_list"+str(demand_set)+".txt"

    horecas = read_horeca_data(file_horecas)
    nof_horeca = min(nof_horeca, len(horecas))
    horecas = [horecas[i] for i in range(nof_horeca)]

    ids_chosen,selected_satellites, satellites_xxx, xxx_horeca, road_nodes = FLP.facility_location_weighted(demand_set, scenario, nof_satellites) 
    
    #satellites = read_satellites(file_satellites)
    
    #satellites = [satellite for satellite in satellites if satellite.index in ids_chosen]

    return horecas, selected_satellites

def locate_garage(problem_instance):

    xcor = sum(node.xcoordinate for node in problem_instance.customers)/ len(problem_instance.customers)
    ycor = sum(node.ycoordinate for node in problem_instance.customers)/ len(problem_instance.customers)
    node_closest = [[node, abs(node.xcoordinate - xcor) +abs ( node.ycoordinate - ycor)] for node in problem_instance.satellites] 
    node_closest.sort(key=lambda x: x[1])
    node_closest = node_closest[0][0]
    garage = Node("D", 0, xcor, ycor, 0, 0, 28800, 0)
    garage.road_node = node_closest.road_node

    return garage 

def locate_CDC(problem_instance):

    xcor = min(node.xcoordinate for node in problem_instance.network)
    ycor = min(node.ycoordinate for node in problem_instance.network)
    CDC = Node("CDC", len(problem_instance.network), xcor, ycor, 0, 0, 28800, 0)
    CDC.canal_node = 127 #dc 2
    return CDC 

def add_TW_span(problem_instance, hours = 8):
    minutess = hours*60*60
    for i in range(len(problem_instance.network)):
        current = problem_instance.network[i]
        if current.type == "C" or current.type == "S":                
            current.end_interval = minutess
        else:
            current.end_interval = minutess + 60*60

    return
def read_generate_case(demand_set = 1, nof_horeca = 1600, nof_satellites = 10, scenario = "equal", unbounded_vehicle_time = True):
    #everything is in seconds
    horecas, satellites = read_nodes(nof_horeca, nof_satellites, scenario,(demand_set+1) ) #demand starts at 1...to... 10
    
    #network
    problem = instance(demand_set, horecas)
    problem.customers = horecas
    problem.satellites = satellites
    
    problem.depot = locate_garage(problem)
    problem.nodes.insert(0,problem.depot)  
    problem.depot.name = "D"

    #check problem nodes matrix !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    problem.network.extend(problem.nodes)
    problem.network.extend(problem.satellites)

    problem.CDC =   locate_CDC(problem)
    problem.network.append(problem.CDC)
    problem.CDC.name = "CDC"

    #operational
    problem.benchmark_vehicle_capacity = 20#
    problem.capacity_SEV = 5# (0.25)*problem.benchmark_vehicle_capacity  
    problem.capacity_FEV = 25 #(2)*problem.benchmark_vehicle_capacity   
    problem.driving_speed = 5 #m\s
    problem.vessel_driving_speed = 1.6 #m\s
    problem.TransferDuration = 3*60 #as in benchmark, fit something for the use case
    
    add_TW_span(problem,8)  
    for ii,node in enumerate(problem.network):
        node.index = ii
        if node.type == "C":
            node.unloading_time = 1.5*60# service_at_customer
        elif node.type == "S":
            node.unloading_time = 3*60
        else: #garage or CDC
            node.unloading_time = 0
    #indexing for distance matrix 
    for node in problem.network:
        node.ID = node.index
    problem.CreateDistMatrix_use_case() #later check triangle inequality for dist matrix on the use case 
    
    problem.distance_matrix = problem.DistMatrix
    #problem.check_travel_matrix()   
           

    problem.minVessel_bound = math.ceil(sum(x.demand for x in problem.nodes if x in problem.customers)/problem.capacity_FEV)   
    
    problem.case_study = True
    problem.hard_tw = False
    problem.name = str(problem.name)
    problem.customers = [node for node in problem.customers if node.type == "C"]
    if unbounded_vehicle_time: #customer time windows define the timing, vehicles are free in time
        problem.set_unbounded_vehicle_time()
    
    #read_initial_case()
    return problem 

def check_feasibility_all_nodes(problem):
    import Repair_heuristcs as RH
    removed = []
    
    for ii,node in enumerate(problem.network):
        if node.type == "C":
            route = [node]
            result = RH.is_route_feasible(problem, route)
            if not result[0]:                
                removed.append(node) 
    
    if len(removed)>1:
        print("infeasible customer service nodes number of removed: ", len(removed))

        print("solver execution ends")
        SystemExit()

    return removed


    

