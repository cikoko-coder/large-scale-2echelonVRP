import os
import io
import numpy
import math 
import time
from pathlib import Path
from os import walk
import csv
from json import load, dump
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import Repair_heuristcs as RH

# BIG_INF_COST is used for penalties or impossibly large travel times/distances.
BIG_INF_COST = 100000000 

class Node:
    """
    Represents a location in the network (Depot, Customer, Satellite, or CDC).
    """
    def __init__(self, 
                 node_type: str, 
                 index: int | str, 
                 x_coord: float, 
                 y_coord: float, 
                 demand: int, 
                 earliest: float, 
                 latest: float, 
                 service_time: float):
        
        # Core Properties
        self.type = node_type              # "D", "C", "S", or "CDC"
        self.index = index                 # Unique identifier (integer or string label)
        self.xcoordinate = x_coord
        self.ycoordinate = y_coord
        self.demand = demand 
        
        # Time Window Properties (seconds)
        self.start_interval = earliest     # Earliest service time
        self.end_interval = latest         # Latest service time (due time)
        self.unloading_time = service_time # Time spent at the node (service/transfer)
        
        # Extended Properties (Initialized to None/default, set by data readers)
        self.name: str = str(index)        
        self.road_node: Optional[int] = None # Mapping index for the road network (SL)
        self.canal_node: Optional[int] = None # Mapping index for the canal network (WL)
        self.addres: Optional[str] = None     # Address details
        self.ID: int | str = index            # Redundant ID, kept for compatibility
        
    def __repr__(self):
        return f"Node(type='{self.type}', index={self.index}, name='{self.name}', demand={self.demand})"
    

def readGenerate_benchmark(instance_name, size):
    """
    Reads VRP data, initializes a problem instance for benchmarking,
    and sets essential parameters.

    Args:
        instance_name (str): The name of the VRP instance.
        size (int): The size or specific configuration of the instance.

    Returns:
        problem: The configured problem instance object.
    """
    nodes, capacity_FEV, capacity_SEV = read_vrp_data(instance_name, size)
    problem = instance(instance_name, nodes)

    # Configure the depot
    problem.depot = problem.nodes[0]
    problem.depot.type = "D"
    problem.depot.name = "0"

    # Identify customers
    problem.customers = [x for x in problem.nodes if x.type == "C"]

    # Set vehicle capacities
    problem.capacity_SEV = capacity_SEV
    problem.capacity_FEV = capacity_FEV

    # Set constant parameters
    problem.driving_speed = 1
    problem.TransferDuration = 0

    # Generate the problem structure
    problem.network.extend(problem.nodes)
    problem.GenerateProblem(use_case=False)

    # Adjust time windows for Depot (D) and Central Distribution Center (CDC)
    problem.depot.end_interval = 1.5 * problem.depot.end_interval
    problem.CDC.end_interval = 1.5 * problem.CDC.end_interval

    # Update distance matrix
    problem.distance_matrix = problem.DistMatrix

    # Calculate minimum vessel bound (lower bound on the number of FEVs)
    total_demand = sum(x.demand for x in problem.customers)
    problem.minVessel_bound = math.ceil(total_demand / problem.capacity_FEV)

    # Set case study flag
    problem.case_study = False

    return problem


class instance:
    """
    Represents a Vehicle Routing Problem (VRP) instance, holding network data,
    parameters, and solution tracking variables.
    """
    def __init__(self, name, nodes):
        # ----------------------------------------------------------------------
        # Instance Identification & Network Data
        # ----------------------------------------------------------------------
        self.name = name
        self.nodes = nodes          # Initial list of all nodes
        self.satellites = []        # List of satellite nodes
        self.customers = []         # List of customer nodes
        self.network = []           # List of all nodes (Depot, Customers, CDC, Satellites)

        # ----------------------------------------------------------------------
        # Problem Configuration & Constraints
        # ----------------------------------------------------------------------
        self.singleTransfer = False
        self.hard_tw = True         # Hard time window constraint flag
        self.TransferDuration = 0
        self.vrp = True
        self.twovrp = False
        self.consolidated = False

        # ----------------------------------------------------------------------
        # Cost & Objective Parameters
        # ----------------------------------------------------------------------
        self.cost_unit_travel = 1
        self.cost_unit_travel_FEV = 1
        self.cost_vehicle_use = 9 * 60 * 60
        self.cost_vehicle_use_FEV = 9 * 60 * 60 #42000
        self.trip_unit_cost = 5000

        # ----------------------------------------------------------------------
        # Solution Tracking & Best Knowns
        # ----------------------------------------------------------------------
        self.bestObjectve = 10000000
        self.bestSolution = None
        self.best_found = None
        self.best_cost = math.inf

        # ----------------------------------------------------------------------
        # Solution Lists & Feasibility
        # ----------------------------------------------------------------------
        self.heuristics_solutions = []
        self.feas_solutions = []
        self.inf_solutions = []
        self.checked_solutions = []
        self.tabu_list = []
        self.earliest_times_feasible = []
        self.latest_times_feasible = []

        # ----------------------------------------------------------------------
        # Solver/Algorithm Settings
        # ----------------------------------------------------------------------
        self.timelimit_SEmodel = 50
        self.timelimit_FEmodel = 1000
        self.cutoff = math.inf
        self.SAA_rate = 0.01
        self.removal_max = 300
        self.tabu_search_active = True
    
    def GenerateProblem(self, use_case = False):      
        #as in benchmark 50/150/3/3  
        self.AddCDC()
        #as in benchmark 3/3 and satellites have transfer duration
        self.AddSatellites()
        if use_case:
            self.CreateDistMatrix_use_case()
        else:
            self.CreateDistMatrix_coords()
        
        self.distance_matrix = self.DistMatrix

        return
    
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
        # #Grangier, P., Gendreau, M., Lehuédé, F., & Rousseau, L. M. (2016).
        #satellites are located at the city center based on benchmark in the literature by Grangier et al
        #transfer durations or any handling time is assumed zero in the paper/reference paper.
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

    def CreateDistMatrix_coords(self): #use Euclidean distance to calculate distances
        ii = self.CDC
        jj = self.depot
                
        self.DistMatrix = numpy.zeros( (len(self.network), len(self.network)) ) 
        self.travel_matrix = [[math.inf for l in self.network] for l in self.network]

        for i in range(len(self.network)):
            for j in range(len(self.network)):
                ii = self.network[i]
                jj = self.network[j]
                cost =round( math.sqrt((ii.xcoordinate - jj.xcoordinate)*(ii.xcoordinate - jj.xcoordinate) 
                                       + (ii.ycoordinate - jj.ycoordinate)*(ii.ycoordinate - jj.ycoordinate)), 1)
                self.DistMatrix[i][j] = cost
                

        
        for i in range(len(self.network)): #triangle inequality 
            for j in range(len(self.network)):
                for k in range(len(self.network)):
                    cc = self.DistMatrix[i][k] + self.DistMatrix[k][j] 
                    if cc<=self.DistMatrix[i][j]:
                        self.DistMatrix[i][j] = cc
                    self.travel_matrix[i][j] = round(self.DistMatrix[i][j]/self.driving_speed,1)
        
    def check_feasibility_all_nodes(self):
        removed = []

        for ii,node in enumerate(self.network):
            if node.type == "C":
                route = [node]
                result = RH.is_route_feasible(self, route)
                if not result[0]:                
                    removed.append(node) 
        
        if len(removed)>1:
            print("Meeting the demand is infeasible: customer service nodes number of removed: ", len(removed))

            print("solver execution TERMINATED")
            SystemExit()

        return removed
    
def read_vrp_data(instance_name, N):
    """
    Reads VRP data from a benchmark file and extracts nodes and vehicle capacities.

    Args:
        instance_name (str): The filename of the VRP instance data.
        N (int): The maximum index of a node to include.

    Returns:
        tuple: (nodes, capacity_FEV, capacity_SEV)
               nodes (list): List of Node objects.
               capacity_FEV (int): The first capacity listed in the file.
               capacity_SEV (int): The second capacity listed in the file.
    """
    nodes = []
    capacities = []
    projectPath = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(projectPath, "data", "dataset", instance_name)

    # Read the entire content of the file
    with open(filename, "r") as file:
        data = file.read()

    # Split the content into lines and remove leading/trailing whitespace
    lines = [line.strip() for line in data.split("\n")]

    current_section = None
    is_header_row = False
    for line in lines:
        if line in ["VEHICLE", "POINT", "CUSTOMER", "SATELLITE"]:
            current_section = line
            if current_section == "CUSTOMER":
                continue
            is_header_row = True
            continue

        if is_header_row:
            is_header_row = False
            continue

        # Extract data based on the current section
        if current_section == "VEHICLE" and line:
            _, capacity = line.split()
            capacities.append(int(capacity))

        elif current_section == "POINT" and line:
            if line.startswith(("CDC", "SATELLITE", "SLDEPOT")):
                continue

            label, xcoord, ycoord, demand, ready_time, due_date, service_time = line.split()
            # The 'Node' object must be defined and accessible in the scope
            nodes.append(Node("C", int(label), int(xcoord), int(ycoord), int(demand), int(ready_time), int(due_date),
                              int(service_time)))
            nodes[-1].name = label

        elif current_section == "CUSTOMER" and line:
            customer, xcoord, ycoord, demand, ready_time, due_date, service_time = line.split()
            nodes.append(Node("C", int(customer), int(xcoord), int(ycoord), int(demand), int(ready_time), int(due_date),
                              int(service_time)))
            nodes[-1].name = customer

    # Filter nodes based on the maximum index N
    nodes = [node for node in nodes if node.index <= N]

    return nodes, capacities[0], capacities[1]