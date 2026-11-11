import sys
import copy
BigNumber = 100000

class two_echelon_solution:
    def __init__(self):
        self.is_feasible = True #default feasible vehicle
        self.index = 0 #the order of the solution found in BB process
        self.objective = BigNumber
        self.nFEVs = 0
        self.nSEVs = 0
        self.travelWater = 0
        self.travelStreet = 0
        self.routes_sevs = [] #list of used arcs
        self.routes_fevs = [] #list of used arcs
        self.trips = [] #list of transfer task
        self.satellites = []
        self.checked_optimality = False


        def deep_copy(self):
            newSolution = two_echelon_solution()
            newSolution.is_feasible = self.is_feasible
            newSolution.index = self.index
            newSolution.objective = self.objective
            newSolution.nFEVs = self.nFEVs
            newSolution.nSEVs = self.nSEVs
            newSolution.travelWater = self.travelWater
            newSolution.travelStreet = self.travelStreet
            newSolution.routes_sevs = [copy.deepcopy(xx) for xx in self.routes_sevs] 
            newSolution.routes_fevs = [copy.deepcopy(xx) for xx in self.routes_fevs] 
            newSolution.trips = self.trips
            newSolution.satellites = self.satellites
            newSolution.checked_optimality = self.checked_optimality

            return newSolution

        def check_solution_to_optimality(self, problem_instance):


            return
class trip:
    def __init__(self):
        self.node_after = None
        self.node_before = None
        self.satellite = None
        self.load = None
        self.start_time = None


class street_vehicle:
    def __init__(self):
        self.isFeasible = True #default feasible vehicle
        self.index = -1
        self.sequence  = []
        self.cost = BigNumber
        self.travel = BigNumber # distance traveled
        self.makespan = BigNumber # arrival time at depot at the end
        self.waiting = BigNumber # sum diff arrival and start interval 
        self.start = 0
        self.end = 0
        self.load = 0
        self.transfers = []
        self.arcs = []
        self.available_saving = 0
        self.modified_waiting = 0
        self.idle = 0

    def __repr__(self):
        return f"Route {self.sequence} with index {self.index} {'is feasible' if self.isFeasible else ''} and objectives travel={self.travel}, makespan={self.makespan}, waiting={self.waiting}"
        
    def deep_copy(self):
        newRoute = street_vehicle()
        newRoute.isFeasible = self.isFeasible
        newRoute.sequence.extend(self.sequence) #check extend vs manipulation
        newRoute.travel = self.travel
        newRoute.makespan = self.makespan
        newRoute.waiting = self.waiting
        newRoute.start = self.start
        newRoute.end = self.end
        newRoute.load = self.load
        newRoute.transfers.extend(self.transfers)
        newRoute.index = self.index
        newRoute.arcs =  self.arcs
        newRoute.available_saving = self.available_saving
        newRoute.modified_waiting = self.modified_waiting 
        newRoute.idle = self.idle      
        
        return newRoute
    
    def generate_arcs(self):
        self.arcs = []
        self.arcs.append([0, self.sequence[0].index])

        for ii, node_s in enumerate(self.sequence):
            if ii < len(self.sequence) - 1 :
                self.arcs.append([node_s.index, self.sequence[ii + 1].index])
            else:
                self.arcs.append([node_s.index, 0])
        
        return
    
    def generate_transfers_full_load(self, problem, remove = False):
        self.transfers = []
        
        transfer_node = self.sequence[-1].index
        loads = []
        while transfer_node:
            self.transfers.append(transfer_node)
            next_cust = transfer_node    
            load_current =  problem.nodes[transfer_node].demand
            
            incoming = next(j for [j,k] in self.arcs if k == next_cust)
            while  incoming != 0 and load_current + problem.nodes[incoming].demand <= problem.CarCapacity:                    
                load_current += problem.nodes[incoming].demand
                next_cust = incoming
                incoming = next(j for [j,k] in self.arcs if k == next_cust)           

            loads.append(load_current)

            if incoming != 0:
                transfer_node = incoming
            else:
                transfer_node = None
            
            self.feas_check(problem)

        if sum(loads) != sum(x.demand for x in self.sequence):
            print("infeasible transfer generation")
        
        removed_nodes = []
        infeasible_node = self.feas_check(problem)
        while not self.isFeasible and remove:
            removed_nodes.append(infeasible_node) 
            self.sequence.remove(infeasible_node)
            if infeasible_node.index in self.transfers:
                self.transfers.remove(infeasible_node.index)
            print("infeasible transfer assignment")
            infeasible_node = self.feas_check(problem)
        

        return removed_nodes        

    def feas_check(self, problem): #no load capacity        
        self.isFeasible = True

        #arrival start at first customer
        self.start = problem.DistMatrix[0][self.sequence[0].index]
        self.travel = self.start
        arrival_to_next = self.start

        for cc, node in enumerate(self.sequence):
            if arrival_to_next > node.latest:
                self.isFeasible = False                
                return node
            
            node.arrival = arrival_to_next
            node.service_start = max(arrival_to_next, node.earliest)
            node.waiting = node.service_start - node.arrival
            
            if cc == len(self.sequence) - 1 :
                next = 0
            else:
                next = self.sequence[cc + 1].index
            arrival_to_next = node.service_start + node.serviceTime + problem.DistMatrix[node.index][next]
            self.travel += problem.DistMatrix[node.index][next]
            
            if node.index in self.transfers:
                minDetour = min(problem.DistMatrix[node.index][s] + problem.DistMatrix[s][next] for s in problem.satellites)
                arrival_to_next += problem.TransferDuration + minDetour
                self.travel += minDetour
            node.modified_slack = problem.nodes[next].latest - problem.DistMatrix[node.index][next] -  node.serviceTime - node.arrival

        if self.end > problem.nodes[0].latest:
            self.isFeasible = False
            return self.sequence[-1]
        self.end = arrival_to_next

        next = problem.nodes[0]
        next.slack_time = next.latest - self.end

        self.idle = 0
        for i in range(len(self.sequence)-1, -1, -1):
            self.sequence[i].pseudoLatest = min(next.latest - problem.DistMatrix[self.sequence[i].index][next.index] - self.sequence[i].serviceTime, self.sequence[i].latest)
            self.sequence[i].slack_time = min(self.sequence[i].pseudoLatest - self.sequence[i].service_start, next.slack_time) +  self.sequence[i].waiting
            if self.sequence[i].waiting > 0:
                self.idle += self.sequence[i].slack_time
            next = self.sequence[i]
        self.waiting = max(0, sum(xx.waiting for xx in self.sequence) - (self.sequence[0].pseudoLatest - self.sequence[0].arrival)) # - self.start #check makespan-waiting correction
        self.serving_time = sum(xx.serviceTime for xx in self.sequence)
        self.makespan = self.travel + self.waiting + self.serving_time

        #best working version
        self.modified_waiting = min(0, sum(xx.waiting for xx in self.sequence) - min(xx.modified_slack for xx in self.sequence)) # - self.start #check makespan-waiting correction
        self.available_saving = self.modified_waiting + self.makespan
        self.idle += self.travel
        return

class Insertion:
    def __init__(self,details = [False, BigNumber, -1,-1,-1]):        
        self.isFeasible = details[0]
        self.insertionCost = details[1]
        self.insertionRoute = details[2]
        self.insertionIndex = details[3]
        self.unservedWP = details[4]
    def PerformInsertion(self,problem, partialSolution):
        if self.isFeasible:
            #perform insertion
            if(self.insertionRoute != -1):
                #insert an existing route
                self.insertionRoute.insert(self.insertionIndex,self.unservedWP)
                partialSolution.CheckFeasibilityStreet(problem)
            else:
                partialSolution.streetSequence.append([self.unservedWP])
                partialSolution.CheckFeasibilityStreet(problem)
        else:
            print("infeasible insertion operation cannot be performed")
