#!/usr/bin/env python
# coding: utf-8

# In[1]:

import instanceGenerator
import time
import sys
from datetime import date
import math
import Multi_level_destroy as MLD
import Repair_heuristcs as RH




def generate_statistics_report_files(scenario_name):
    
    report = "report_"+str(scenario_name)+str(time.time())+".txt"
    
    file1 = open(report,"w")
    file1.close()

    initial_report = "initial_report_"+str(scenario_name)+str(time.time())+str(date.today())+".txt"
    file1 = open(initial_report,"w")
    file1.close()

    multi_report = "multi_report_"+str(scenario_name)+str(time.time())+str(date.today())+".txt"
    file1 = open(multi_report,"w")
    file1.close()


    return report, initial_report, multi_report

if __name__ == "__main__":    
    ##problem inputs and parameters    
    benhmark_problem = True #bool(sys.argv[1])  #the min index 0
    nof_demand_scenario = 1 #int(sys.argv[2]) 
    n_satellites = 8# int(sys.argv[3]) #important parameter for the case study, defining the network size
    synchronized = True# bool(sys.argv[4])# 
    #method parameters
    start_cons = 5 # int(sys.argv[5])
    rep_localS = 5# int(sys.argv[6])
    iter_localS = 500# int(sys.argv[7])
    iter_ALNS =  10#int(sys.argv[8])
    tabu_search = True # int(sys.argv[9])
    rep_its = 2#int(sys.argv[10])

    scenario_name = 0#sys.argv[11]    
    report, initial_report, multi_report = generate_statistics_report_files(scenario_name)

    instances = [ "c208.txt" ,"c201.txt", "r201.txt","r208.txt",  "rc201.txt","rc208.txt"]
    
    for ds in range(nof_demand_scenario): 
        if benhmark_problem:
            problem = instanceGenerator.readGenerate_benchmark(instances[ds],size=100)##instance, size
        else:
            problem = instanceGenerator.read_generate_case(ds ,nof_horeca=1000, nof_satellites=n_satellites)#
        
        #reporting statistics and results to files
        problem.initial_report = initial_report
        problem.multi_report = multi_report
        problem.report = report  

        #synchronization and tabu search parameters
        problem.tabu_search_active = tabu_search
        problem.synch_check = synchronized #synch or asynch    

        problem.points_found = []  #to plot the found solutions in the objective space in the order of finding
        
        
        #to see whether possible to serve all customers (assigning each customer to a single vehicle)    
        removed = instanceGenerator.check_feasibility_all_nodes(problem)
        #if not feasible, we exit the program by reporting the infeasibility.
        
        #2e-vrp is infeasible if all customers cannot be served
        #Otherwise,  its feasible and we continue with the solution process.
        final_solution = None # RH.read_initial_SEV_routes_benchmark(problem, demand_set,  usecase = True)
        
        #time required for building the problem is ignored and not reported. 
        problem.modelStartTime = time.time() #we keep the computing statistics after instance generation
        iters = 0        
        
        while iters< rep_its:
            iters += 1
            problem.its_iter = iters
            final_solution = MLD.Solve_by_decomposition(random_num= 32, 
                                                    problem_instance=problem, 
                                                    nstart= start_cons, 
                                                    repLS = rep_localS, 
                                                    iterLS = iter_localS, 
                                                    iterALNS = iter_ALNS)
        final_solution = problem.best_found
        RH.write_out_solution(problem, final_solution)        
        RH.plot_found_solutions(problem)
        
    
       
    
