import pandas as pd
import pickle
import os
import math
import instanceGenerator as instance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

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
            satellite_n = instance.Node("S", row[1], row[3], row[4], 0, 0, 28800, 0) 
            satellite_n.name = row[2]
            satellite_n.canal_node = row[6]
            satellite_n.road_node = row[5]
            satellites.append(satellite_n)

    return satellites
def read_traffic_data(file_content):
# Usage
    #0-RS,1-Nieuwe Amsterdamseweg,2-218813,3-4.96843815,4-52.2791627,5-2019,6-6160,2025,5542,2030,5389,2035,5456,2040,5494,
    unique_links_ids = []
    road_noads = []

    with open(file_content, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for ii, row in enumerate(reader):
            if ii==0:
                continue
            row = [x for x in row if x]
            float_params = [ 3, 4]
            int_params = [2,5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            for i in range(len(row)):
                if i in float_params:
                    row[i] = float(row[i])
                if i in int_params:
                    row[i] = int(row[i])
            unique_links_ids.append(row[2])        
            road_node = instance.Node("RS", row[2], row[3], row[4], 0, 0, 28800, 0) 
            road_node.name = row[1]
            road_node.predictions = []
            if len(row) >=6:
                road_node.predictions.append([row[5], row[6]])
            if len(row) >=8:
                road_node.predictions.append([row[7], row[8]])
            if len(row) >=10:
                road_node.predictions.append([row[9], row[10]])
            if len(row) >=12:
                road_node.predictions.append([row[11], row[12]])
            if len(row) >=14:
                road_node.predictions.append([row[13], row[14]])
            if len(row) >=2:
                road_node.predictions.sort(key=lambda x: x[0])
            road_noads.append(road_node)

    uniqu_ids = list(set(unique_links_ids)) #must be no repetition, must be equal in size 
    if len(road_noads) != len(uniqu_ids):
        print("ERROR: ids are not uniques, multiplication of road segments")
    return road_noads
def read_horeca_data(file_content):
#type,index,xcoordinate,ycoordinate,demand,start_interval,end_interval,unloading_time,road_node,name,addres

    unique_links_ids = []
    satellites = []

    with open(file_content, 'r', encoding='ISO-8859-1') as file:
        reader = csv.reader(file, delimiter=',')
        for ii, row in enumerate(reader):
            if ii==0:
                continue
            row = [x for x in row if x]
            float_params = [2,3, 5,6,7]
            int_params = [1,4,8]
            for i in range(len(row)):
                if i in float_params:
                    row[i] = float(row[i])
                if i in int_params:
                    row[i] = int(row[i])
            unique_links_ids.append(row[1])        
            satellite_n = instance.Node("S", row[1], row[2], row[3], row[4], row[5], row[6], row[7]) 
            satellite_n.addres = row[10]
            satellite_n.name = row[9]
            satellite_n.road_node = row[8]
            satellites.append(satellite_n)

    return satellites
 
def generate_scenarios(road_noads, satellites, horecas, plot_satellites = True, classify_network_wise = True, within_demand = True):  
    if classify_network_wise:
        max_raw = max([sum(x[1] for x in node.predictions)/len(node.predictions) for node in road_noads])
        min_raw = min([sum(x[1] for x in node.predictions)/len(node.predictions) for node in road_noads])
    
    max_x_sat = max(max([sat.xcoordinate for sat in satellites]), max([sat.xcoordinate for sat in horecas])) 
    min_x_sat = min(min([sat.xcoordinate for sat in satellites]), min([sat.xcoordinate for sat in horecas])) 
    max_y_sat = max(max([sat.ycoordinate for sat in satellites]), max([sat.ycoordinate for sat in horecas])) 
    min_y_sat =  min(min([sat.ycoordinate for sat in satellites]), min([sat.ycoordinate for sat in horecas])) 
    if within_demand:        
        road_nodes_within = [node for node in road_noads if node.xcoordinate >= min_x_sat-0.001 and node.xcoordinate <= max_x_sat+ 0.001 and node.ycoordinate >= min_y_sat-0.005 and node.ycoordinate <= max_y_sat + 0.005]
        road_noads = road_nodes_within
        #select inner ring area

    average_traffic = [sum(x[1] for x in node.predictions)/len(node.predictions) for node in road_noads]
    #normalize
    if not classify_network_wise:
        min_raw = min(average_traffic)
        max_raw = max(average_traffic)

    for i in range(len(average_traffic)):
        average_traffic[i] = (average_traffic[i]-min_raw)/(max_raw-min_raw)

    min_n = min(average_traffic)
    max_n = max(average_traffic) 
    diff = max_n - min_n
    thres_1 = diff /4
    thres_2 = 2*diff /4

    sum_degrees = [0,0,0]
    counts_degrees = [0,0,0]
    for i in range(len(average_traffic)):
        road_noads[i].average_traffic = average_traffic[i] 
        road_noads[i].marker = "."           
        if average_traffic[i] >= thres_1 and average_traffic[i] <= thres_2: #M
            road_noads[i].scenario = "M"
            road_noads[i].color = "Orange"
            road_noads[i]
            sum_degrees[1] += average_traffic[i]
            counts_degrees[1] += 1
        elif average_traffic[i] > thres_2:
            road_noads[i].scenario = "H"
            road_noads[i].color = "red"
            sum_degrees[2] += average_traffic[i] #H
            counts_degrees[2] += 1
        else:
            road_noads[i].scenario = "L"
            road_noads[i].color = "green"
            sum_degrees[0] += average_traffic[i] #L
            counts_degrees[0] += 1

    average_traffic_per_scenario = [sum_degrees[i]/counts_degrees[i] for i in range(len(sum_degrees))]

    road_noads.sort(key=lambda x: x.average_traffic)

    #plot roads
    xcors_roads = [node.xcoordinate for node in road_noads] 
    ycors_roads = [node.ycoordinate for node in road_noads] 
    colors_roads = [node.color for node in road_noads] 
    sizes_roads = [10]*len(road_noads) 
    
    plt.scatter(ycors_roads, xcors_roads, c = colors_roads, marker='.', s=sizes_roads, label='roads')
    plt.xlabel('Longitude')
    plt.ylabel('Lattitude')
    plt.legend(loc="upper right")

    
    #plot satellites
    if plot_satellites:
        xcors_satellites = [sat.xcoordinate for sat in satellites]
        ycors_satellites = [sat.ycoordinate for sat in satellites]
        colors_satellites = ['blue']*len(satellites)
        sizes_satellites = [20]*len(satellites)
        plt.scatter(ycors_satellites, xcors_satellites, c = colors_satellites, marker='^', s=sizes_satellites, label='satellites')    
    else:
        rect = patches.Rectangle((min_y_sat, min_x_sat), max_y_sat-min_y_sat, max_x_sat-min_x_sat, linewidth=1, edgecolor='black', facecolor='none') #demand network
        plt.gca().add_patch(rect)
    #markers = [node.marker for node in road_noads] + ['P']*len(satellites)
    if within_demand:        
        xcors_horecas = [horeca.xcoordinate for horeca in horecas]
        ycors_horecas = [horeca.ycoordinate for horeca in horecas]
        colors_horecas = ['black']*len(horecas)
        sizes_horecas = [10]*len(horecas)
        plt.scatter(ycors_horecas, xcors_horecas, c = colors_horecas, marker='.', s=sizes_horecas, label='horecas')
    
    if within_demand:
        plt.title("Traffic predictions nodes inner ring_classified based on demand_region"+str(classify_network_wise))
    else:
        plt.title("Traffic predictions nodes within the city of AMS based on demand_region"+str(classify_network_wise))
    plt.savefig("traffic_info_by_scenario_AMS_is_ring_area_and_classify_city_based"+str(within_demand)+str(classify_network_wise)+".png")
    plt.close()

    return road_noads, satellites, horecas

def get_network_and_scenarios(demand_set = 1, scenario = "equal"):
    projectPath = os.path.dirname(os.path.abspath(__file__))
    files_use_case = projectPath+"/usecase/"
    
    file_traffic = files_use_case + "road_segments_predictions_daytime.txt"
    file_satellites = files_use_case + "satellite_nodes_list.txt"
    file_horecas = files_use_case + "daily_horeca_list"+str(demand_set)+".txt"

    satellites = read_satellites(file_satellites)

    horecas = read_horeca_data(file_horecas)

    if scenario == "equal":
        return [], satellites, horecas
    
    road_noads = read_traffic_data(file_traffic)

    return generate_scenarios(road_noads, satellites, horecas,True, True, True)
