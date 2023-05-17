import math
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import os

import geopandas
import geoplot
from geopandas import GeoDataFrame
import folium

import time

BRONX_HOTEL = []
BROOKLYN_HOTEL = []
MANHATTAN_HOTEL = []
QUEENS_HOTEL = []
STATEN_ISLAND_HOTEL = []


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))

def df_to_list(df):
    return df.values.tolist()

def list_to_df(l, columns=['id','host_id','host_name','borough','lat','long',
    'price','service_fee','number_reviews','geometry']):
    return pd.DataFrame(l, columns=columns)
    
def evaluate(solution):
    # calculate total distance
    total_distance = 0
    for i in range(len(solution)):
        if i == len(solution) - 1:
            total_distance += haversine(solution[i][4], solution[i][5], solution[0][4], solution[0][5])
        else:
            total_distance += haversine(solution[i][4], solution[i][5], solution[i+1][4], solution[i+1][5])
    return total_distance

def evaluate_wsm(solution, weight=0.3):
    price = 0
    total_distance = 0
    for i in range(len(solution)):
        if i == len(solution) - 1:
            total_distance += haversine(solution[i][4], solution[i][5], solution[0][4], solution[0][5])
        else:
            total_distance += haversine(solution[i][4], solution[i][5], solution[i+1][4], solution[i+1][5])
            
    for i in range(len(solution)):
        price = price + solution[i][6] + solution[i][7]
            
            
    return (1-weight)*total_distance + weight * price


def random_index_hotel(allHotel):
    return random.randint(0, len(allHotel) - 1)

def neighbor(solution):
    # randomly select 2 hotels
    index1 = random.randint(0, len(solution) - 1)
    index2 = random.randint(0, len(solution) - 1)
    
    _neighbor = solution.copy()
    
    _neighbor[index1], _neighbor[index2] = _neighbor[index2], _neighbor[index1]
    
    return _neighbor

def dftomap(df_solution,file_path):
    m = folium.Map(location=[40.69183, -73.86272], zoom_start=11)
        
    folium.TileLayer('stamenterrain').add_to(m)

    place_lat = df_solution['lat'].values.tolist()
    place_lng = df_solution['long'].values.tolist()
    price = df_solution['price'].values.tolist()
    fee = df_solution['service_fee'].values.tolist()

    points = []
    for i in range(len(place_lat)):
        points.append([place_lat[i], place_lng[i]])
    points.append([place_lat[0], place_lng[0]])

    for index,lat in enumerate(place_lat):
        folium.Marker([lat, 
                    place_lng[index]],
                    popup=(f"Hotel:{index}\nPrice:{price[index]}\nService Fee:{fee[index]}"),
                    icon = folium.Icon(color='darkblue',icon='plus')).add_to(m)
    folium.PolyLine(points, color='red').add_to(m)
    
    m.save(file_path)

class SA:
    
    def __init__(self,intialsolution,solutionEvaluator, neighborOperator,initialTemp=300,iterationPerTemp=1000,alpha=0.98,finalTemp=0.01,):
        self.currTemp = initialTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.solution = intialsolution
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        
        self.evaluate = solutionEvaluator
        self.neighborOperator = neighborOperator
        
        self.record = []
        self.record.append(self.evaluate(self.solution))
    
    def resetrecord(self):
        self.record = []
        
    def tempReduction(self):
        self.currTemp = self.currTemp * self.alpha
        
    def isTerminationCriteriaMet(self):
        return self.currTemp <= self.finalTemp
    
    def run(self):
        
        self.resetrecord()
        while not self.isTerminationCriteriaMet():
           
            for i in range(self.iterationPerTemp):
               
                #print('solution', self.evaluate(self.solution))
                
                neighbor = self.neighborOperator(self.solution)
                
                cost = self.evaluate(neighbor) - self.evaluate(self.solution) 
                
                #print('neighbor', self.evaluate(neighbor))
                # new solution is better
                if cost < 0:
                    self.solution = neighbor
                else:
                    if random.uniform(0, 1) < math.exp(-cost / self.currTemp):
                        #print('-cost / self.currTemp',-cost / self.currTemp)
                        #print('prob = ',math.exp(-cost / self.currTemp))
                        self.solution = neighbor
                        
            self.record.append(self.evaluate(self.solution))
            self.tempReduction()
            
            #print('Temp:',self.currTemp, 'fitness value:',self.evaluate(self.solution))
                    
        return self.solution, self.evaluate(self.solution),self.record



def average_lists_by_index(list_of_lists):
    if not list_of_lists:
        return None

    num_lists = len(list_of_lists)
    list_length = len(list_of_lists[0])
    
    averages = [0] * list_length  # Initialize the averages list with zeros
    
    for sublist in list_of_lists:
        for i in range(list_length):
            averages[i] += sublist[i]
    
    averages = [average / num_lists for average in averages]  # Calculate the averages by dividing by the number of lists
    
    return averages
 
if __name__ == "__main__":
    
    # load data and convert to list
    df = pd.read_csv('airbnb_12.csv')

    allHotel = df_to_list(df)
    
    TOTAL_HOTEL = len(allHotel)
    
    ### parameter setup
    initialSolution = [ allHotel[i] for i in range(TOTAL_HOTEL)]
    evaluate_ = evaluate
    
    
    list_parameter = [0.98,0.9,0.70,0.6,0.5]
    
    
    record = []
    all_record = []
    avg_record = []
    
    cost_all = []
    
    best_cost = float('inf')
    best_solution = []
    
    start = 0
    end = 0
    print('\n\n\n')
    
    for i in range(1):
        print(i)
        optimize = SA(intialsolution=initialSolution,solutionEvaluator=evaluate_,neighborOperator=neighbor, \
                initialTemp=100,iterationPerTemp=1,alpha=0.98,finalTemp=0.01)
        start = time.time()
        solution, cost,record = optimize.run()
        end = time.time()
        all_record.append(record)
        
        cost_all.append(cost)
        
        if best_cost > cost:
            best_cost = cost
            best_solution = solution
        
        del optimize
        
    avg_record = average_lists_by_index(all_record)
    
    print('time',end-start)
    print('Best solution fitness value:',best_cost)
    print('The average best fitness value is:',sum(cost_all )/len(cost_all))
    
    plt.plot(avg_record)
    plt.title('Simulated Annealing')
    plt.ylabel('Cost')
    plt.xlabel('iteration')
    plt.savefig('solution/solution_plot.png')
    
    solution_df = list_to_df(list(best_solution))
    solution_df.to_csv('solution/solution.csv', index=True)
    dftomap(solution_df, 'solution/solution.html')
    
    print(cost)
    '''
    for parameter in list_parameter:
        
        optimize = SA(intialsolution=initialSolution,solutionEvaluator=evaluate_,neighborOperator=neighbor, \
            initialTemp=200,iterationPerTemp=200,alpha=parameter,finalTemp=0.01)
        
        solution, cost,record = optimize.run()
        print(solution, cost)
        
        cost_all.append(cost)
        
        
        plt.plot(record)
        plt.title('Simulated Annealing alpha  = ' + str(parameter))
        plt.ylabel('Cost')
        plt.xlabel('iteration')
        plt.savefig('solution/solution'+str(parameter)+'_plot.png')
        
        
        solution_df = list_to_df(list(solution))
        solution_df.to_csv('solution/solution_'+str(parameter)+'.csv', index=True)
        
        dftomap(solution_df, 'solution/solution_'+str(parameter)+'.html')
        
    
    for i in range(len(list_parameter)):
        plt.plot(record[i],label=str(list_parameter[i]))
        
    plt.title('Compare different alpha Simulated Annealing')
    plt.legend(list_parameter, loc='upper right')
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.savefig('solution/solution_all_plot.png')
    
    
    for i in range(len(list_parameter)):
        print('alpha  = ',list_parameter[i],'cost = ',cost_all[i])
    '''
    
        
        
    
    
    
    
    
    
    
    
    
    
            
        
       
    
    
