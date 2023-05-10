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


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
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
        self.halfsize = len(self.solution) //2
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        
        self.evaluate = solutionEvaluator
        self.neighborOperator = neighborOperator
        
        self.record_wsm = []
        self.record_1 = []
        self.record_2 = []
    
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
                
                cost1 = self.evaluate(neighbor[self.halfsize:]) - self.evaluate(self.solution[self.halfsize:]) 
                cost2 = self.evaluate(neighbor[:self.halfsize]) - self.evaluate(self.solution[:self.halfsize]) 
                
                cost = 0.5*cost1 + 0.5*cost2
                
                #print('neighbor', self.evaluate(neighbor))
                # new solution is better
                if cost < 0:
                    self.solution = neighbor
                else:
                    if random.uniform(0, 1) < math.exp(-cost / self.currTemp):
                        #print('-cost / self.currTemp',-cost / self.currTemp)
                        #print('prob = ',math.exp(-cost / self.currTemp))
                        self.solution = neighbor
                        
                first = self.evaluate(self.solution[self.halfsize:])
                second = self.evaluate(self.solution[:self.halfsize])
                
            self.record_wsm.append(0.5*first+0.5*second)
            self.record_1.append(first)
            self.record_2.append(second)
            self.tempReduction()
            
            print('Temp:',self.currTemp, 'fitness value:',0.5*first+0.5*second)
                    
        return self.solution, self.evaluate(self.solution),self.record_wsm,self.record_1,self.record_2

def pareto_front_plot_minimize(x, y):
    # Sort the arrays based on the first array (x) in ascending order
    sorted_indices = sorted(range(len(x)), key=lambda i: (x[i], y[i]))
    sorted_x = [x[i] for i in sorted_indices]
    sorted_y = [y[i] for i in sorted_indices]

    pareto_front_x = [sorted_x[0]]
    pareto_front_y = [sorted_y[0]]

    for i in range(1, len(sorted_x)):
        if sorted_y[i] <= min(pareto_front_y):
            pareto_front_x.append(sorted_x[i])
            pareto_front_y.append(sorted_y[i])

    plt.plot(pareto_front_x, pareto_front_y, '-o')
    plt.xlabel('First Trip distance')
    plt.ylabel('Second Trip distance')
    plt.title('Pareto Front Plot')
    plt.savefig('solution_multi/solution_pareto_plot.png')
    plt.show()
    
if __name__ == "__main__":
    
    # load data and convert to list
    df = pd.read_csv('airbnb_10.csv')
    
    allHotel = df_to_list(df)
    
    TOTAL_HOTEL = len(allHotel)
    
    ### parameter setup
    initialSolution = [ allHotel[i] for i in range(TOTAL_HOTEL)]
    
    solution = []
    fitness = 0 
    record_cost_1 = []
    record_cost_2 = []
    record_wsm = []
    
    
    optimize = SA(initialSolution,evaluate,neighbor,initialTemp=200,iterationPerTemp=200,alpha=0.98,finalTemp=0.01)
    solution, fitness,record_wsm,record_cost_1,record_cost_2 = optimize.run()

    print('solution',len(record_cost_1))
        
    plt.plot(record_wsm)
    plt.title('Simulated Annealing')
    plt.ylabel('Cost')
    plt.xlabel('iteration')
    plt.savefig('solution_multi/solution_plot.png')
    
    plt.cla()
    plt.clf()
    
    solution_df = list_to_df(list(solution))
    solution_df.to_csv('solution_multi/solution.csv', index=True)
        
    dftomap(solution_df, 'solution_multi/solution.html')
    
    
    print('finish')
    del optimize
    
    

