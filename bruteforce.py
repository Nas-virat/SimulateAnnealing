import itertools
import math
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm

import os

import geopandas
import geoplot
from geopandas import GeoDataFrame
import folium

import time

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

def evaluate(solution):
    # calculate total distance
    total_distance = 0
    for i in range(len(solution)):
        if i == len(solution) - 1:
            total_distance += haversine(solution[i][4], solution[i][5], solution[0][4], solution[0][5])
        else:
            total_distance += haversine(solution[i][4], solution[i][5], solution[i+1][4], solution[i+1][5])
    return total_distance


def df_to_list(df):
    return df.values.tolist()

def list_to_df(l, columns=['id','host_id','host_name','borough','lat','long',
    'price','service_fee','number_reviews','geometry']):
    return pd.DataFrame(l, columns=columns)
class TSPBruteForce:
    def __init__(self, cities,evaluate):
        self.cities = cities
        self.num_cities = len(cities)
        self.best_route = None
        self.shortest_distance = float('inf')
        self.evaluate = evaluate

    def calculate_distance(self, city1, city2):
        # Calculate the distance between two cities
        # Replace this method with your own implementation
        return self.evaluate([city1,city2])

    def solve(self):
        # Generate all possible permutations of cities
        permutations = itertools.permutations(self.cities)
        #count = 0
        # Iterate through each permutation
        for route in permutations:
            total_distance = 0
            #print(count)
            # Calculate the total distance for the current permutation
            for i in range(self.num_cities - 1):
                city1 = route[i]
                city2 = route[i + 1]
                distance = self.calculate_distance(city1, city2)
                total_distance += distance

            # Add the distance from the last city back to the starting city
            distance = self.calculate_distance(route[self.num_cities - 1], route[0])
            total_distance += distance

            # Update the best route and shortest distance if necessary
            if total_distance < self.shortest_distance:
                self.best_route = route
                self.shortest_distance = total_distance
            #count += 1
        return self.best_route, self.shortest_distance / 2


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



if __name__ == "__main__":
    # load data and convert to list
    df = pd.read_csv('airbnb_8.csv')

    allHotel = df_to_list(df)
    
    TOTAL_HOTEL = len(allHotel)
    
    ### parameter setup
    cities = [ allHotel[i] for i in range(TOTAL_HOTEL)]
    
    tsp_solver = TSPBruteForce(cities,evaluate)
    start = 0
    end = 0
    start = time.time()
    best_route, shortest_distance = tsp_solver.solve()
    end = time.time()
    
    print("Time:", end - start)
    solution_df = list_to_df(list(best_route))
    solution_df.to_csv('solution/brute_force/solution_bruteforce.csv', index=True)
    dftomap(solution_df, 'solution/brute_force/solution_bruteforce.html')

    print("Best route:")
    
    print("Shortest distance:", shortest_distance)
    
    
    