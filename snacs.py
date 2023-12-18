import pandas as pd
import numpy as np
import re
import random
import json
import matplotlib.pyplot as plt
from operator import itemgetter
import pickle
import networkx as nx
import os
import time
import statistics as stat # statistics about a network
import visualise as v # visualising a smalll graph
import ifunctions as f # functions with igraph


#______________________________Settings_________________________________________
# Graph (set size for generating a graph, set the names of the paths correctly)
size = 4000
#network_path = "generate"
network_path = "networks/out.loc-gowalla_edges"
#network_path = "networks/out.douban"
#network_path = "networks/out.petster-friendships-cat-uniq"
#network_path = "networks/out.loc-brightkite_edges"
#network_path = "networks/out.petster-hamster-friend"

# Set a project name
graph_name = "gowalla"
#graph_name = "douban"
#graph_name = "petster-friendships"
#graph_name = "loc-brightkite_edges"
#graph_name = "hamster_friend"

data_path = f"data/{graph_name}/"



# Landmarks (set the range and the selection method)
landmark_range = [10, 50, 100, 200, 500, 700, 1000]#100, 1000]
method_names = {'R':"random", 'D':"degree", 'PR':"pagerank", 'C':"closeness", 'B':"betweenness"}
landmark_selection_methods = ["R", "D", "PR", "C"] #, "B"] # Random, Degree, PageRank, Closeness, Betweenness
store_path = "data/real_dist_300.csv"
real_dist_path = "data/real_dist_300.csv"
GraphStatistics = True
numPairs = 5000
saveSpace = True
randomseed = 42




#_______________________________________________________________________________
#*******************************************************************************
def create_folder(path):
	if not os.path.exists(path): # Check if the path exists
		os.makedirs(path) # If it doesn't exist, create the folders
		print(f"Path '{path}' created.")
	else: print(f"Path '{path}' already exists. Data might be overwritten!")
#*******************************************************************************
# Initialise
def init():
	"""create graph and if needed find largest connected component"""
	if network_path == "generate":
		print("Generating graph {graph_name}...", end="", flush=True)
		G = f.create_smallworld_graph(size)
	else:
		print(f"Reading graph {network_path}...", end="", flush=True)
		G = f.read_graph_file(network_path)
		G = f.largest_cc(G)
	print("done")
	return G
#*******************************************************************************
def writeClock(process, time, file_path):
	# Open the file in append mode (a+ creates the file if it doesn't exist)
	with open(file_path+"Timer.txt", 'a+') as file: # Write lines with variables a and b
		file.write(f'{process}: {time}\n')
#*******************************************************************************
def main():
	create_folder(data_path) # Prepare data folder
	G = init() # Initialise graph
	writeClock(f"=================== {graph_name} ==================", "", data_path)

	#v.visualise(G)

	# Get statistics about graph
	if input("Do you want to run statistics? (yes/no)") == "yes": # Only required once per dataset and is time expensive
		stat.GetStatistics(G, data_path, name=graph_name)

	#========================== OFFLINE CALCULATIONS ============================#

	# Selecting landmarks for each method
	Landmarks = {}
	G.cache = True # to avoid redundant calculations
	print(f"\033[94m\nSelecting landmarks with different methods\033[0m")
	for method in landmark_selection_methods:
		print(f"Selecting landmarks with method {method_names[method]}...", end="", flush=True)
		Landmarks[method] = f.LandmarkSelection(G, method, max(landmark_range))
		print("done")

		# Select X random pairs of nodes for the experiment
	print(f"\033[94m\nSelecting random pairs\033[0m")
	pairs = f.SelectRandomNodePairs(G, numPairs, randomseed) # returns a list of lists [a,b]
	print("done")

	if saveSpace:
		pair_items = sorted(set(item for tuple_ in pairs for item in tuple_))




	# Creating landmark matrices and store them
	matrix = {}
	print(f"\033[94m\nCalculating landmark matrices\033[0m")
	for method in landmark_selection_methods:

		if saveSpace:
			if os.path.exists(f"{data_path}{method}_matrix.json"):
				if ("yes" == input(f"{method} Matrix exists. Do you want to overwrite this file (yes/no)")):
					f.saveSpace_calc_landmark_matrix(G, Landmarks[method], f"{data_path}{method}_matrix.json", method_names[method], pair_items)
			else: 
				f.saveSpace_calc_landmark_matrix(G, Landmarks[method], f"{data_path}{method}_matrix.json", method_names[method], pair_items)
		else:
			f.calc_landmark_matrix(G, Landmarks[method], f"{data_path}{method}_matrix.json", method_names[method])


	#========================== ONLINE CALCULATIONS =============================#


	# Calculating the real shortest paths between those Xpairs
	print(f"\033[94m\nCalculating real shortest paths\033[0m")

	tik = time.time()
	real_distances = f.CalcAndStoreRealDist(G, pairs, data_path, graph_name)
	tok = time.time()
	writeClock("Calculating real distances", tok-tik, data_path)

	if saveSpace: real_distances = sum(list(real_distances.values()))
	else: real_distances = [real_distances[str(i)] for i in pairs] # to ensure the distances are in correct order



	# Calculating shortest paths estimation using the landmarks
	estimates = {method: {} for method in landmark_selection_methods}
	print(f"\033[94m\nCalculating estimated shortest paths\033[0m")

	for method in landmark_selection_methods:
		with open(f"{data_path}{method}_matrix.json", "r") as file:
			matrix[method] = json.load(file)

		print(f"{method_names[method]}:", end="", flush=True)
		for numLandmarks in landmark_range:
			print(f" {numLandmarks}", end="", flush=True)
			tik = time.time()
			if saveSpace: 
				estimates[method][numLandmarks] = sum(f.CalcEstimateDist(Landmarks[method], pairs, matrix[method], numLandmarks))
			else: 
				estimates[method][numLandmarks] = f.CalcEstimateDist(Landmarks[method], pairs, matrix[method], numLandmarks)
			tok = time.time()
			writeClock(f"Estimating distances. Method: {method}, NumLandmarks: {numLandmarks}", tok-tik, data_path)
		print()
		del matrix[method]
	
	print("done")



	# Calculate losses (differences) between estimate distances and real distances
	losses = []
	print(f"\033[94m\nCalculating losses\033[0m")
	for method in landmark_selection_methods:
		losses_per_method = []
		for numLandmarks in landmark_range:
			if saveSpace: losses_per_method.append(abs(estimates[method][numLandmarks]-real_distances)/real_distances)
			else: losses_per_method.append(abs(sum(estimates[method][numLandmarks]-sum(real_distances))/sum(real_distances)))
		losses.append(losses_per_method)
	print("done")



	# Make a losses plot for different methods with on the x-axis different num landmarks
	print(f"\033[94m\nProducing losses-plot\033[0m")
	f.combined_loss_plot_methods(losses, 
										landmark_range, 
										[method_names[i] for i in landmark_selection_methods], 
										"landmarks", 
										path = data_path+graph_name)



	exit()

#*******************************************************************************

if __name__ == '__main__':
	main()



