import pandas as pd
import numpy as np
import re
import random
import json
import matplotlib.pyplot as plt
from operator import itemgetter
import pickle
import networkx as nx

random.seed(42)
import igraph as ig

#*******************************************************************************

# Custom made progressbar to keep track of the progress
def custom_progress_bar(current_step, total_steps, task=""):
	if task != "": task+=" "
	percent_complete = (current_step / total_steps) * 100
	progress = "[" + "#" * int(percent_complete) + "-" * (100 - int(percent_complete)) + "]"
	progress_text = f"{task}{int(percent_complete)}% Complete ({current_step}/{total_steps})"
	print(f"\r{progress_text} {progress}", end="") # Use carriage return to overwrite the previous progress
	if current_step == total_steps: # When all steps are completed, print a newline to move to the next line
		print("", flush=True)
		
#*******************************************************************************

# Generate random network or read in network to do small tests with
def create_smallworld_graph(size):
	G = ig.Graph.Watts_Strogatz(1, size, 5, 0.05)
	for idx, v in enumerate(G.vs):
		v["id"] = idx
	if G.is_connected():
		return G

#*******************************************************************************

# Extract largest connected component if needed
def largest_cc(G):
	components = G.components()
	giant_component_index = components.sizes().index(max(components.sizes()))
	return  components.subgraph(giant_component_index)

#*******************************************************************************

# Read data to graph
def read_graph_file(path):
	with open(path, 'r') as file:
		edgelist = [line.strip().split() for line in file]
	G = ig.Graph.TupleList(edgelist, directed=False, weights=None)
	return G

#*******************************************************************************

# Choose landmarks with Degree
def degree_landmarks(G, num_landmarks):
	att = G.vs.attributes()[0]
	sort_degree = sorted([(G.vs[i][att], G.vs[i].degree()) for i in range(len(G.vs))], key=lambda x: x[1], reverse=True)
	Landmarks = sorted(list(map(itemgetter(0), list(sort_degree[:num_landmarks]))), reverse=False)
	return Landmarks
#------------------------------------------------------------------------------#
# Choose landmarks with PageRank
def page_rank_landmarks(G, num_landmarks):
	pagerank_scores = G.pagerank()#niter=100)
	top_nodes_indices = sorted(range(len(pagerank_scores)), key=lambda k: pagerank_scores[k], reverse=True)[:num_landmarks]
	return list(top_nodes_indices)
#------------------------------------------------------------------------------#
# Choose landmarks with Closeness
def closeness_landmarks(G, num_landmarks):
	closeness_scores = G.closeness(mode="ALL", cutoff=0.9)
	top_nodes_indices = sorted(range(len(closeness_scores)), key=lambda k: closeness_scores[k], reverse=True)[:num_landmarks]
	return list(top_nodes_indices)
#------------------------------------------------------------------------------#
# Choose landmarks with Betweenness
def betweenness_landmarks(G, num_landmarks):
	betweenness_scores = G.betweenness(directed=False, cutoff=5, weights=None)
	top_nodes_indices = sorted(range(len(betweenness_scores)), key=lambda k: betweenness_scores[k], reverse=True)[:num_landmarks]
	return list(top_nodes_indices)
#------------------------------------------------------------------------------#
# Choose landmarks with Random
def random_landmarks(G, num_landmarks):
	att = G.vs.attributes()[0]
	Landmarks = random.sample(G.vs[att], num_landmarks)
	return Landmarks

#*******************************************************************************

# Computing distance landmarks and every other node (efficient)
def saveSpace_calc_landmark_matrix(G, Landmarks, path, method, nodes):
	distances = {}
	for step, mark in enumerate(Landmarks):
		custom_progress_bar(step+1, len(Landmarks), task=f"Calculate landmark matrix {method}")
		inter_dist = {}
		t = 0
		for i in G.shortest_paths(mark, nodes, mode="all", weights=None)[0]: #, algorithm="unweighted")# weights = NULL, predecessors = FALSE, inbound.edges = FALSE,
			inter_dist[nodes[t]] = i
			t += 1
		distances[mark] = inter_dist
	with open(path, 'w') as file:
		json.dump(distances, file)
	del distances

#*******************************************************************************

# Computing distance landmarks and every other node
def calc_landmark_matrix(G, Landmarks, path, method):
	distances = {}
	for step, mark in enumerate(Landmarks):
		custom_progress_bar(step+1, len(Landmarks), task=f"Calculate landmark matrix {method}")
		inter_dist = []
		for i in G.shortest_paths(mark, G.vs, mode="all", weights=None)[0]: #, algorithm="unweighted")# weights = NULL, predecessors = FALSE, inbound.edges = FALSE,
			inter_dist.append(i)
		distances[mark] = inter_dist
	with open(path, 'w') as file:
		json.dump(distances, file)
	del distances

#*******************************************************************************

# Selecting random node pairs
def SelectRandomNodePairs(G, numPairs, randomseed):
	pairs = {}
	random.seed(randomseed)
	nodes = list(G.vs[G.vs.attributes()[0]])
	while len(pairs) < numPairs:
		#pairs.append( random.sample(list(G.vs["id"]), k=2) )
		a,b = random.sample(nodes, k=2)
		pairs[tuple(sorted([a,b]))] = None
	return list(pairs.keys())

#*******************************************************************************

# Calculating actual distances between nodes in pairs
def CalcAndStoreRealDist(G, pairs, path, name):
	real_distances = {}
	for step, (a,b) in enumerate(pairs):
		custom_progress_bar(step+1, len(pairs), task="Calculate real distances")
		shortest = G.shortest_paths(a, b, mode="all", weights=None)[0][0]
		real_distances[str(tuple(sorted([a,b])))] = shortest
	with open(f"{path}{name}_real_distances.json", 'w') as file:
		json.dump(real_distances, file)
	return real_distances

#*******************************************************************************

# input is a list of landmarks, pairs of nodes, and the landmark distance matrix
def CalcEstimateDist(landmarks, pairs, matrix, numLandmarks):
	estimates = []
	for s,t in pairs:
		shortest = float('inf')
		for mark in landmarks[:numLandmarks]:
			dist_st = matrix[str(mark)][str(s)] + matrix[str(mark)][str(t)]
			shortest = min(shortest, dist_st)
		estimates.append(shortest)
	return estimates

#*******************************************************************************

# Picking the right landmark selection function
def LandmarkSelection(G, method, numLandmarks):
	if method == 'D': Landmarks = degree_landmarks(G, numLandmarks) # Degree
	elif method == 'PR': Landmarks = page_rank_landmarks(G, numLandmarks)  # PageRank
	elif method == 'C': Landmarks = closeness_landmarks(G, numLandmarks) # Closeness
	elif method == 'B': Landmarks = betweenness_landmarks(G, numLandmarks) # Betweenness
	elif method == 'R': Landmarks = random_landmarks(G, numLandmarks) # Random
	else: exit("Method is not implementend!")
	return Landmarks

#*******************************************************************************

# Produce loss plot and save it
def combined_loss_plot_methods(losses_zip, landmark_range, methods, title, path):
	fig, plot = plt.subplots(figsize=(8, 8))

	for idx, losses in enumerate(losses_zip):
		plot.plot(landmark_range, losses, linestyle='--', marker='o', label=methods[idx], linewidth=2)

	#plot.set_title("Loss values using different sizes " + title, fontsize=23)
	plot.set_xlabel("Number of " + title, fontsize=21)
	plot.set_ylabel("Computed loss (approx - real) / real", fontsize=21)

	plot.spines['top'].set_visible(False)
	plot.spines['right'].set_visible(False)
	plt.xticks(fontsize=19)
	plt.yticks(fontsize=19)
	plot.legend(fontsize=20)
	plt.tight_layout()
	plt.savefig(f'{path}_losses.png')
	print(f"Losses-plot is stored as {path}_losses.png")
	plt.close()



