import igraph as ig
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from igraph import mean

#*******************************************************************************

# Custom made progress bar to keep track of processes
def custom_progress_bar(current_step, total_steps):
	percent_complete = (current_step / total_steps) * 100
	progress = "[" + "#" * int(percent_complete) + "-" * (100 - int(percent_complete)) + "]"
	progress_text = f"{int(percent_complete)}% Complete ({current_step}/{total_steps})"
	print(f"\r{progress_text} {progress}", end="") # Use carriage return to overwrite the previous progress
	if current_step == total_steps: # When all steps are completed, print a newline to move to the next line
		print("", flush=True)
#*******************************************************************************

# Given the shortest pathlengths from one node to all others, add them to the counting dict
def count_distanceDistribution(pathlengths, paths_from_current_node):
	for length in paths_from_current_node: # count pathlengths
		if length not in pathlengths: pathlengths[length] = 0
		pathlengths[length] += 1
	return pathlengths

#*******************************************************************************

# Making a shortest paths distance distribution (or estimation)
def DistanceDistribution(G):
	# Initialize dictionaries
	shortest_path_lengths = {}
	pathlengths = {}

	# Get a subset of nodes to make the problem smaller and make an estimate
	samplesize = 1000 #100000
	print(f"\033[94mCalculating pathlengths\033[0m")
	nrNodes = G.vcount()
	att = G.vs.attributes()[0]

	if nrNodes > samplesize: # if G is too large, make an estimate
		subset_nodes = np.random.choice(list(G.vs[att]), size=samplesize, replace=False)
	else: subset_nodes = list(G.vs[att])

	# Get all shortest paths
	for step, node in enumerate(subset_nodes):
		custom_progress_bar(step+1, len(subset_nodes)) # Progressbar
		sp_lengths = G.shortest_paths(source=node, mode=ig.ALL)
		pathlengths = count_distanceDistribution(pathlengths, sp_lengths[0]) # keep dictionary with counts of pathlengths

	del pathlengths[0] # remove the 0 length pathlengths
	if nrNodes > samplesize: # calculate back estamination by mulitiplying
		pathlengths = {key: value * nrNodes/samplesize for key, value in pathlengths.items()}
	return pathlengths

#*******************************************************************************

# Gets dictionary as input and turns it into barplot
def DistributionPlot(D, Measure="Distance", Title="DistributionPlot", path=""):
	#D = {k: v for k, v in sorted(D.items(), key=lambda item: item[1], reverse=True)}
	D = dict(sorted(D.items()))
	print(f"\033[1;92m{Title}-Make {Measure} distribution plot: \033[0m", end ="")
	names = list(D.keys())
	values = list(D.values())
	plt.bar(range(len(D)), values, tick_label=names, color='gray')
	#plt.scatter(list(D.keys()), list(D.values()), marker='.', c='b', s=10, label='Counts')
	plt.xlabel(Measure, fontsize=21)
	plt.ylabel('Frequency', fontsize=21)
	plt.xticks(fontsize=19)#, rotation=90)
	plt.yticks(fontsize=19)
	plt.yscale('log')
	plt.tight_layout()
	plt.savefig(path+Title+'.png', format='png')
	print(f"Plot saved as {Title}.png")
	#plt.show()
	plt.close()

#*******************************************************************************

# Convert graph from networkx to igraph
def nx_to_igraph(nx_graph):
	igraph_graph = ig.Graph() # Create an igraph Graph object
	igraph_graph.add_vertices(list(nx_graph.nodes)) # Add vertices from NetworkX graph to igraph graph
	igraph_graph.add_edges(nx_graph.edges) # Add edges from NetworkX graph to igraph graph
	return igraph_graph

#*******************************************************************************

# Storing statistics
def save_statistics_to_file(path, name, variable_names, values):
	if len(variable_names) != len(values): print("Statistics not saved. Error: unequal lenghts!")
	else:
		with open(path+f"{name}_statistics.txt", 'w') as file:
			[file.write(f"{variable_names[i]}:\t{values[i]}\n") for i in range(len(variable_names))]

#*******************************************************************************

# Calculating network characteristics
def GetStatistics(G, path, name="test"):

	if type(G) is nx.Graph: # Check if the object is an NetworkX graph
		G = nx_to_igraph(G) # convert to networkx

	values = [G.vcount(), G.ecount(), G.transitivity_undirected(), mean(G.degree())]
	print(f"Number of nodes: {values[0]}")
	print(f"Number of edges: {values[1]}")
	print(f"Clustering coefficient: {values[2]}")
	print(f"Average degree: {values[3]}")
	save_statistics_to_file(path, name, ["Nodes", "Edges", "Clustering Coefficient", "Average degree"], values)

	pathlengths = DistanceDistribution(G) # Make a distance distribution
	DistributionPlot(pathlengths, Measure="Distance", Title=name+"_DistanceDistrPlot", path=path)

#*******************************************************************************
