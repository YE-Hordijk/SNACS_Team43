import networkx as nx
from igraph import Graph as igraphGraph
import matplotlib.pyplot as plt

#*******************************************************************************
def DrawGraph(G):
	pos = nx.spring_layout(G)
	nx.draw(G, pos, with_labels=True, node_color='skyblue', font_size=9, font_color='black', width=1.5) # node_size=300
	#edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
	#nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red')
	plt.axis('off')
	plt.show()


#*******************************************************************************
def convert_to_networkx(graph):
	if isinstance(graph, igraphGraph):
		nx_graph = nx.Graph(graph.get_edgelist()) # Convert igraph graph to networkx graph
		return nx_graph
	else:
		return graph
#*******************************************************************************
def visualise(G):
	G = convert_to_networkx(G)
	DrawGraph(G)
#*******************************************************************************



