import osmnx as ox

# Get street network as a graph
G = ox.graph_from_place("Manhattan, New York, USA", network_type="drive")

# Nodes = intersections, Edges = road segments
print(len(G.nodes), "nodes,", len(G.edges), "edges")


# Find shortest path between two nodes
origin = list(G.nodes)[0]
destination = list(G.nodes)[100]
route = ox.shortest_path(G, origin, destination)

# Visualize
ox.plot_graph_route(G, route)
