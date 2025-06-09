# The core part of the experiment: to find a better sampling method
# for each function defined here: take a networkx graph and number of samples
# return single source shortest distance to sampled points.
import numpy as np
import networkx as nx


def random_sampling(graph: nx.Graph, samples_per_source):
    src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    # random
    target = np.random.choice(graph.number_of_nodes(), (samples_per_source, ), replace=False)
    distance = [distance[t] for t in target]
    return [src] * samples_per_source, target.tolist(), distance

def distance_based(graph: nx.Graph, samples_per_source):
    src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    # random
    vertices = np.arange(graph.number_of_nodes())
    row_num = int(np.around(graph.number_of_nodes() ** 0.5))
    hops = abs(src // row_num - vertices // row_num) + abs(src % row_num - vertices % row_num) + 1
    probs = 1 / hops
    probs = probs / probs.sum()
    target = np.random.choice(vertices, (samples_per_source, ), p=probs, replace=False)
    distance = [distance[t] for t in target]
    return [src] * samples_per_source, target.tolist(), distance






def sample_queries(method, graph, samples_per_source):
    if method == "random":
        return random_sampling(graph, samples_per_source)
    elif method == "distance":
        return distance_based(graph, samples_per_source)
    else:
        return NotImplementedError(f"{method} is not supported.")