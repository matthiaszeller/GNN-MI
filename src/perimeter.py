

from copy import deepcopy

import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.utils import to_scipy_sparse_matrix


def compute_perimeters(data):
    a1, a2 = find_edge_annuluses(data.edge_index)
    path = find_path(data, a1, a2)

    g, old, new, mapping, reverse_mapping = cut_duplicate_join(data, path)
    pred, dist = compute_shortest_paths_floyd_warshall(g)

    perimeters = {
        path[0][0]: path[1]
        for path in iter_shortest_paths(pred, dist, mapping, reverse_mapping, old)
    }
    return perimeters


def get_neighbours(nnz_r, nnz_c, i):
    mask = nnz_r == i
    return nnz_c[mask]


def find_edge_annulus(edge_index, start_node, maxiter=1000):
    nnz_r, nnz_c = edge_index.numpy()
    # Set of visited nodes for efficient computations
    annulus = set([start_node])
    # List of visited nodes to keep trace of the order
    nodes = [start_node]

    current = start_node
    hop_counter = 0
    while hop_counter < maxiter:
        # while current != start_node or hop_counter == 0:
        # Neighbours of current nodes
        unfiltered_neighs = set(get_neighbours(nnz_r, nnz_c, current))
        # Remove already-visited nodes
        neighs = unfiltered_neighs.difference(annulus)

        # Detect termination: got back to the original node
        if hop_counter > 1 and start_node in unfiltered_neighs:
            return nodes

        # Compute degree of each neighbour
        n_neighs = {
            (i, len(get_neighbours(nnz_r, nnz_c, i))) for i in neighs
        }
        # Select neighbour with minimal degree
        next_node = min(n_neighs, key=lambda e: e[1])[0]

        # Add selected node
        annulus.add(next_node)
        nodes.append(next_node)
        hop_counter += 1
        current = next_node

    return nodes


def find_edge_annuluses(edge_index):
    # Find nodes with minimal number of neighbours
    A = to_scipy_sparse_matrix(edge_index)
    degree = np.array(A.sum(axis=0)).flatten()
    sorted_degree = np.argsort(degree)

    annulus1 = find_edge_annulus(edge_index, sorted_degree[0])
    # Remove all nodes from annulus 1
    mask = np.isin(sorted_degree, annulus1)
    sorted_degree = sorted_degree[~mask]

    annulus2 = find_edge_annulus(edge_index, sorted_degree[0])

    return annulus1, annulus2


def find_path(data, a1, a2):
    net = to_networkx(data)
    return nx.shortest_path(net, source=a1[0], target=a2[0])


def get_all_neighbours(edge_index, nodes):
    """Get all neighbours of given nodes without including `nodes`"""
    neighs = set()
    for i in nodes:
        neighs.update(get_neighbours(*edge_index, i).tolist())
    neighs.difference_update(nodes)

    return neighs


def prepare_cut(data, path):
    path_neighbours = get_all_neighbours(data.edge_index, path)

    net = to_networkx(data, to_undirected=True)

    # Consider subgraph with only nodes of the path neighbours
    net = net.subgraph(path_neighbours)

    # sub, mapped_nodes = subgraph(data, path_neighbours, map_nodes=path_neighbours)

    # Find the "edges" of the unfolded graph, those are two disconnected components
    components = tuple(nx.connected_components(net))
    if len(components) != 2:
        raise ValueError(f'should have 2 CC, have {len(components)}')

    return components


def add_edge_weights(data):
    out = data.clone()
    D = distance_matrix(out.coord, out.coord)
    i1, i2 = out.edge_index
    out.edge_weight = D[i1, i2]
    return out


def cut_and_duplicate(weighted_graph, path):
    max_node = max(weighted_graph.nodes)

    g1 = deepcopy(weighted_graph)
    old_nodes = set(g1.nodes)
    # Cut the graph along the path
    g1.remove_nodes_from(path)

    # Only map nodes of g1 and not of net
    node_mapping = {
        original: original + max_node + 1
        for original in g1.nodes
    }

    g2 = deepcopy(g1)
    # Relabel nodes of second graph
    g2 = nx.relabel_nodes(g2, node_mapping)

    new_nodes = set(g2.nodes)
    return g1, g2, old_nodes, new_nodes, node_mapping


def join(g1, g2, path_with_attrs, path_edges, c1, c2, mapping):
    # Join the two graphs
    g = nx.union(g1, g2)
    # Nodes of the path will be automatically added when connecting the two graphs,
    # but still need to set their coordinates
    g.add_nodes_from(path_with_attrs)

    # Connect the two graphs
    # u is a path node, v a node either in c1, c2 or in the path itself
    c1, c2 = set(c1), set(c2)
    path = set(e[0] for e in path_with_attrs)
    for u, v, d in path_edges:
        assert (v in c1) or (v in c2) or (v in path)
        # If node v is in c2 -> map it to dual graph
        # i.e, join the graphs on the c1 side
        if v in c2:
            v = mapping[v]

        g.add_edge(u, v, **d)

    # Add the "dual path" nodes and its connections
    max_node = max(g.nodes)
    for i, (u, d) in enumerate(path_with_attrs):
        dual_u = i + max_node + 1
        # Update node mapping
        mapping[u] = dual_u
        # Add the new node
        g.add_node(dual_u, **d)
    # Connectivity for dual path
    for u, v, d in path_edges:
        # If edge within original path: duplicate edge in dual path
        if u in path and v in path:
            g.add_edge(mapping[u], mapping[v], **d)
        # otherwise, connect dual path
        # we joined on c2 side,
        # so need to joint the dual of c1 to the dual path nodes
        if v in c1:
            g.add_edge(mapping[u], mapping[v], **d)

    return g, mapping


def collect_nx_edges(g, nodes):
    edges = []
    for u in nodes:
        for v, d in g[u].items():
            edges.append((u, v, d))
    return edges


def cut_duplicate_join(data, path):
    data_weighted = add_edge_weights(data)
    weighted_graph = to_networkx(data_weighted, to_undirected=True, edge_attrs=['edge_weight'], node_attrs=['coord'])

    path_edges = collect_nx_edges(weighted_graph, path)

    g1, g2, old_nodes, new_nodes, mapping = cut_and_duplicate(weighted_graph, path)

    c1, c2 = prepare_cut(data, path)

    path_with_attrs = [
        (u, weighted_graph.nodes[u]) for u in path
    ]
    g, mapping = join(g1, g2, path_with_attrs, path_edges, c1, c2, mapping)

    # Sanity check
    nodes_original, nodes_dual = set(mapping.keys()), set(mapping.values())
    assert len(nodes_original.intersection(nodes_dual)) == 0
    # Reverse mapping
    reverse_mapping = {v: k for k, v in mapping.items()}
    # Map original nodes to themselves
    for u in mapping.keys():
        reverse_mapping[u] = u

    return g, old_nodes, new_nodes, mapping, reverse_mapping


def get_path_from_floyd(predecessors, distances, source, target):
    """Returns the path from target to source (i.e., reversed order)
    predecessors is output of networkx.floyd_warshall_predecessor_and_distance"""
    # Shortest path has overlapping subproblem structure
    # Generate the path source->target by walking backward
    path = [target]
    current = target
    length = distances[source][target]
    while True:
        predecessor = predecessors[source][current]
        path.append(predecessor)
        if predecessor == source:
            return path, length

        current = predecessor


def map_path(path, reverse_mapping):
    return [reverse_mapping[u] for u in path]


def compute_shortest_paths_floyd_warshall(g):
    predecessors, distances = nx.floyd_warshall_predecessor_and_distance(g, weight='edge_weight')
    return predecessors, distances


def iter_shortest_paths(predecessors, distances, mapping, reverse_mapping, original_nodes):
    for u in original_nodes:
        path, length = get_path_from_floyd(predecessors, distances, u, mapping[u])
        path = map_path(path, reverse_mapping)
        # First and last element of path is u
        yield path[:-1], length


# Debugging

def gen_annulus(xpos, n, radius, randomness=True):
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]

    # Random jitter: uniformly distributed, max 1/4 of distance btw 2 pts
    if randomness:
        jitter = np.random.rand(n) * 2 - 1
        jitter *= (2 * np.pi) / (n - 1) / 4
        angles += jitter

    y = radius * np.cos(angles)
    z = radius * np.sin(angles)
    x = np.full_like(y, xpos)

    return np.stack((x, y, z)).T


def gen_init_annulus(n, radius):
    coord = gen_annulus(xpos=0.0, n=n, radius=radius, randomness=True)

    g = nx.Graph()
    for u in range(n):
        g.add_node(u, coord=coord[u, :])

    # Connect
    for u in range(1, n):
        g.add_edge(u, u - 1)
    g.add_edge(0, n - 1)

    annulus = list(range(n))
    return g, annulus


def rotation_matrix_yz(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])


def gen_next_annulus(g, prev_annulus, dx, delta_angle):
    positions = nx.get_node_attributes(g, 'coord')
    R = rotation_matrix_yz(delta_angle)

    # Add x-pos jitter
    dx += (np.random.rand() * 2 - 1) * dx / 2

    dpos = np.array([dx, 0.0, 0.0])
    new_pos = {
        u: R @ positions[u] + dpos
        for u in prev_annulus
    }
    max_node = max(g.nodes)
    new_nodes = {
        u: max_node + i
        for i, u in enumerate(prev_annulus, 1)
    }
    add_nodes = {
        new_nodes[u]: new_pos[u]
        for u in prev_annulus
    }
    for i, u in enumerate(prev_annulus):
        g.add_node(new_nodes[u], coord=new_pos[u])
        # connect to form triangle
        node_prev_annulus = prev_annulus[(i + 1) % len(prev_annulus)]
        g.add_edge(new_nodes[u], node_prev_annulus)
        g.add_edge(new_nodes[u], u)
    # Connect nodes within new annulus
    new_annulus = list(add_nodes.keys())
    for u, v in zip(new_annulus[1:], new_annulus[:-1]):
        g.add_edge(u, v)
    g.add_edge(new_annulus[0], new_annulus[-1])

    return g, new_annulus


def gen_tube(n, radius, n_slice, dx=0.1, delta_angle=np.pi / 8):
    g, annulus = gen_init_annulus(n, radius)
    prev_annulus = annulus
    for _ in range(n_slice - 1):
        g, prev_annulus = gen_next_annulus(g, prev_annulus, dx, delta_angle)

    return g

