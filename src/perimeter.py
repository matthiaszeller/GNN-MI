

import json
import logging
from collections import Counter
from copy import deepcopy

import networkx as nx
import numpy as np
import torch
from scipy.spatial import distance_matrix
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_scipy_sparse_matrix


def compute_perimeters(data, save_path=None):
    a1, a2 = find_edge_annuluses(data.edge_index, data.coord)
    path = find_path(data, a1, a2)
    g, old, new, mapping, reverse_mapping = cut_duplicate_join(data, path)
    res = get_shortest_paths(g, mapping, reverse_mapping, old)

    if save_path is not None:
        dump = {
            'annulus_1': list(map(int, a1)),
            'annulus_2': list(map(int, a2)),
            'cutting_path': list(map(int, path)),
            'shortest_paths': res
        }
        logging.info(f'saving perimeter data in {str(save_path)}')
        with open(save_path, 'w') as f:
            json.dump(dump, f, indent=4)


# For the floyd-warshall algo, super inefficient
# def compute_perimeters(data):
#     a1, a2 = find_edge_annuluses(data.edge_index)
#     path = find_path(data, a1, a2)
#
#     g, old, new, mapping, reverse_mapping = cut_duplicate_join(data, path)
#     pred, dist = compute_shortest_paths_floyd_warshall(g)
#
#     perimeters = {
#         path[0][0]: path[1]
#         for path in iter_shortest_paths(pred, dist, mapping, reverse_mapping, old)
#     }
#     return perimeters


def get_neighbours(nnz_r, nnz_c, i):
    """nnz_r, nnz_c can be: edge_index of torch data or scipy.sparse.spmatrix.nonzero()"""
    mask = nnz_r == i
    return nnz_c[mask]


def find_edge_annulus(edge_index, start_node, coords, maxiter=1000):
    """Starting from a node at the edge of the mesh, return all nodes of the annulus
    The search is based on minimum-degree criterion."""
    def angle(u, v):
        """Angle between two vectors"""
        cos = u.dot(v) / np.linalg.norm(u) / np.linalg.norm(v)
        angle = np.arccos(cos) * 360 / (2 * np.pi)
        return angle

    def select_next_node(n_neighs, current_node, previous_node=None):
        """For some pathological cases, some nodes at the border have high degree (5 instead of 4)"""
        counts = Counter(e[1] for e in n_neighs)
        min_degree = min(counts)
        # Previous node is None if we are starting the algorithm... then just choose a min degree node
        if counts[min_degree] == 1 or previous_node is None:
            return min(n_neighs, key=lambda e: e[1])[0]

        # Otherwise need to break tie: we have a current direction, pick node whose direction has the smallest angle
        # with the current direction
        current_direction = coords[current_node] - coords[previous_node]
        candidates = [e[0] for e in n_neighs if e[1] == min_degree]
        candidate_directions = [
            (node, angle(coords[node] - coords[current_node], current_direction))
            for node in candidates
        ]
        return min(candidate_directions, key=lambda e: e[1])[0]


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
        next_node = select_next_node(n_neighs, current, previous_node=nodes[-2] if hop_counter > 1 else None)

        # Add selected node
        annulus.add(next_node)
        nodes.append(next_node)
        hop_counter += 1
        current = next_node

    return nodes


def find_edge_annuluses(edge_index, coords):
    """Find the two annuluses"""
    # Find nodes with minimal number of neighbours
    A = to_scipy_sparse_matrix(edge_index)
    degree = np.array(A.sum(axis=0)).flatten()
    sorted_degree = np.argsort(degree)

    annulus1 = find_edge_annulus(edge_index, sorted_degree[0], coords)
    # Remove all nodes from annulus 1
    mask = np.isin(sorted_degree, annulus1)
    sorted_degree = sorted_degree[~mask]

    annulus2 = find_edge_annulus(edge_index, sorted_degree[0], coords)

    return annulus1, annulus2


def find_path(data, a1, a2):
    """Return set of nodes connecting nodes in a1 with nodes in a2"""
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
    """Given a path cutting the graph in two parts, find its two parallel paths"""
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
    """Compute edge weights based on Euclidean distances."""
    out = data.clone()
    D = distance_matrix(out.coord, out.coord)
    i1, i2 = out.edge_index
    out.edge_weight = D[i1, i2]
    return out


def cut_and_duplicate(weighted_graph, path):
    """Given a path cutting the weighted graph in two, remove the nodes in the path and duplicate the graph"""
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
    """Merge the original and the duplicated graph, add back nodes from the path and connect it accordingly"""
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
    """Build a list of (source, target, dict) for all edges involving the `nodes` in `g`"""
    edges = []
    for u in nodes:
        for v, d in g[u].items():
            edges.append((u, v, d))
    return edges


def cut_duplicate_join(data, path):
    """Prepare the graph for shortest path search"""
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


def get_shortest_paths(g, mapping, reverse_mapping, original_nodes):
    res = []
    for u in original_nodes:
        p = nx.shortest_path(g, source=u, target=mapping[u], weight='edge_weight')
        length = nx.path_weight(g, p, weight='edge_weight')
        p = map_path(p, reverse_mapping)
        # np.int64 -> int for serializability when dumping in json
        p = list(map(int, p))
        res.append((p, length))

    return res


# The code below works but is super inefficient, cubic time w.r.t. num of nodes
# def compute_shortest_paths_floyd_warshall(g):
#     predecessors, distances = nx.floyd_warshall_predecessor_and_distance(g, weight='edge_weight')
#     return predecessors, distances
#
#
# def iter_shortest_paths(predecessors, distances, mapping, reverse_mapping, original_nodes):
#     for u in original_nodes:
#         path, length = get_path_from_floyd(predecessors, distances, u, mapping[u])
#         path = map_path(path, reverse_mapping)
#         # First and last element of path is u
#         yield path[:-1], length


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


if __name__ == '__main__':
    data=torch.load('../data/CHUV03_LAD.pt')
    compute_perimeters(data, 'trash.json')
