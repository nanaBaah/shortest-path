"""A graph path discovery coding task.

@author: Nana Opoku Baah
@email: nana.o.baah@gmail.com

In this file, you are presented with the task to implement the function `compute_shortest_paths`
which discovers some shortest paths from a start node to an end node in an undirected weighted graph
with strictly positive edge lengths. The function is marked with "TODO: Write" below and carries a
more precise specification of the expected behavior in its docstring.

Please write the implementation with the highest quality standards in mind which you would also use
for production code. Functional correctness is the most important criterion. After that it will be
evaluated in terms of maintainability and wall clock runtime for large graphs (in decreasing order
of importance). Please submit everything you have written, including documentation and tests.

Your implementation of `compute_shortest_paths` should target Python 3.9 and not use any external
dependency except for the Python standard library. Outside of the implementation of
`compute_shortest_paths` itself, you are free to use supporting libraries as long as they are
available on PyPi.org. If you use additional packages, please add a requirements.txt file which
lists them with their precise versions ("packageA==1.2.3").
"""

from functools import total_ordering
from typing import Any, List, Optional, Set, Tuple, cast
import heapq


class Node:
    """A node in a graph."""

    def __init__(self, id: int):
        self.id: int = id
        self.adjacent_edges: List["UndirectedEdge"] = []

    def edge_to(self, other: "Node") -> Optional["UndirectedEdge"]:
        """Returns the edge between the current node and the given one (if existing)."""
        matches = [edge for edge in self.adjacent_edges if edge.other_end(self) == other]
        return matches[0] if len(matches) > 0 else None

    def is_adjacent(self, other: "Node") -> bool:
        """Returns whether there is an edge between the current node and the given one."""
        return other in {edge.other_end(self) for edge in self.adjacent_edges}

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.id == other.id

    def __le__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.id <= other.id

    def __hash__(self) -> int:
        return self.id

    def __repr__(self) -> str:
        return f"Node({self.id})"


class UndirectedEdge:
    """An undirected edge in a graph."""

    def __init__(self, end_nodes: Tuple[Node, Node], length: float):
        self.end_nodes: Tuple[Node, Node] = end_nodes
        if 0 < length:
            self.length: float = length
        else:
            raise ValueError(
                f"Edge connecting {end_nodes[0].id} and {end_nodes[1].id}: "
                f"Non-positive length {length} not supported."
            )

        if any(e.other_end(end_nodes[0]) == end_nodes[1] for e in end_nodes[0].adjacent_edges):
            raise ValueError("Duplicate edges are not supported")

        self.end_nodes[0].adjacent_edges.append(self)
        if self.end_nodes[0] != self.end_nodes[1]:
            self.end_nodes[1].adjacent_edges.append(self)
        self.end_node_set = set(self.end_nodes)

    def other_end(self, start: Node) -> Node:
        """Returns the other end of the edge, given one of the end nodes."""
        return self.end_nodes[0] if self.end_nodes[1] == start else self.end_nodes[1]

    def is_adjacent(self, other_edge: "UndirectedEdge") -> bool:
        """Returns whether the current edge shares an end node with the given edge."""
        return len(self.end_node_set.intersection(other_edge.end_node_set)) > 0

    def __repr__(self) -> str:
        return (
            f"UndirectonalEdge(({self.end_nodes[0].__repr__()}, "
            f"{self.end_nodes[1].__repr__()}), {self.length})"
        )


class UndirectedGraph:
    """A simple undirected graph with edges attributed with their length."""

    def __init__(self, edges: List[UndirectedEdge]):
        self.edges: List[UndirectedEdge] = edges
        self.nodes_by_id = {node.id: node for edge in self.edges for node in edge.end_nodes}


@total_ordering
class UndirectedPath:
    """An undirected path through a given graph."""

    def __init__(self, nodes: List[Node]):
        assert all(
            node_1.is_adjacent(node_2) for node_1, node_2 in zip(nodes[:-1], nodes[1:])
        ), "Path edges must be a chain of adjacent nodes"
        self.nodes: List[Node] = nodes
        self.length = sum(
            cast(UndirectedEdge, node_1.edge_to(node_2)).length
            for node_1, node_2 in zip(nodes[:-1], nodes[1:])
        )

    @property
    def start(self) -> Node:
        return self.nodes[0]

    @property
    def end(self) -> Node:
        return self.nodes[-1]

    def prepend(self, edge: UndirectedEdge) -> "UndirectedPath":
        if self.start not in edge.end_nodes:
            raise ValueError("Edge is not adjacent")
        return UndirectedPath([edge.other_end(self.start)] + self.nodes)

    def append(self, edge: UndirectedEdge) -> "UndirectedPath":
        if self.end not in edge.end_nodes:
            raise ValueError("Edge is not adjacent")
        return UndirectedPath(self.nodes + [edge.other_end(self.end)])

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, UndirectedPath) and self.nodes == other.nodes

    def __le__(self, other: Any) -> bool:
        return isinstance(other, UndirectedPath) and self.length <= other.length

    def __hash__(self) -> int:
        return hash(n.id for n in self.nodes)

    def __repr__(self) -> str:
        nodestr: str = ", ".join([node.__repr__() for node in self.nodes])
        return f"UndirectedPath([{nodestr}])"


def compute_shortest_paths(
        graph: UndirectedGraph, start: Node, end: Node, length_tolerance_factor: float
) -> Set[UndirectedPath]:
    """Computes and returns the N shortest paths between the given end nodes.

    The discovered paths always contain the shortest path between the two nodes. In addition, the
    second shortest, third shortest and following paths are also added (in ascending order by path
    length) up to (excluding) the path whose length is larger than the length of the shortest path
    multiplied with the given tolerance factor.

    We do not constrain this function to acyclic paths, i.e., cyclic paths should be found as well.

    For a given input of start node A, end node B and a tolerance factor of 2, the result has to
    contain all paths from A to B whose length is at most twice the length of the shortest path
    from A to B.

    Args:
        graph: The undirected graph in which the N shortest paths shall be found.
        start: The start node of the paths
        end: The end node of the paths
        length_tolerance_factor: The maximum length ratio which is allowed for the discovered paths
            (minimum: 1.0, maximum: infinite)

    Returns:
        The discovered paths. If no path from A to B exists, the result shall be empty.
    """

    adjacent_lengths: dict[float] = {keys: float('inf') for keys in graph.nodes_by_id.keys()}
    adjacent_lengths[start.id] = 0

    visited_nodes: list = []
    node_paths: list[Node] = []
    all_routes: list = []

    def shortest_path(begin: Node,
                      current_node_id: int,
                      destination_node_id: int,
                      visited_node_lists: list,
                      current_node_routes: list[Node],
                      start_to_end_routes: list,
                      adjacent_lens: dict):

        current_node: Node = graph.nodes_by_id[current_node_id]

        visited_node_lists.append(current_node.id)
        current_node_routes.append(current_node)

        if current_node_id == destination_node_id:
            start_to_end_routes.append((adjacent_lens[current_node_id], [_.id for _ in current_node_routes]))

        for edge in current_node.adjacent_edges:
            current_adjacent_node: Node = edge.other_end(current_node)

            if current_adjacent_node.id not in visited_node_lists:
                adjacent_lens[current_adjacent_node.id] = adjacent_lens[current_node.id] + edge.length

                shortest_path(begin,
                              current_adjacent_node.id,
                              destination_node_id,
                              visited_node_lists,
                              current_node_routes,
                              start_to_end_routes,
                              adjacent_lens)

        if (begin in current_node_routes) and (end in current_node_routes):
            extracted_path: list = [_.id for _ in current_node_routes]

            total_len: float = 0

            for idx, value in enumerate(current_node_routes):
                if idx < len(current_node_routes) - 1:
                    connected_pair_nodes = UndirectedPath([value, current_node_routes[idx + 1]]).length
                    total_len += connected_pair_nodes

            possible_route = (total_len, extracted_path)

            if possible_route not in start_to_end_routes:
                start_to_end_routes.append((total_len, extracted_path))

        current_node_routes.pop()
        visited_node_lists.pop()

    shortest_path(start, start.id, end.id, visited_nodes, node_paths, all_routes, adjacent_lengths)

    if not node_paths:
        final_routes = extend_route(all_routes, graph, length_tolerance_factor)
        heapq.heapify(final_routes)

        route_sets = [route_set for (weight, route_set) in final_routes]

        return route_sets


def extend_route(all_routes, graph, length_tolerance_factor):
    final_route: list = []
    all_routes = sorted(all_routes)

    start2end_min_len = all_routes[0][0] * length_tolerance_factor

    for located_path in all_routes:
        if float(located_path[0]) <= start2end_min_len:
            final_route.append(located_path)

    for (computed_len, computed_route) in final_route:
        update_len = 0
        update_route = computed_route

        duplicate_adj_nodes(final_route, graph, start2end_min_len, update_len, update_route, append=True)
        duplicate_adj_nodes(final_route, graph, start2end_min_len, update_len, update_route, append=False)
    return final_route


def duplicate_adj_nodes(full_paths, graph, min_start2end_len, update_len, update_route, append=True):
    while update_len < min_start2end_len:
        if append:
            update_route = update_route[0:2] + update_route
        else:
            update_route = update_route + update_route[-2:]

        update_len = recompute_route_length(update_route, graph)

        if update_len > min_start2end_len:
            break

        if update_route not in full_paths:
            full_paths.append((update_len, update_route))


def recompute_route_length(route, graph):
    update_len: int = 0
    for k, v in enumerate(route):
        if k < len(route) - 1:
            adjacent_node_pair = [graph.nodes_by_id[v], graph.nodes_by_id[route[k + 1]]]
            update_len += UndirectedPath(adjacent_node_pair).length
    return update_len


# Usage example
n1, n2, n3, n4 = Node(1), Node(2), Node(3), Node(4)
demo_graph = UndirectedGraph(
    [
        UndirectedEdge((n1, n2), 10),
        UndirectedEdge((n1, n3), 30),
        UndirectedEdge((n2, n4), 10),
        UndirectedEdge((n3, n4), 10),
    ]
)

# Should print the path [1, 2, 4]
print(compute_shortest_paths(demo_graph, n1, n4, 1.0))

# Should print the paths [1, 2, 4], [1, 3, 4], [1, 2, 4, 2, 4], [1, 2, 1, 2, 4], [1, 2, 4, 3, 4]
print(compute_shortest_paths(demo_graph, n1, n4, 2.0))
