"""

@author: Nana Opoku Baah
@email: nana.o.baah@gmail.com

"""

from path_discovery import *


def setUp():
    n1, n2, n3, n4 = Node(1), Node(2), Node(3), Node(4)
    demo_graph = UndirectedGraph(
        [
            UndirectedEdge((n1, n2), 10),
            UndirectedEdge((n1, n3), 30),
            UndirectedEdge((n2, n4), 10),
            UndirectedEdge((n3, n4), 10),
        ]
    )
    return demo_graph, n1, n2, n3, n4


def test_compute_shortest_paths():
    demo_graph, n1, n2, n3, n4 = setUp()
    assert compute_shortest_paths(demo_graph, n1, n4, 1.0) == [[1, 2, 4]]


def test_extend_route():
    demo_graph, n1, n2, n3, n4 = setUp()
    all_routes = [(20, [1, 2, 4]), (30, [1, 2, 4, 3]), (40, [1, 3, 4]), (50, [1, 3, 4, 2])]
    len_tolerance_factor = 1.0

    routes = extend_route(all_routes, demo_graph, len_tolerance_factor)
    assert routes == [(20, [1, 2, 4])]


def test_additional_extend_route_with_higher_factor():
    demo_graph, n1, n2, n3, n4 = setUp()
    all_routes = [(20, [1, 2, 4]), (30, [1, 2, 4, 3]), (40, [1, 3, 4]), (50, [1, 3, 4, 2])]
    len_tolerance_factor = 2.0

    routes = extend_route(all_routes, demo_graph, len_tolerance_factor)
    assert routes == [(20, [1, 2, 4]), (30, [1, 2, 4, 3]), (40, [1, 3, 4]), (40, [1, 2, 1, 2, 4]),
                      (40, [1, 2, 4, 2, 4])]


def test_recompute_route_length():
    demo_graph, n1, n2, n3, n4 = setUp()

    update_route = [1, 2, 4]
    assert recompute_route_length(update_route, demo_graph) == 20


def test_shortest_path():
    demo_graph, n1, n2, n3, n4 = setUp()

    assert compute_shortest_paths(demo_graph, n1, n4, 1.0) == [[1, 2, 4]]

    assert compute_shortest_paths(demo_graph, n1, n4, 2.0) == [[1, 2, 4],
                                                               [1, 2, 4, 3],
                                                               [1, 3, 4],
                                                               [1, 2, 1, 2, 4],
                                                               [1, 2, 4, 2, 4]]


if __name__ == '__main__':
    test_compute_shortest_paths()
