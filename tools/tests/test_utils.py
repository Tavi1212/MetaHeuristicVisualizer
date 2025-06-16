import pytest
import math
import networkx as nx
from scripts import utils# Adjust if your file is in a subfolder

def test_normalize():
    assert utils.normalize(5, 0, 10, 0, 100) == 50
    assert utils.normalize(0, 0, 10, 0, 100) == 0
    assert utils.normalize(10, 0, 10, 0, 100) == 100
    assert utils.normalize(10, 10, 10, 0, 100) == 50

def test_adjust_color_lightness():
    assert utils.adjust_color_lightness("#ff0000", 50).startswith("#")

def test_shannon_entropy_vector():
    assert math.isclose(utils.shannon_entropy_vector([1, 1, 2, 2]), 1.0)

def test_remove_node_between_two():
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=1)
    G.add_edge("b", "c", weight=3)
    utils.remove_node_between_two(G, "b")
    assert G.has_edge("a", "c")
    assert not G.has_node("b")

def test_parse_vectors_string_modes():
    assert utils.parse_vectors_string("00000001", "hex") == [[1]]
    assert utils.parse_vectors_string("101", "binary") == [[1, 0, 1]]
    assert utils.parse_vectors_string("[1.0, 2.0]", "float") == [[1.0, 2.0]]
    assert utils.parse_vectors_string("abc", "categorical") == [["a", "b", "c"]]

def test_parse_discrete_solutions():
    assert utils.parse_discrete_solutions("101", "binary") == [[1, 0, 1]]
    assert utils.parse_discrete_solutions("abc", "categorical") == [["a", "b", "c"]]
    assert utils.parse_discrete_solutions("[1,2,3]", "permutation") == [[1, 2, 3]]

def test_parse_continuous_solution():
    assert utils.parse_continuous_solution("[1.0, 2.0]") == [1.0, 2.0]

def test_distance_functions():
    assert utils.hamming_distance([1, 0], [1, 1]) == 1
    assert math.isclose(utils.euclidean_distance([0, 0], [3, 4]), 5)
    assert utils.manhattan_distance([1, 2], [3, 4]) == 4

def test_compute_distance():
    assert utils.compute_distance([1, 2], [1, 3], utils.manhattan_distance) == 1

def test_is_entropy_applicable():
    assert utils.is_entropy_applicable("binary")
    assert not utils.is_entropy_applicable("real")

def test_merge_graphs_with_count():
    G1 = nx.DiGraph()
    G1.add_node("x", count=1)
    G2 = nx.DiGraph()
    G2.add_node("x", count=2)
    merged = utils.merge_graphs_with_count([G1, G2])
    assert merged.nodes["x"]["count"] == 3

def test_assign_cluster_levels_simple():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    clusters = [["a"], ["b"]]
    levels = utils.assign_cluster_levels(G, clusters)
    assert levels[1] == 1  # second cluster depends on first

# Add more edge-case tests as needed
