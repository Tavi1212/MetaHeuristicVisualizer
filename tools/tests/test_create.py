# test_create_stn.py

import unittest
import pandas as pd
import networkx as nx
from io import StringIO
from scripts.create import create_stn

class TestCreateSTN(unittest.TestCase):

    def setUp(self):
        self.sample_data = StringIO(
            "Run Solution1 Solution2 Fitness1 Fitness2\n"
            "1 A B 10 9\n"
            "1 B C 9 8\n"
            "2 D E 12 11\n"
            "2 E F 11 10\n"
            "3 A C 10 8\n"
        )

    def test_graph_creation(self):
        graph = create_stn(self.sample_data, nr_of_runs=-1, best_solution="C", objective_type="minimization")

        # Basic graph checks
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertTrue(graph.has_node("A"))
        self.assertTrue(graph.has_edge("A", "B"))
        self.assertEqual(graph.nodes["C"]["type"], "best")
        self.assertEqual(graph.nodes["A"]["count"], 2)

    def test_edge_weights(self):
        graph = create_stn(self.sample_data, nr_of_runs=-1, best_solution=None, objective_type="minimization")
        weight = graph["A"]["B"]["weight"]
        self.assertEqual(weight, 1)

    def test_run_filtering(self):
        graph = create_stn(self.sample_data, nr_of_runs=1, best_solution=None, objective_type="minimization")
        self.assertIn("A", graph.nodes)
        self.assertIn("C", graph.nodes)
        self.assertNotIn("D", graph.nodes)
        self.assertNotIn("E", graph.nodes)

    def test_best_solution_insertion(self):
        graph = create_stn(self.sample_data, nr_of_runs=-1, best_solution="Z", objective_type="minimization")
        self.assertIn("Z", graph.nodes)
        self.assertEqual(graph.nodes["Z"]["type"], "best")

    def test_fitness_assignment(self):
        graph = create_stn(self.sample_data, nr_of_runs=-1, best_solution=None, objective_type="minimization")
        self.assertAlmostEqual(graph.nodes["C"]["fitness"], 8.0)

if __name__ == "__main__":
    unittest.main()
