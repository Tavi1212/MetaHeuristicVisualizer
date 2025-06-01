import pytest
import networkx as nx
from scripts.create import create_stn
from scripts.partition import apply_partitioning_from_config
from scripts.structures import ConfigData

@pytest.fixture
def test_discrete_graph():
    G = create_stn("input/BRKGA.txt")  # Must exist
    return G

def test_discrete_shannon_partitioning(test_discrete_graph):
    initial_nodes = len(test_discrete_graph.nodes)
    config = ConfigData(
        problemType="discrete",
        objectiveType="minimization",
        partitionStrategy="shannon",
        dPartitioning=50,
        dCSize=0, dVSize=0,
        dDistance="hamming",
        dMinBound=0,
        cMaxBound=0,
        cDimension=0,
        cHypercube=0,
        cClusterSize=0,
        cVolumeSize=0,
        cDistance="euclidean"
    )

    G = apply_partitioning_from_config(test_discrete_graph, config)
    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes) < initial_nodes  # Expect node removal
    print(f"Nodes reduced from {initial_nodes} to {len(G.nodes)}")

