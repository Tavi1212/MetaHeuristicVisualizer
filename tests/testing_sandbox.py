from scripts.create import create_stn
from scripts.partition import cont_standard_partitioning
from scripts.structures import ConfigData
from pyvis.network import Network

input_path = "../input/stn_input1.txt"

config = ConfigData(
    problemType="continuous",
    objectiveType="minimization",
    partitionStrategy="standard",
    cMinBound=50,
    cMaxBound=55,
    cDimension=2,
    cHypercube=-2
)

G = create_stn(input_path)

# Apply standard partitioning
H, stats = cont_standard_partitioning(
    G,
    hypercube_factor=config.cHypercube,
    min_bound=config.cMinBound,
    max_bound=config.cMaxBound
)

# Print stats
print("Partitioning Stats:")
for k, v in stats.items():
    print(f"{k}: {v}")
if stats["original_edges"] > 0:
    print(f"→ Edge retention: {100 * stats['kept_edges'] / stats['original_edges']:.2f}%")

# Visualize
def visualize_graph_pyvis(G, output_path="partitioned_graph.html"):
    net = Network(height="800px", width="100%", directed=True)
    for node, data in G.nodes(data=True):
        label = f"{node}\ncount: {data.get('count', '')}"
        net.add_node(node, label=label, size=20)
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, title=str(data.get("weight", 1)))
    net.show(output_path, notebook=False)

visualize_graph_pyvis(H)
