import matplotlib.pyplot as plt
import networkx as nx
from scripts.create import create_stn

G = create_stn("../input/DE_Rana_3D.txt")

pos = nx.spring_layout(G, seed=42)

# Noduri după tip
start_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "start"]
end_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "end"]
inter_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "intermediate"]

# Desenăm nodurile
nx.draw_networkx_nodes(G, pos, nodelist=start_nodes, node_color="yellow", label="Start", node_size=200)
nx.draw_networkx_nodes(G, pos, nodelist=end_nodes, node_color="red", label="End", node_size=200)
nx.draw_networkx_nodes(G, pos, nodelist=inter_nodes, node_color="skyblue", label="Intermediate", node_size=100)

# Desenăm muchiile
nx.draw_networkx_edges(G, pos, alpha=0.5)

# Etichete doar pentru primele câteva noduri
label_subset = {n: f"{G.nodes[n]['count']:.0f}" for i, n in enumerate(G.nodes) if i < 10}
nx.draw_networkx_labels(G, pos, labels=label_subset, font_size=6)

# Legenda + titlu
plt.legend()
plt.title("Search Trajectory Network (STN)")
plt.axis("off")
plt.tight_layout()
plt.show()