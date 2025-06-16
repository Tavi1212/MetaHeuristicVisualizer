import json
import networkx as nx
import scripts.utils as utils
from collections import Counter
from collections import defaultdict
from scripts.structures import ConfigData, AdvancedOptions
from scripts.utils import normalize, adjust_color_lightness
from pyvis.network import Network
import matplotlib.pyplot as plt
from scripts.utils import adjust_color_lightness, normalize


# Takes a graph and colors it based on node types:
# Start:        Yellow
# End:          Grey
# Intermediate: Given as argument
# Best:         Golden
def tag_graph_origin(G, origin_color):
    for node in G.nodes:
        node_type = G.nodes[node].get("type", "intermediate")

        match node_type:
            case "start":
                G.nodes[node]["origin_color"] = "#f5e642"  # yellow
                G.nodes[node]["shape"] = "dot"
            case "end":
                G.nodes[node]["origin_color"] = "#575757"  # grey
                G.nodes[node]["shape"] = "diamond"
            case "best":
                G.nodes[node]["origin_color"] = "#f5b111"  # golden
                G.nodes[node]["shape"] = "star"
            case _:
                G.nodes[node]["origin_color"] = origin_color  # custom
                G.nodes[node]["shape"] = "dot"


# Adds a legend box to the HTML graph showing color mappings and optionally a "merged nodes" entry
def add_legend(html_file, color_name_map, include_gray):
    # Static entries for known types
    static_legend = {
        "#f5e642": ("Start Node", "square"),
        "#575757": ("End Node", "diamond")
    }

    if include_gray:
        static_legend["#787878"] = ("Merged Node (Multiple Origins)", "dot")

    # Dynamic entries from user algorithms — assume intermediate/dot
    dynamic_legend = {
        color: (name, "dot") for color, name in color_name_map.items()
    }

    # Combine everything
    combined_legend = {**static_legend, **dynamic_legend}

    # Build HTML for each legend item
    legend_items = ""
    for color, (label, shape) in combined_legend.items():
        # Determine the visual for each shape
        base_style = f"""
        background:{color}; background-color:{color}; 
        border:1px solid #000; 
        margin-right:6px;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
        """

        if shape == "dot":
            style = f"{base_style} width:12px; height:12px; border-radius:50%;"
        elif shape == "square":
            style = f"{base_style} width:12px; height:12px; border-radius:0%;"
        elif shape == "diamond":
            style = f"{base_style} width:12px; height:12px; transform: rotate(45deg); border-radius:0%;"
        else:
            style = f"{base_style} width:12px; height:12px;"  # fallback

        legend_items += f"""
        <div style="display:flex; align-items:center; margin-bottom:4px;">
            <div style="{style}; display:inline-block;"></div>
            <span style="margin-left:6px;">{label}</span>
        </div>
        """

    # Inject legend into the HTML file
    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()

    legend_html = f"""
    <div style="position: absolute; top: 20px; right: 20px; background: white; padding: 10px; 
                border: 1px solid #ccc; font-family: sans-serif; font-size: 13px; z-index:999;">
        <b>Legend</b><br>
        {legend_items}
        <hr style="margin:6px 0;">
        <div style="font-size: 11px;">Size ∝ Visit Frequency</div>
        <div style="font-size: 11px;">Shade ∝ Fitness</div>
    </div>
    """

    html = html.replace("</body>", legend_html + "\n</body>")

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)


# Adds layout and graph statistics (nodes, edges, components) to the HTML visualization
def add_info(html_file, layout, num_nodes, num_edges, num_components):
    # Build the HTML snippet with graph statistics
    stats_html = f"""
    <div style="position: absolute; bottom: 20px; right: 20px; 
                font-family: sans-serif; font-size: 13px; color: #444;
                background: rgba(255,255,255,0.85); padding: 6px 10px;">
        <div><b>Layout:</b> {layout.upper()}</div>
        <div><b>Nodes:</b> {num_nodes}</div>
        <div><b>Edges:</b> {num_edges}</div>
        <div><b>Components:</b> {num_components}</div>
    </div>
    """

    # Read the existing HTML content
    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()

    # Inject the stats before the closing </body> tag
    html = html.replace("</body>", stats_html + "\n</body>")

    # Write the updated content back to the file
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)


# Applies a hierarchical tree layout to the graph visualization
# 'direction' can be "UD" (top-down), "LR" (left-right), etc.
def apply_tree_layout(net: Network, direction: str = "UD"):
    layout_options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": direction,
                "sortMethod": "hubsize",
                "nodeSpacing": 200,
                "treeSpacing": 300,
                "levelSeparation": 100
            }
        },
        "edges": {
            "smooth": True  # Make edges appear smooth
        }
    }

    # Apply layout settings to the network
    net.set_options(json.dumps(layout_options))


# Takes one or more graph and creates a Search Trajectory Network
# Result will take the form of a .html page containing the STN visualization
def visualize_stn(graphs: list, advanced, config, output_file="stn_graph.html", minmax="minimization",
                  legend_entries=None, layout="fr"):
    merged = nx.DiGraph()
    node_origins = {}

    # Merge the graphs, keeping track of each node's origin algorithm, on basis of it's orgin color
    for G in graphs:
        merged = utils.merge_graphs_with_count(graphs)
        for node in G.nodes:
            origin = G.nodes[node].get("origin_color", "#1f77b4")
            if node not in node_origins:
                node_origins[node] = set()
            node_origins[node].add(origin)

    net = Network(height="100%", width="100%", directed=True)

    if not advanced.tree_layout:
        if layout == "fr":
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 75,
                  "springConstant": 0.5,
                  "damping": 0.7,
                  "avoidOverlap": 0
                },
                "minVelocity": 0.01,
                "timestep": 0.35,
                "stabilization": {
                  "enabled": true,
                  "iterations": 1000,
                  "updateInterval": 25,
                  "fit": true
                }
              }
            }
            """)

        elif layout == "kk":
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "solver": "barnesHut",
                "barnesHut": {
                  "gravitationalConstant": -15000,
                  "centralGravity": 0.5,
                  "springLength": 75,
                  "springConstant": 0.5,
                  "damping": 0.09,
                  "avoidOverlap": 0
                },
                "minVelocity": 0.01,
                "timestep": 0.35,
                "stabilization": {
                  "enabled": true,
                  "iterations": 1000,
                  "updateInterval": 25,
                  "fit": true
                }
              }
            }
            """)

    shape_map = {
        "start": "square",
        "end": "diamond",
        "intermediate": "dot",
        "best": "star"
    }

    fixed_colors = {
        "start": "#f5e642",
        "end": "#f54242",
        "best": "#f5d000"
    }

    # Extract the counts and fitnesses for each node in the merged STN
    counts = [attrs.get("count", 1) for _, attrs in merged.nodes(data=True) if attrs.get("type") == "intermediate"]
    fitnesses = [attrs.get("fitness", 0) for _, attrs in merged.nodes(data=True) if attrs.get("type") == "intermediate"]

    # Extract the minimum and maximum node counts and fitnesses for determining node size and shade
    min_count, max_count = min(counts, default=1), max(counts, default=1)
    min_fit, max_fit = min(fitnesses, default=0), max(fitnesses, default=1)

    # Compute the node size range
    raw_min_size = 25
    raw_max_size = 60
    vertex_scale = advanced.vertex_size
    max_node_size = 70

    has_merged_nodes = False
    for node, attrs in merged.nodes(data=True):
        node_type = attrs.get("type", "intermediate")
        shape = shape_map.get(node_type, "dot")
        fitness = attrs.get("fitness", 0)
        count = attrs.get("count", 1)
        fitness_str = f"{fitness:.4f}" if fitness is not None else "N/A"
        title = f"ID: {node}\nFitness: {fitness_str}\nCount: {count}"

        scaled_size = normalize(count, min_count, max_count, raw_min_size * vertex_scale, raw_max_size * vertex_scale)
        size = min(scaled_size, max_node_size)

        if node_type in fixed_colors:
            color = attrs.get("origin_color", fixed_colors[node_type])
        else:
            base_color = list(node_origins[node])[0]
            if len(node_origins[node]) > 1:
                color = "#787878"
                has_merged_nodes = True
            else:
                lightness = normalize(fitness, min_fit, max_fit, 30, 90) if minmax == "minimization" else normalize(
                    fitness, max_fit, min_fit, 30, 90)
                color = adjust_color_lightness(base_color, lightness)

        net.add_node(node, label=" ", title=title, color=color, shape=shape, size=size)

    for u, v, attrs in merged.edges(data=True):
        net.add_edge(u, v, color=attrs.get("color", "black"), value=15)

    if advanced.tree_layout:
        apply_tree_layout(net, direction="UD")

    net.save_graph(output_file)

    if legend_entries:
        add_legend(output_file, legend_entries, include_gray=has_merged_nodes)

    num_components = nx.number_weakly_connected_components(merged)
    add_info(output_file, layout, len(merged.nodes), len(merged.edges), num_components)


def visualize_clusters(final_clusters, G, output_file="stn_graph.html", legend_entries=None,
                       objective_type="minimization", advanced=None, layout="fr"):
    net = Network(height="100%", width="100%", notebook=False)

    shape_map = {
        "start": "square",
        "end": "diamond",
        "intermediate": "dot",
        "best": "star"
    }

    fixed_colors = {
        "start": "#f5e642",
        "end": "#f54242",
        "best": "#f5b111"
    }

    # Normalize cluster sizes to [20, 80]
    raw_sizes = [len(c) for c in final_clusters]
    min_raw, max_raw = min(raw_sizes), max(raw_sizes)

    total_clusters = len(final_clusters)
    total_nodes = len(G.nodes)
    max_cluster_size = total_nodes

    def normalize_size(raw):
        return 20 + (raw / max_cluster_size) * 60

    # Global fitness range
    fitness_dict = nx.get_node_attributes(G, "fitness")
    global_min_fitness = min(fitness_dict.values())
    global_max_fitness = max(fitness_dict.values())
    global_range = global_max_fitness - global_min_fitness or 1e-9

    cluster_levels = {}
    if advanced and advanced.tree_layout:
        cluster_levels = utils.assign_cluster_levels(G, final_clusters)

    cluster_nodes = []
    for i, cluster in enumerate(final_clusters):
        size = normalize_size(len(cluster))

        type_counts = defaultdict(int)
        color_counts = []

        for n in cluster:
            node_type = G.nodes[n].get("type", "intermediate")
            type_counts[node_type] += 1
            color = G.nodes[n].get("origin_color", "#787878")
            color_counts.append(color)

        dominant_type = max(type_counts, key=type_counts.get)
        if any(G.nodes[n].get("type") == "best" for n in cluster):
            shape = "star"
        else:
            shape = shape_map[dominant_type]

        fitness_values = [G.nodes[n]["fitness"] for n in cluster if "fitness" in G.nodes[n]]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)

        # Determine color
        if dominant_type in fixed_colors:
            dominant_color = fixed_colors[dominant_type]
        else:
            base_color = Counter(color_counts).most_common(1)[0][0]

            if dominant_type == "intermediate":
                if global_range == 0:
                    lightness = 60  # default mid-lightness if no variation
                else:
                    if objective_type == "minimization":
                        lightness = normalize(max_fitness, global_min_fitness, global_max_fitness, 30, 90)
                    else:
                        lightness = normalize(max_fitness, global_min_fitness, global_max_fitness, 90, 30)
                dominant_color = adjust_color_lightness(base_color, lightness)
            else:
                dominant_color = base_color

        volume = (max_fitness - min_fitness) / global_range * 100
        percentage = (len(cluster) / total_nodes) * 100

        label = f"{len(cluster)} [{min_fitness:.1f}-{max_fitness:.1f}]"
        title = (
            f"Cluster {i + 1}\n"
            f"Size: {len(cluster)} ({percentage:.2f}%)\n"
            f"Min Fitness: {min_fitness:.4g}\n"
            f"Max Fitness: {max_fitness:.4g}\n"
            f"Volume: {volume:.2f}%"
        )

        net.add_node(
            i,
            label=label,
            size=size,
            color=dominant_color,
            title=title,
            shape=shape,
            level=cluster_levels.get(i, 0) if advanced and advanced.tree_layout else None
        )
        cluster_nodes.append((i, set(cluster)))

    # Draw edges if any connection between cluster members
    for i, (id_a, nodes_a) in enumerate(cluster_nodes):
        for j, (id_b, nodes_b) in enumerate(cluster_nodes):
            if i >= j:
                continue
            if any(G.has_edge(u, v) or G.has_edge(v, u) for u in nodes_a for v in nodes_b):
                net.add_edge(id_a, id_b, color="black")

    net.save_graph(output_file)

    if legend_entries:
        has_merged_nodes = any(
            G.nodes[n].get("origin_color", "") == "#787878"
            for cluster in final_clusters for n in cluster
        )
        add_legend(output_file, legend_entries, include_gray=has_merged_nodes)

    num_components = nx.number_weakly_connected_components(G)
