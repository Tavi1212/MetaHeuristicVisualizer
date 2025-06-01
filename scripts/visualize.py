import networkx as nx
from scripts.utils import normalize, adjust_color_lightness
from pyvis.network import Network


# Takes a graph and colors it based on node types:
# Start:        Yellow
# End:          Red
# Intermediate: Given as argument
def tag_graph_origin(G, origin_color):
    for node in G.nodes:
        match G.nodes[node].get("type", "intermediate"):
            case "start":
                G.nodes[node]["origin_color"] = "#f5e642"  # Yellow
            case "end":
                G.nodes[node]["origin_color"] = "#f54242"  # Red
            case _:
                G.nodes[node]["origin_color"] = origin_color


def shape_style(shape):
    if shape == "dot":
        return "border-radius: 50%; width: 12px; height: 12px;"
    elif shape == "square":
        return "border-radius: 0%; width: 12px; height: 12px;"
    elif shape == "diamond":
        return "width: 12px; height: 12px; transform: rotate(45deg); border-radius: 0%;"
    return "width: 12px; height: 12px;"


def add_legend(html_file, color_name_map, include_gray):
    # Static entries for known types
    static_legend = {
        "#f5e642": ("Start Node", "square"),
        "#f54242": ("End Node", "diamond")
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
        base_style = f"background:{color}; border:1px solid #000; margin-right:6px;"

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
            <div style="{style}"></div>
            <span>{label}</span>
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


# Takes one or more graph and creates a Search Trajectory Network
# Result will take the form of a .html page containing the STN visualization
def visualize_stn(graphs: list, output_file="stn_graph.html", minmax="minimization", legend_entries=None):
    merged = nx.DiGraph()
    node_origins = {}

    for G in graphs:
        merged = nx.compose(merged, G)
        for node in G.nodes:
            origin = G.nodes[node].get("origin_color", "#1f77b4")
            if node not in node_origins:
                node_origins[node] = set()
            node_origins[node].add(origin)

    net = Network(height="600px", width="100%", directed=True)

    shape_map = {
        "start": "square",
        "end": "diamond",
        "intermediate": "dot"
    }

    fixed_colors = {
        "start": "#f5e642",  # Yellow
        "end": "#f54242"  # Red
    }

    # Normalize by intermediate nodes only
    counts = [attrs.get("count", 1) for _, attrs in merged.nodes(data=True) if attrs.get("type") == "intermediate"]
    fitnesses = [attrs.get("fitness", 0) for _, attrs in merged.nodes(data=True) if attrs.get("type") == "intermediate"]
    min_count, max_count = min(counts, default=1), max(counts, default=1)
    min_fit, max_fit = min(fitnesses, default=0), max(fitnesses, default=1)

    has_merged_nodes = False
    for node, attrs in merged.nodes(data=True):
        node_type = attrs.get("type", "intermediate")
        shape = shape_map.get(node_type, "dot")
        fitness = attrs.get("fitness", 0)
        title = f"ID: {node}\nFitness: {fitness:.4f}"

        # Determine node color
        if node_type == "start":
            color = fixed_colors[node_type]
            size = 14  # smaller for clarity
        elif node_type == "end":
            color = fixed_colors[node_type]
            size = 14  # smallest, to make end nodes less dominant
        else:
            count = attrs.get("count", 1)
            size = normalize(count, min_count, max_count, 20, 60)  # raised base size

            if minmax == "minimization":
                lightness = normalize(fitness, max_fit, min_fit, 30, 90)
            else:
                lightness = normalize(fitness, min_fit, max_fit, 30, 90)

            base_color = list(node_origins[node])[0]

            if len(node_origins[node]) > 1:
                color = "#787878"
                has_merged_nodes = True
            else:
                color = adjust_color_lightness(base_color, lightness)

            if minmax == "minimization":
                lightness = normalize(fitness, max_fit, min_fit, 30, 90)
            else:
                lightness = normalize(fitness, min_fit, max_fit, 30, 90)

            base_color = list(node_origins[node])[0]

            if len(node_origins[node]) > 1:
                color = "#787878"
                has_merged_nodes = True
            else:
                color = adjust_color_lightness(base_color, lightness)

        net.add_node(
            node,
            label=" ",
            title=title,
            color=color,
            shape=shape,
            size=size
        )

    for u, v, attrs in merged.edges(data=True):
        net.add_edge(
            u,
            v,
            color=attrs.get("color", "black"),
            value=attrs.get("weight", 1)
        )

    net.save_graph(output_file)
    if legend_entries:
        add_legend(output_file, legend_entries, include_gray=has_merged_nodes)

