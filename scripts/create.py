from turtledemo.penrose import start
from venv import create

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.classes import DiGraph

def createSTN(infile):
    df = pd.read_table(infile, sep=r"\s+")
    nruns = df["Run"].iloc[-1]

    nodes  = pd.DataFrame()
    edges = pd.DataFrame()

    run_changes = df["Run"].diff() != 0

    start_ids = df.loc[run_changes, "Solution1"].tolist()
    end_ids = df["Solution2"].shift(1).loc[run_changes].tolist()

    end_ids.pop(0)
    end_ids.append(df["Solution2"].iloc[-1])


    for run_id in range(0, nruns):
        temp_df = df[df["Run"] == run_id + 1]
        temp_df = temp_df.drop(["Run"], axis = 1)
        temp_df = temp_df.rename(
                                columns = {
                                    "Fitness1": "fit1",
                                    "Solution1": "node1",
                                    "Fitness2": "fit2",
                                    "Solution2": "node2"
                                })
        f2r = temp_df[["fit1", "node1"]]
        s2r = temp_df[["fit2", "node2"]]
        f2r = f2r.rename(columns = {"fit1": "fit", "node1": "node"})
        s2r = s2r.rename(columns = {"fit2": "fit", "node2": "node"})
        nodes = pd.concat([nodes, f2r, s2r])

        temp_df = temp_df.drop(["fit1", "fit2"], axis = 1)
        edges = pd.concat([edges, temp_df])

    nodes["count"] = nodes.groupby("node")["node"].transform("size")
    nodes = nodes.drop_duplicates(subset="node", keep='first').reset_index(drop=True)

    edges["weight"] = edges.groupby(["node1", "node2"]).transform("size")
    edges = edges.drop_duplicates(subset=["node1", "node2"], keep='first').reset_index(drop=True)
    edges = edges[edges["node1"] != edges["node2"]]
    edges = edges.rename(columns={"node1": "Start", "node2": "End"})

    G = nx.DiGraph()
    G.add_nodes_from(nodes["node"].tolist())
    G.add_edges_from(edges[["Start", "End"]].itertuples(index=False, name=None))
    if "weight" in edges.columns:
        nx.set_edge_attributes(G, values=dict(
            zip(edges[["Start", "End"]].itertuples(index=False, name=None), edges["weight"])), name="weight")
    pos = nx.spring_layout(G, weight="weight")

    #------------LESGOOOOOOOOOOOOOOOO---------
    node_shapes = {
        "s" : [],
        "d" : [],
        "o" : []
    }

    node_list = nodes["node"].tolist()

    for node_id in node_list:
        if node_id in start_ids:
            node_shapes["s"].append(node_id)
        elif node_id in end_ids:
            node_shapes["d"].append(node_id)
        else:
            node_shapes["o"].append(node_id)

    shape_colors = {
        "s" : '#f59505',
        "d" : '#f50505',
        "o" : '#787878'
    }
    #-----------------------------------------

    #------HAI SA INCERCAM O PRINTARE---------
    nx.draw_networkx_edges(G, pos, arrows=True)
    for shape, noduri in node_shapes.items():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist = noduri,
            node_shape = shape,
            node_color = shape_colors[shape]
        )
    plt.savefig("static/pdf/plot.pdf")
    #plt.axis("off")
    #plt.show()
    #-----------------------------------------

    #------------TO DELETE LATER--------------
    # nx.draw_networkx(G, pos, with_labels=False, node_size=nodes["fit"]/5+1)
    #plt.savefig("../static/pdf/plot.pdf")
    # plt.show()
    #-----------------------------------------

