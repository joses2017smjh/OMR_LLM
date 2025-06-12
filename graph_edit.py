import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.similarity import graph_edit_distance



def check_node(G, arg):
    if not G.has_node(arg):
        if arg.startswith("const_"):
            G.add_node(arg, label="const")
        elif arg.startswith("n"):
            G.add_node(arg, label="Num_ref")
        # elif arg.startswith("#"):
        #     G.add_node(arg, label="op_output")
        # else:
        #     G.add_node(arg, label="input")

def build_dag(op_string):
    G = nx.DiGraph()
    op_list = [op for op in op_string.strip("|").split("|")]
    #print(op_list)

    for idx, expr in enumerate(op_list):
        op_id = f"#{idx}"

        op_name, args = expr.split("(")
        op_name.strip()
        #print(op_name)

        args = args.strip(")").split(",")

        #print(args)

        G.add_node(op_id, label=op_name)

        for arg in args:
            check_node(G, arg)
            G.add_edge(arg, op_id, label=arg)

    return G


def draw_dag(G, title="Graph"):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    nx.draw(G, pos, with_labels=True, labels=node_labels,
            node_size=1500, node_color='skyblue', font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.show()



def ged(G1, G2):
    return graph_edit_distance(
        G1, G2,
        node_match=lambda n1, n2: n1['label'] == n2['label'],
        edge_match=lambda e1, e2: e1['label'] == e2['label'],
        #node_subst_cost=lambda n1, n2: 0 if n1['label'] == n2['label'] else 1,
        #edge_subst_cost=lambda e1, e2: 0 if e1['label'] == e2['label'] else 1,

    )


def compute(trg_graph, pred_graph):
    G1 = build_dag(trg_graph)
    G2 = build_dag(pred_graph)
    return ged(G1, G2)


if __name__ == "__main__":
    
    graph_str1 = "multiply(n2,n2)|subtract(#0,const_1)|"
    graph_str2 = "multiply(n1,n2)|subtract(#0,const_1)|"
    G1 = build_dag(graph_str1)
    G2 = build_dag(graph_str2)

    # draw_dag(G1, title="Graph 1")
    # draw_dag(G2, title="Graph 2")

    ged = ged(G1, G2)
    print(f"Graph Edit Distance: {ged}")
