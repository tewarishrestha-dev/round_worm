
import pandas as pd
import networkx as nx

DATA_PATH = "data/connectome.csv"  # tab-separated despite the .csv extension


def load_connectome(path: str = DATA_PATH) -> nx.MultiDiGraph:
    
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]

    g = nx.MultiDiGraph()

    for _, row in df.iterrows():
        pre, post, syn_type, n_synapses = row["pre"], row["post"], row["type"], row["synapses"]

        g.add_edge(
            pre,
            post,
            synapse_type=syn_type,      # "chemical" or "electrical"
            weight=float(n_synapses),   # synapse count -> use as connection strength
        )

        # Electrical synapses (gap junctions) are physically bidirectional —
        # the raw data lists them once per direction inconsistently, so we
        # explicitly add the reverse edge to be safe. Running this twice on
        # an already-bidirectional pair just re-adds the same MultiDiGraph
        # edge, which is harmless.
        if syn_type == "electrical":
            g.add_edge(post, pre, synapse_type=syn_type, weight=float(n_synapses))

    return g


def print_stats(g: nx.MultiDiGraph) -> None:
    n_chemical = sum(1 for _, _, d in g.edges(data=True) if d["synapse_type"] == "chemical")
    n_electrical = sum(1 for _, _, d in g.edges(data=True) if d["synapse_type"] == "electrical")

    degrees = sorted(g.degree(), key=lambda x: -x[1])

    print(f"Nodes (neurons/cells): {g.number_of_nodes()}")
    print(f"Total edges:           {g.number_of_edges()}")
    print(f"  chemical synapses:   {n_chemical}")
    print(f"  electrical (gap jn): {n_electrical}")
    print()
    print("Top 5 most-connected nodes (hub neurons):")
    for name, deg in degrees[:5]:
        print(f"  {name:10s} degree={deg}")


if __name__ == "__main__":
    graph = load_connectome()
    print_stats(graph)
