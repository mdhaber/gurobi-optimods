import io
import unittest

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    nx = None

from gurobi_optimods.datasets import (
    _convert_pandas_to_digraph,
    _convert_pandas_to_scipy,
)


def load_graph(graph_csv, backend):
    """Load (headerless) source, target, capacity CSV into the requested backend."""
    names = ["source", "target", "capacity"]
    graph = pd.read_csv(io.StringIO(graph_csv), header=None, names=names)
    graph = graph.set_index(["source", "target"])
    if backend == "scipy":
        graph = _convert_pandas_to_scipy(graph, None, cost=False, demand=False)[1]
    elif backend == "networkx":
        if nx is None:
            raise unittest.SkipTest("networkx is not installed")
        graph = _convert_pandas_to_digraph(graph, None, demand=False)
    return graph


def check_solution_pandas(solution, candidates):
    # Checks whether the solution (`pd.Series`) matches any of the list of
    # candidates. Each candidate is a list of tuples ((i, j), v) tuples,
    # compare with the solution in sorted order.
    solution_list = sorted(solution.items())
    return any(solution_list == sorted(candidate) for candidate in candidates)


def check_solution_scipy(solution, candidates):
    # Checks whether the solution (`sp.sparray`) matches any of the list of
    # candidates (containing `np.ndarray`)
    arr = solution.toarray()
    return any(np.array_equal(arr, c) for c in candidates)


def check_solution_networkx(solution, candidates):
    # Checks whether the solution (`nx.DiGraph`) matches any of the list of
    # candidates (containing tuples dict `{(i, j): data}`)
    solution_list = sorted(
        [((i, j), data["flow"]) for i, j, data in solution.edges(data=True)],
        key=str,
    )
    return any(solution_list == sorted(candidate, key=str) for candidate in candidates)
