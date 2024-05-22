import numpy as np
import tigramite.data_processing as pp

from collections import Counter, defaultdict
from nose.tools import assert_equal
from tigramite.toymodels import structural_causal_processes as toys


# CONVENIENCE FUNCTIONS ########################################################
def assert_graphs_equal(actual, expected):
    """
    Check whether lists in dict have the same elements.
    This ignore the order of the elements in the list.
    """
    for j in list(expected):
        assert_equal(Counter(iter(actual[j])), Counter(iter(expected[j])))


def _get_parent_graph(parents_neighbors_coeffs, exclude=None):
    """
    Iterates through the input parent-neighghbour coefficient dictionary to
    return only parent relations (i.e. where tau != 0)
    """
    graph = defaultdict(list)
    for j, i, tau, _ in toys._iter_coeffs(parents_neighbors_coeffs):
        if tau != 0 and (i, tau) != exclude:
            graph[j].append((i, tau))
    return dict(graph)


def _select_links(link_ids, true_parents):
    """
    Select links given from the true parents dictionary
    """
    if link_ids is None:
        return None
    return {
        par: {true_parents[par][link]: "-->"}
        for par in true_parents
        for link in link_ids
    }


def _get_parents_from_results(pcmci, results):
    """
    Select the significant parents from the MCI-like results at a given
    alpha_level
    """
    significant_parents = pcmci.return_parents_dict(
        graph=results["graph"], val_matrix=results["val_matrix"]
    )
    return significant_parents


def gen_data_frame(links_coeffs, time, seed_val):
    # Set the random seed
    np.random.seed(seed_val)
    # Generate the data
    data, _ = toys.var_process(links_coeffs, T=time)
    # Get the true parents
    true_parents = _get_parent_graph(links_coeffs)
    return pp.DataFrame(data), true_parents
