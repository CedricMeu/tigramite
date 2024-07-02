from collections import Counter, defaultdict
from nose.tools import assert_equal
import pytest

import numpy as np

from tigramite.toymodels import structural_causal_processes as toys
import tigramite.data_processing as pp

# Define the verbosity at the global scope
VERBOSITY = 1


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


def assert_graphs_equal(actual, expected):
    """
    Check whether lists in dict have the same elements.
    This ignore the order of the elements in the list.
    """
    for j in list(expected):
        assert_equal(Counter(iter(actual[j])), Counter(iter(expected[j])))


# TEST LINK GENERATION #########################################################
def a_chain(auto_corr, coeff, length=3):
    """
    Generate a simple chain process with the given auto-correlations and
    parents with the given coefficient strength.  A length can also be defined
    to get a longer chain.

    Parameters
    ----------
    auto_corr: float
        Autocorrelation strength for all nodes
    coeff : float
        Parent strength for all relations
    length : int
        Length of the chain
    """
    return_links = dict()
    return_links[0] = [((0, -1), auto_corr)]
    for lnk in range(1, length):
        return_links[lnk] = [((lnk, -1), auto_corr), ((lnk - 1, -1), coeff)]
    return return_links


# TODO implement common_driver: return two variables commonly driven by N common
# drivers which are random noise, autocorrelation as parameter
# TODO implement independent drivers, autocorrelated noise
# TODO check common_driver, independent driver cases for current variable sets
# TODO implement USER_INPUT dictionary,
# USER_INPUT = dict()


# TEST DATA GENERATION #########################################################
@pytest.fixture(
    params=[
        # Generate a test data sample
        # Parameterize the sample by setting the autocorrelation value, coefficient
        # value, total time length, and random seed to different numbers
        # links_coeffs,               time,  seed_val
        (a_chain(0.1, 0.9), 1000, 2),
        (a_chain(0.5, 0.6), 1000, 11),
        (a_chain(0.5, 0.6, length=5), 10000, 42),
    ]
)
def a_sample(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_data_frame(links_coeffs, time, seed_val)
