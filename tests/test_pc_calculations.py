"""
Tests for pcmci.py, including tests for run_pc_stable, run_mci, and run_pcmci.
"""
from __future__ import print_function
import pytest
from tests import assert_graphs_equal, gen_data_frame
from tigramite.independence_tests.parcorr import ParCorr

from tigramite.pcstable import PCStable
from tigramite.pcparallel import PCParallelInner

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
VERBOSITY = 1


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


# PC-Stable TESTING ############################################################
@pytest.fixture(
    params=[
        # Keep parameters for the pc_parallel algorithm here
        # pc_alpha,  max_conds_dim,  max_comb, save_iterations
        (None, None, 3, False),
        (0.05, None, 1, False),
        (0.05, None, 10, False),
        # (0.05,      None,           1,        True),
        # (0.05,      3,              1,        False)
    ]
)
def a_pc_stable_params(request):
    # Return the parameters for the pc_parallel test
    return request.param


@pytest.fixture(
    params=[
        # Keep parameters common for all the run_ algorithms here
        # tau_min, tau_max,  sel_link,
        (1, 2, None),
        # (1,       2,        [0])
    ]
)
def a_common_params(request):
    # Return the requested parameters
    return request.param


@pytest.fixture()
# Parameterize and return the independence test.
# Currently just a wrapper for ParCorr, but is extendable
def a_test(request):
    return ParCorr(verbosity=VERBOSITY)


@pytest.fixture()
def a_pc_stable(a_test):
    return PCStable(a_test)


@pytest.fixture()
def a_run_pc_stable(a_sample, a_common_params, a_pc_stable, a_pc_stable_params):
    # Unpack the test data and true parent graph
    _, true_parents = a_sample

    # Unpack the common parameters
    tau_min, tau_max, _ = a_common_params

    # Unpack the pcmci, true parents, and common parameters
    pc = a_pc_stable

    # Unpack the pc_stable parameters
    pc_alpha, max_conds_dim, max_combinations, save_iter = a_pc_stable_params

    # Run PC stable
    pc.run(
        link_assumptions=None,
        tau_min=tau_min,
        tau_max=tau_max,
        save_iterations=save_iter,
        pc_alpha=pc_alpha,
        max_conds_dim=max_conds_dim,
        max_combinations=max_combinations,
    )

    return pc.all_parents, true_parents


def test_pc_stable(a_run_pc_stable):
    """
    Test the pc_stable algorithm and check it calculates the correct parents.
    """
    # Unpack the calculated and true parents
    parents, true_parents = a_run_pc_stable

    # Ensure they are the same
    assert_graphs_equal(parents, true_parents)


# PC-Parallel-Inner TESTING ############################################################
@pytest.fixture(
    params=[
        # Keep parameters for the pc_parallel algorithm here
        # pc_alpha,  max_conds_dim,  max_comb, save_iterations
        (None, None, 3, False),
        (0.05, None, 1, False),
        (0.05, None, 10, False),
        # (0.05,      None,           1,        True),
        # (0.05,      3,              1,        False)
    ]
)
def a_pc_parallel_inner_params(request):
    # Return the parameters for the pc_parallel test
    return request.param


@pytest.fixture()
def a_pc_inner(a_test):
    return PCParallelInner(a_test)


@pytest.fixture()
def a_pc_parallel_inner(a_test):
    return PCStable(a_test)


@pytest.fixture()
def a_run_pc_parallel_inner(
    a_sample, a_common_params, a_pc_parallel_inner, a_pc_parallel_inner_params
):
    # Unpack the test data and true parent graph
    _, true_parents = a_sample

    # Unpack the common parameters
    tau_min, tau_max, _ = a_common_params

    # Unpack the pcmci, true parents, and common parameters
    pc = a_pc_parallel_inner

    # Unpack the pc_parallel parameters
    pc_alpha, max_conds_dim, max_combinations, save_iter = a_pc_parallel_inner_params

    # Run PC parallel inner
    pc.run(
        link_assumptions=None,
        tau_min=tau_min,
        tau_max=tau_max,
        save_iterations=save_iter,
        pc_alpha=pc_alpha,
        max_conds_dim=max_conds_dim,
        max_combinations=max_combinations,
    )

    return pc.all_parents, true_parents


def test_pc_parallel_inner(a_run_pc_parallel_inner):
    """
    Test the pc_parallel algorithm and check it calculates the correct parents.
    """
    # Unpack the calculated and true parents
    parents, true_parents = a_run_pc_parallel_inner

    # Ensure they are the same
    assert_graphs_equal(parents, true_parents)
