"""
Tests for pcmci.py, including tests for run_pc_stable, run_mci, and run_pcmci.
"""

from __future__ import print_function
from collections import defaultdict
import itertools
import numpy as np
from nose.tools import assert_equal
import pytest

from tests import (
    VERBOSITY,
    assert_graphs_equal,
    _select_links,
    _get_parents_from_results,
    a_sample,
)
from tests.pc.test_pc_stable_calculations import a_pc_stable_params

from tigramite.pc.pcstable import PCStable
from tigramite.pcmci.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.oracle_conditional_independence import OracleCI
import tigramite.data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

# Pylint settings
# pylint: disable=redefined-outer-name


# PCMCI CONSTRUCTION ###########################################################
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
# Parameterize and return the independence test.
# Currently just a wrapper for ParCorr, but is extendable
def a_pc_impl(request):
    return PCStable


@pytest.fixture(params=[None])
# Fixture to build and return a parameterized PCMCI.  Different selected
# variables can be defined here.
def a_pcmci(a_sample, a_pc_impl, a_test, a_common_params, request):
    # Unpack the test data and true parent graph
    dataframe, true_parents = a_sample

    # Unpack the common parameters
    tau_min, tau_max, sel_link = a_common_params

    # Get the parameters from this request
    select_vars = request.param

    # Build the PCMCI instance
    pcmci = PCMCI(
        dataframe=dataframe, pc=a_pc_impl, cond_ind_test=a_test, verbosity=VERBOSITY
    )

    # Select the correct links if they are given
    select_links = _select_links(sel_link, true_parents)

    # print(select_links)
    # Ensure we change the true parents to be the same as the selected links
    if select_links is not None:
        true_parents = select_links

    # Return the constructed PCMCI, expected results, and common parameters
    return pcmci, true_parents, tau_min, tau_max, select_links


# MCI TESTING ##################################################################
@pytest.fixture(
    params=[
        # Keep parameters for the mci algorithm here
        # alpha_level, max_conds_px, max_conds_py
        (0.01, None, None)
    ]
)
def a_mci_params(request):
    # Return the parameters for the mci test
    return request.param


@pytest.fixture()
def a_run_mci(a_pcmci, a_mci_params):
    # Unpack the pcmci and the true parents, and common parameters
    pcmci, true_parents, tau_min, tau_max, select_links = a_pcmci
    # Unpack the MCI parameters
    alpha_level, max_conds_px, max_conds_py = a_mci_params
    # Run the MCI algorithm with the given parameters
    results = pcmci.run_mci(
        link_assumptions=None,
        tau_min=tau_min,
        tau_max=tau_max,
        parents=true_parents,
        max_conds_py=max_conds_px,
        max_conds_px=max_conds_py,
        alpha_level=alpha_level,
    )
    # Return the calculated and expected results
    return _get_parents_from_results(pcmci, results), true_parents


def test_mci(a_run_mci):
    """
    Test the mci algorithm and check it calculates the correct parents.
    """
    # Unpack the calculated and true parents
    parents, true_parents = a_run_mci
    # Ensure they are the same
    assert_graphs_equal(parents, true_parents)


# PCMCI TESTING ################################################################
@pytest.fixture()
def a_run_pcmci(a_pcmci, a_pc_stable_params, a_mci_params):
    # Unpack the pcmci and the true parents, and common parameters
    pcmci, true_parents, tau_min, tau_max, select_links = a_pcmci
    # Unpack the pc_stable parameters
    pc_alpha, max_conds_dim, max_combinations, save_iter = a_pc_stable_params
    # Unpack the MCI parameters
    alpha_level, max_conds_px, max_conds_py = a_mci_params
    # Run the PCMCI algorithm with the given parameters
    results = pcmci.run_pcmci(
        link_assumptions=None,
        tau_min=tau_min,
        tau_max=tau_max,
        save_iterations=save_iter,
        pc_alpha=pc_alpha,
        max_conds_dim=max_conds_dim,
        max_combinations=max_combinations,
        max_conds_px=max_conds_px,
        max_conds_py=max_conds_py,
        alpha_level=alpha_level,
    )
    # Return the results and the expected result
    return _get_parents_from_results(pcmci, results), true_parents


def test_pcmci(a_run_pcmci):
    """
    Test the pcmci algorithm and check it calculates the correct parents.
    """
    # Unpack the calculated and true parents
    parents, true_parents = a_run_pcmci
    # Ensure they are the same
    assert_graphs_equal(parents, true_parents)
