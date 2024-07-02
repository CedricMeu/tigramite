import pytest
from tests import VERBOSITY, assert_graphs_equal, _select_links, a_sample
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pc.pcparallelinner import PCParallelInner


# PCParallelInner CONSTRUCTION ###########################################################
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
def a_test():
    return ParCorr(verbosity=VERBOSITY)


@pytest.fixture()
# Fixture to build and return a parameterized PCMCI.  Different selected
# variables can be defined here.
def a_pcparallelinner(a_sample, a_test, a_common_params):
    # Unpack the test data and true parent graph
    dataframe, true_parents = a_sample

    # Unpack the common parameters
    tau_min, tau_max, sel_link = a_common_params

    # Build the PCStable instance
    pc = PCParallelInner(dataframe=dataframe, cond_ind_test=a_test, verbosity=VERBOSITY)

    # Select the correct links if they are given
    select_links = _select_links(sel_link, true_parents)

    # print(select_links)
    # Ensure we change the true parents to be the same as the selected links
    if select_links is not None:
        true_parents = select_links

    # Return the constructed PCMCI, expected results, and common parameters
    return pc, true_parents, tau_min, tau_max, select_links


# PCParallelInner TESTING ############################################################
@pytest.fixture(
    params=[
        # Keep parameters for the pc_stable algorithm here
        # pc_alpha,  max_conds_dim,  max_comb, save_iterations
        (None, None, 3, False),
        (0.05, None, 1, False),
        (0.05, None, 10, False),
        # (0.05,      None,           1,        True),
        # (0.05,      3,              1,        False)
    ]
)
def a_pc_stable_params(request):
    # Return the parameters for the pc_stable test
    return request.param


@pytest.fixture()
def a_run_pc_stable(a_pcparallelinner, a_pc_stable_params):
    # Unpack PC, true parents, and common parameters
    pc, true_parents, tau_min, tau_max, _ = a_pcparallelinner

    # Unpack the pc_stable parameters
    pc_alpha, max_conds_dim, max_combinations, save_iter = a_pc_stable_params

    # Run PC stable
    all_parents, *_ = pc(
        link_assumptions=None,
        tau_min=tau_min,
        tau_max=tau_max,
        save_iterations=save_iter,
        pc_alpha=pc_alpha,
        max_conds_dim=max_conds_dim,
        max_combinations=max_combinations,
    )
    # Return the calculated and expected results
    return all_parents, true_parents


def test_pc_stable(a_run_pc_stable):
    """
    Test the pc_stable algorithm and check it calculates the correct parents.
    """
    # Unpack the calculated and true parents
    parents, true_parents = a_run_pc_stable
    # Ensure they are the same
    assert_graphs_equal(parents, true_parents)
