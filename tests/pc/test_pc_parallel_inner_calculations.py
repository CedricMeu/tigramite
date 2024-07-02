import pytest
from tests import VERBOSITY, assert_graphs_equal, _select_links, a_sample
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pc.pcstable import PCStable


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
def a_pc_parallel_inner(a_test):
    return PCParallelInner(a_test)


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
