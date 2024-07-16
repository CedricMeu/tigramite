from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool
from typing import Dict, List, Tuple
from tigramite.pc import PCParallelBase
from tigramite import _create_nested_dictionary, _nested_to_normal
import numpy as np


def _run_single_prepared(
    pc_impl,
    j,
    _int_link_assumptions,
    tau_min,
    tau_max,
    save_iterations,
    pc_alpha_here,
    max_conds_dim,
    max_combinations,
    select_optimal_alpha,
):

    # Get the results for this alpha value
    result = pc_impl._run_single(
        j,
        _int_link_assumptions,
        tau_min,
        tau_max,
        save_iterations,
        pc_alpha_here,
        max_conds_dim,
        max_combinations,
    )

    # Figure out the best score if there is more than one pc_alpha
    # value
    score = None
    if select_optimal_alpha:
        score = pc_impl.cond_ind_test.get_model_selection_criterion(
            j, result["parents"], tau_max
        )

    return result, score


class PCParallelOuter(PCParallelBase):
    def _run_single(
        self,
        j,
        link_assumptions_j,
        tau_min,
        tau_max,
        save_iterations,
        pc_alpha,
        max_conds_dim,
        max_combinations,
    ):
        """Lagged PC algorithm for estimating lagged parents of single variable.

        Parameters
        ----------
        j : int
            Variable index.
        link_assumptions_j : dict
            Dictionary of form {j:{(i, -tau): link_type, ...}, ...} specifying
            assumptions about links. This initializes the graph with entries
            graph[i,j,tau] = link_type. For example, graph[i,j,0] = '-->'
            implies that a directed link from i to j at lag 0 must exist.
            Valid link types are 'o-o', '-->', '<--'. In addition, the middle
            mark can be '?' instead of '-'. Then '-?>' implies that this link
            may not exist, but if it exists, its orientation is '-->'. Link
            assumptions need to be consistent, i.e., graph[i,j,0] = '-->'
            requires graph[j,i,0] = '<--' and acyclicity must hold. If a link
            does not appear in the dictionary, it is assumed absent. That is,
            if link_assumptions is not None, then all links have to be specified
            or the links are assumed absent.
        tau_min : int, optional (default: 1)
            Minimum time lag to test. Useful for variable selection in
            multi-step ahead predictions. Must be greater zero.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        save_iterations : bool, optional (default: False)
            Whether to save iteration step results such as conditions used.
        pc_alpha : float or None, optional (default: 0.2)
            Significance level in algorithm. If a list is given, pc_alpha is
            optimized using model selection criteria provided in the
            cond_ind_test class as get_model_selection_criterion(). If None,
            a default list of values is used.
        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int, optional (default: 1)
            Maximum number of combinations of conditions of current cardinality
            to test in PC1 step.

        Returns
        -------
        parents : list
            List of estimated parents.
        val_min : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum absolute
            test statistic value of a link.
        pval_max : dict
            Dictionary of form {(0, -1):float, ...} containing the maximum
            p-value of a link across different conditions.
        iterations : dict
            Dictionary containing further information on algorithm steps.
        """

        if pc_alpha < 0.0 or pc_alpha > 1.0:
            raise ValueError("Choose 0 <= pc_alpha <= 1")

        # Initialize the dictionaries for the pval_max, val_dict, val_min
        # results
        pval_max = dict()
        val_dict = dict()
        val_min = dict()
        # Initialize the parents values from the selected links, copying to
        # ensure this initial argument is unchanged.
        parents = []
        if link_assumptions_j is not None:
            for itau in link_assumptions_j:
                link_type = link_assumptions_j[itau]
                if itau != (j, 0) and link_type not in ["<--", "<?-"]:
                    parents.append(itau)

        val_dict = {(p[0], p[1]): None for p in parents}
        pval_max = {(p[0], p[1]): None for p in parents}

        # Define a nested defaultdict of depth 4 to save all information about
        # iterations
        iterations = _create_nested_dictionary(4)
        # Ensure tau_min is at least 1
        tau_min = max(1, tau_min)

        # Loop over all possible condition dimensions
        max_conds_dim = self._set_max_condition_dim(max_conds_dim, tau_min, tau_max)
        # Iteration through increasing number of conditions, i.e. from
        # [0, max_conds_dim] inclusive
        converged = False
        for conds_dim in range(max_conds_dim + 1):
            # (Re)initialize the list of non-significant links
            nonsig_parents = list()
            # Check if the algorithm has converged
            if len(parents) - 1 < conds_dim:
                converged = True
                break
            # Print information about
            if self.verbosity > 1:
                print("\nTesting condition sets of dimension %d:" % conds_dim)

            # Iterate through all possible pairs (that have not converged yet)
            for index_parent, parent in enumerate(parents):
                # Print info about this link
                if self.verbosity > 1:
                    self._print_link_info(j, index_parent, parent, len(parents))
                # Iterate through all possible combinations
                nonsig = False
                for comb_index, Z in enumerate(
                    self._iter_conditions(parent, conds_dim, parents)
                ):
                    # Break if we try too many combinations
                    if comb_index >= max_combinations:
                        break
                    # Perform independence test
                    if (
                        link_assumptions_j is not None
                        and link_assumptions_j[parent] == "-->"
                    ):
                        val = 1.0
                        pval = 0.0
                        dependent = True
                    else:
                        val, pval, dependent = self.cond_ind_test.run_test(
                            X=[parent],
                            Y=[(j, 0)],
                            Z=Z,
                            tau_max=tau_max,
                            alpha_or_thres=pc_alpha,
                        )

                    # Print some information if needed
                    if self.verbosity > 1:
                        self._print_cond_info(Z, comb_index, pval, val)

                    # Keep track of maximum p-value and minimum estimated value
                    # for each pair (across any condition)
                    val_min[parent] = min(
                        np.abs(val), val_min.get(parent, float("inf"))
                    )

                    if pval_max[parent] is None or pval > pval_max[parent]:
                        pval_max[parent] = pval
                        val_dict[parent] = val

                    # Save the iteration if we need to
                    if save_iterations:
                        a_iter = iterations["iterations"][conds_dim][parent]
                        a_iter[comb_index]["conds"] = deepcopy(Z)
                        a_iter[comb_index]["val"] = val
                        a_iter[comb_index]["pval"] = pval

                    # Delete link later and break while-loop if non-significant
                    if not dependent:  # pval > pc_alpha:
                        nonsig_parents.append((j, parent))
                        nonsig = True
                        break

                # Print the results if needed
                if self.verbosity > 1:
                    self._print_a_pc_result(nonsig, conds_dim, max_combinations)

            # Remove non-significant links
            for _, parent in nonsig_parents:
                del val_min[parent]
            # Return the parents list sorted by the test metric so that the
            # updated parents list is given to the next cond_dim loop
            parents = self._sort_parents(val_min)
            # Print information about the change in possible parents
            if self.verbosity > 1:
                print("\nUpdating parents:")
                self._print_parents_single(j, parents, val_min, pval_max)

        # Print information about if convergence was reached
        if self.verbosity > 1:
            self._print_converged_pc_single(converged, j, max_conds_dim)
        # Return the results
        return {
            "parents": parents,
            "val_min": val_min,
            "val_dict": val_dict,
            "pval_max": pval_max,
            "iterations": _nested_to_normal(iterations),
        }

    def __call__(
        self,
        selected_links=None,
        link_assumptions=None,
        tau_min=1,
        tau_max=1,
        save_iterations=False,
        pc_alpha=0.02,
        max_conds_dim=None,
        max_combinations=1,
    ) -> Tuple[
        Dict[int, List[Tuple[int, int]]],
        np.ndarray,
        np.ndarray,
        Dict[int, Dict],
        Dict[int, Dict],
        Dict[int, Dict],
    ]:
        """Lagged PC algorithm for estimating lagged parents of all variables.

        Parents are made available as self.all_parents

        Parameters
        ----------
        selected_links : dict or None
            Deprecated, replaced by link_assumptions
        link_assumptions : dict
            Dictionary of form {j:{(i, -tau): link_type, ...}, ...} specifying
            assumptions about links. This initializes the graph with entries
            graph[i,j,tau] = link_type. For example, graph[i,j,0] = '-->'
            implies that a directed link from i to j at lag 0 must exist.
            Valid link types are 'o-o', '-->', '<--'. In addition, the middle
            mark can be '?' instead of '-'. Then '-?>' implies that this link
            may not exist, but if it exists, its orientation is '-->'. Link
            assumptions need to be consistent, i.e., graph[i,j,0] = '-->'
            requires graph[j,i,0] = '<--' and acyclicity must hold. If a link
            does not appear in the dictionary, it is assumed absent. That is,
            if link_assumptions is not None, then all links have to be specified
            or the links are assumed absent.
        tau_min : int, default: 1
            Minimum time lag to test. Useful for multi-step ahead predictions.
            Must be greater zero.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        save_iterations : bool, default: False
            Whether to save iteration step results such as conditions used.
        pc_alpha : float or list of floats, default: [0.05, 0.1, 0.2, ..., 0.5]
            Significance level in algorithm. If a list or None is passed, the
            pc_alpha level is optimized for every variable across the given
            pc_alpha values using the score computed in
            cond_ind_test.get_model_selection_criterion().
        max_conds_dim : int or None
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int, default: 1
            Maximum number of combinations of conditions of current cardinality
            to test in PC1 step.

        Returns
        -------
        all_parents : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            containing estimated parents.
        """
        if selected_links is not None:
            raise ValueError(
                "selected_links is DEPRECATED, use link_assumptions instead."
            )

        # Create an internal copy of pc_alpha
        _int_pc_alpha = deepcopy(pc_alpha)

        # Check if we are selecting an optimal alpha value
        select_optimal_alpha = True

        # Set the default values for pc_alpha
        if _int_pc_alpha is None:
            _int_pc_alpha = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        elif not isinstance(_int_pc_alpha, (list, tuple, np.ndarray)):
            _int_pc_alpha = [_int_pc_alpha]
            select_optimal_alpha = False

        # Check the limits on tau_min
        self._check_tau_limits(tau_min, tau_max)
        tau_min = max(1, tau_min)

        # Check that the maximum combinations variable is correct
        if max_combinations <= 0:
            raise ValueError("max_combinations must be > 0")

        # Implement defaultdict for all pval_max, val_max, and iterations
        pval_max = defaultdict(dict)
        val_min = defaultdict(dict)
        val_dict = defaultdict(dict)
        iterations = defaultdict(dict)

        if self.verbosity > 0:
            self._print_pc_params(
                link_assumptions,
                tau_min,
                tau_max,
                _int_pc_alpha,
                max_conds_dim,
                max_combinations,
            )

        # Set the selected links
        # _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max,
        #                                      remove_contemp=True)
        _int_link_assumptions = self._set_link_assumptions(
            link_assumptions, tau_min, tau_max, remove_contemp=True
        )

        # Initialize all parents
        all_parents = dict()

        # Set the maximum condition dimension
        max_conds_dim = self._set_max_condition_dim(max_conds_dim, tau_min, tau_max)

        with Pool(processes=self.processes) as pool:
            state = []
            args = []

            # Initialize the scores for selecting the optimal alpha
            scores = defaultdict(lambda: np.zeros_like(_int_pc_alpha))

            # Loop through the selected variables
            for j in range(self.N):
                # Print the status of this variable
                if self.verbosity > 1:
                    print("\n## Variable %s" % self.var_names[j])
                    print("\nIterating through pc_alpha = %s:" % _int_pc_alpha)

                # Initialize the result
                for iscore, pc_alpha_here in enumerate(_int_pc_alpha):
                    # Print statement about the pc_alpha being tested
                    if self.verbosity > 1:
                        print(
                            "\n# pc_alpha = %s (%d/%d):"
                            % (pc_alpha_here, iscore + 1, scores[j].shape[0])
                        )

                    state.append((j, iscore, pc_alpha_here))

                    args.append(
                        (
                            self,
                            j,
                            _int_link_assumptions[j],
                            tau_min,
                            tau_max,
                            save_iterations,
                            pc_alpha_here,
                            max_conds_dim,
                            max_combinations,
                            select_optimal_alpha,
                        )
                    )

            results = defaultdict(dict)
            for (j, iscore, pc_alpha_here), (result, score) in zip(
                state, pool.starmap(_run_single_prepared, args)
            ):
                results[j][pc_alpha_here] = result
                if score is not None:
                    scores[j][iscore] = score

            for j, *_ in state:
                result = results[j]
                score = scores[j]

                # Record the optimal alpha value
                optimal_alpha = _int_pc_alpha[score.argmin()]

                # Only print the selection results if there is more than one
                # pc_alpha
                if self.verbosity > 1 and select_optimal_alpha:
                    self._print_pc_sel_results(
                        _int_pc_alpha, result, j, score, optimal_alpha
                    )

                # Record the results for this variable
                all_parents[j] = result[optimal_alpha]["parents"]
                val_min[j] = result[optimal_alpha]["val_min"]
                val_dict[j] = result[optimal_alpha]["val_dict"]
                pval_max[j] = result[optimal_alpha]["pval_max"]
                iterations[j] = result[optimal_alpha]["iterations"]

                # Only save the optimal alpha if there is more than one pc_alpha
                if select_optimal_alpha:
                    iterations[j]["optimal_pc_alpha"] = optimal_alpha

            # Save the results in the current status of the algorithm
            val_matrix = self._dict_to_matrix(val_dict, tau_max, self.N, default=0.0)
            p_matrix = self._dict_to_matrix(pval_max, tau_max, self.N, default=1.0)

            # Print the results
            if self.verbosity > 0:
                print("\n## Resulting lagged parent (super)sets:")
                self._print_parents(all_parents, val_min, pval_max)

            # Return the parents
            return (all_parents, val_matrix, p_matrix, iterations, val_min, pval_max)
