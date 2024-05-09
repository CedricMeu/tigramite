from abc import abstractmethod
import itertools
import numpy as np


def run_cond_ind_test(
    cond_ind_test,
    X,
    Y,
    Z,
    pc_alpha,
    tau_max,
    link_assumptions_j_parent,
):
    """Conditional independence test helper for parallel execution

    Parameters
    ----------
    cond_ind_test: The test to run
    pc_alpha: The alpha to use
    tau_max: Max history
    j: Variable to test
    parent: Variable to considder parent
    Z: The set to condition on
    comb_index: Stuff to pass through
    link_assumptions_j_parent: the assumption about the link
    """
    # Perform independence test
    if link_assumptions_j_parent == "-->":
        val = 1.0
        pval = 0.0
        dependent = True
    else:
        val, pval, dependent = cond_ind_test.run_test(
            X=X,
            Y=Y,
            Z=Z,
            tau_max=tau_max,
            alpha_or_thres=pc_alpha,
        )

    return val, pval, dependent


class PC:
    def __init__(self, cond_ind_test, verbosity=0):
        self.cond_ind_test = cond_ind_test

        if isinstance(self.cond_ind_test, type):
            raise ValueError(
                "PCMCI requires that cond_ind_test "
                "is instantiated, e.g. cond_ind_test =  "
                "ParCorr()."
            )

        self.verbosity = verbosity
        self.pcmci = None
        self.var_names = None

    def set_pcmci(self, pcmci):
        self.pcmci = pcmci
        self.cond_ind_test.set_dataframe(self.pcmci.dataframe)
        self.var_names = self.pcmci.dataframe.var_names

    @abstractmethod
    def run(
        self,
        link_assumptions,
        tau_min,
        tau_max,
        save_iterations,
        pc_alpha,
        max_conds_dim,
        max_combinations,
    ):
        pass

    def _print_pc_sel_results(self, pc_alpha, results, j, score, optimal_alpha):
        """Print the results from the pc_alpha selection.

        Parameters
        ----------
        pc_alpha : list
            Tested significance levels in algorithm.
        results : dict
            Results from the tested pc_alphas.
        score : array of floats
            scores from each pc_alpha.
        j : int
            Index of current variable.
        optimal_alpha : float
            Optimal value of pc_alpha.
        """
        print("\n# Condition selection results:")
        for iscore, pc_alpha_here in enumerate(pc_alpha):
            names_parents = "[ "
            for pari in results[pc_alpha_here]["parents"]:
                names_parents += "(%s % d) " % (self.var_names[pari[0]], pari[1])
            names_parents += "]"
            print(
                "    pc_alpha=%s got score %.4f with parents %s"
                % (pc_alpha_here, score[iscore], names_parents)
            )
        print(
            "\n==> optimal pc_alpha for variable %s is %s"
            % (self.var_names[j], optimal_alpha)
        )

    def _iter_conditions(self, parent, conds_dim, all_parents):
        """Yield next condition.

        Yields next condition from lexicographically ordered conditions.

        Parameters
        ----------
        parent : tuple
            Tuple of form (i, -tau).
        conds_dim : int
            Cardinality in current step.
        all_parents : list
            List of form [(0, -1), (3, -2), ...].

        Yields
        -------
        cond :  list
            List of form [(0, -1), (3, -2), ...] for the next condition.
        """
        all_parents_excl_current = [p for p in all_parents if p != parent]
        for cond in itertools.combinations(all_parents_excl_current, conds_dim):
            yield list(cond)

    def _sort_parents(self, parents_vals):
        """Sort current parents according to test statistic values.

        Sorting is from strongest to weakest absolute values.

        Parameters
        ---------
        parents_vals : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum test
            statistic value of a link.

        Returns
        -------
        parents : list
            List of form [(0, -1), (3, -2), ...] containing sorted parents.
        """
        if self.verbosity > 1:
            print(
                "\n    Sorting parents in decreasing order with "
                "\n    weight(i-tau->j) = min_{iterations} |val_{ij}(tau)| "
            )
        # Get the absolute value for all the test statistics
        abs_values = {k: np.abs(parents_vals[k]) for k in list(parents_vals)}
        return sorted(abs_values, key=abs_values.get, reverse=True)

    def _dict_to_matrix(self, val_dict, tau_max, n_vars, default=1.0):
        """Helper function to convert dictionary to matrix format.

        Parameters
        ---------
        val_dict : dict
            Dictionary of form {0:{(0, -1):float, ...}, 1:{...}, ...}.
        tau_max : int
            Maximum lag.
        n_vars : int
            Number of variables.
        default : int
            Default value for entries not part of val_dict.

        Returns
        -------
        matrix : array of shape (N, N, tau_max+1)
            Matrix format of p-values and test statistic values.
        """
        matrix = np.ones((n_vars, n_vars, tau_max + 1))
        matrix *= default

        for j in val_dict.keys():
            for link in val_dict[j].keys():
                k, tau = link
                if tau == 0:
                    matrix[k, j, 0] = matrix[j, k, 0] = val_dict[j][link]
                else:
                    matrix[k, j, abs(tau)] = val_dict[j][link]
        return matrix
