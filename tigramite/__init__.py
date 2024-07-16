from copy import deepcopy
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd

from tigramite.independence_tests.independence_tests_base import CondIndTest


class CDAlgoBase:
    def __init__(
        self, dataframe: pd.DataFrame, cond_ind_test: CondIndTest, verbosity: int = 0
    ):
        self.dataframe = dataframe

        # Set the conditional independence test to be used
        self.cond_ind_test = cond_ind_test
        if isinstance(self.cond_ind_test, type):
            raise ValueError(
                "PCMCI requires that cond_ind_test "
                "is instantiated, e.g. cond_ind_test =  "
                "ParCorr()."
            )
        self.cond_ind_test.set_dataframe(self.dataframe)

        # Set the verbosity for debugging/logging messages
        self.verbosity = verbosity

        # Set the variable names
        self.var_names = self.dataframe.var_names

        # Store the shape of the data in the T and N variables
        self.T = self.dataframe.T
        self.N = self.dataframe.N

    def _set_max_condition_dim(self, max_conds_dim, tau_min, tau_max):
        """
        Set the maximum dimension of the conditions. Defaults to self.N*tau_max.

        Parameters
        ----------
        max_conds_dim : int
            Input maximum condition dimension.
        tau_max : int
            Maximum tau.

        Returns
        -------
        max_conds_dim : int
            Input maximum condition dimension or default.
        """
        # Check if an input was given
        if max_conds_dim is None:
            max_conds_dim = self.N * (tau_max - tau_min + 1)
        # Check this is a valid
        if max_conds_dim < 0:
            raise ValueError("maximum condition dimension must be >= 0")
        return max_conds_dim

    def _print_link_info(
        self, j, index_parent, parent, num_parents, already_removed=False
    ):
        """Print info about the current link being tested.

        Parameters
        ----------
        j : int
            Index of current node being tested.
        index_parent : int
            Index of the current parent.
        parent : tuple
            Standard (i, tau) tuple of parent node id and time delay
        num_parents : int
            Total number of parents.
        already_removed : bool
            Whether parent was already removed.
        """
        link_marker = {True: "o?o", False: "-?>"}

        abstau = abs(parent[1])
        if self.verbosity > 1:
            print(
                "\n    Link (%s % d) %s %s (%d/%d):"
                % (
                    self.var_names[parent[0]],
                    parent[1],
                    link_marker[abstau == 0],
                    self.var_names[j],
                    index_parent + 1,
                    num_parents,
                )
            )

            if already_removed:
                print("    Already removed.")

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

    def _print_cond_info(self, Z, comb_index, pval, val):
        """Print info about the condition

        Parameters
        ----------
        Z : list
            The current condition being tested.
        comb_index : int
            Index of the combination yielding this condition.
        pval : float
            p-value from this condition.
        val : float
            value from this condition.
        """
        var_name_z = ""
        for i, tau in Z:
            var_name_z += "(%s % .2s) " % (self.var_names[i], tau)
        if len(Z) == 0:
            var_name_z = "()"
        print(
            "    Subset %d: %s gives pval = %.5f / val = % .3f"
            % (comb_index, var_name_z, pval, val)
        )

    def _print_a_pc_result(self, nonsig, conds_dim, max_combinations):
        """Print the results from the current iteration of conditions.

        Parameters
        ----------
        nonsig : bool
            Indicate non-significance.
        conds_dim : int
            Cardinality of the current step.
        max_combinations : int
            Maximum number of combinations of conditions of current cardinality
            to test.
        """
        # Start with an indent
        print_str = "    "
        # Determine the body of the text
        if nonsig:
            print_str += "Non-significance detected."
        elif conds_dim > max_combinations:
            print_str += (
                "Still subsets of dimension"
                + " %d left," % (conds_dim)
                + " but q_max = %d reached." % (max_combinations)
            )
        else:
            print_str += "No conditions of dimension %d left." % (conds_dim)
        # Print the message
        print(print_str)

    def _print_parents_single(self, j, parents, val_min, pval_max):
        """Print current parents for variable j.

        Parameters
        ----------
        j : int
            Index of current variable.
        parents : list
            List of form [(0, -1), (3, -2), ...].
        val_min : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum absolute
            test statistic value of a link.
        pval_max : dict
            Dictionary of form {(0, -1):float, ...} containing the maximum
            p-value of a link across different conditions.
        """
        if len(parents) < 20 or hasattr(self, "iterations"):
            print(
                "\n    Variable %s has %d link(s):" % (self.var_names[j], len(parents))
            )

            # if hasattr(self, "iterations") and "optimal_pc_alpha" in list(
            #     self.iterations[j]
            # ):
            #     print("    [pc_alpha = %s]" % (self.iterations[j]["optimal_pc_alpha"]))

            if val_min is None or pval_max is None:
                for p in parents:
                    print("        (%s % .d)" % (self.var_names[p[0]], p[1]))
            else:
                for p in parents:
                    print(
                        "        (%s % .d): max_pval = %.5f, |min_val| = % .3f"
                        % (self.var_names[p[0]], p[1], pval_max[p], abs(val_min[p]))
                    )
        else:
            print(
                "\n    Variable %s has %d link(s):" % (self.var_names[j], len(parents))
            )

    def _print_converged_pc_single(self, converged, j, max_conds_dim):
        """
        Print statement about the convergence of the pc_stable_single algorithm.

        Parameters
        ----------
        convergence : bool
            true if convergence was reached.
        j : int
            Variable index.
        max_conds_dim : int
            Maximum number of conditions to test.
        """
        if converged:
            print("\nAlgorithm converged for variable %s" % self.var_names[j])
        else:
            print(
                "\nAlgorithm not yet converged, but max_conds_dim = %d"
                " reached." % max_conds_dim
            )

    def _check_tau_limits(self, tau_min, tau_max):
        """Check the tau limits adhere to 0 <= tau_min <= tau_max.

        Parameters
        ----------
        tau_min : float
            Minimum tau value.
        tau_max : float
            Maximum tau value.
        """
        if not 0 <= tau_min <= tau_max:
            raise ValueError(
                "tau_max = %d, " % (tau_max)
                + "tau_min = %d, " % (tau_min)
                + "but 0 <= tau_min <= tau_max"
            )

    def _print_pc_params(
        self,
        link_assumptions,
        tau_min,
        tau_max,
        pc_alpha,
        max_conds_dim,
        max_combinations,
    ):
        """Print the setup of the current pc_stable run.

        Parameters
        ----------
        link_assumptions : dict or None
            Dictionary of form specifying which links should be tested.
        tau_min : int, default: 1
            Minimum time lag to test.
        tau_max : int, default: 1
            Maximum time lag to test.
        pc_alpha : float or list of floats
            Significance level in algorithm.
        max_conds_dim : int
            Maximum number of conditions to test.
        max_combinations : int
            Maximum number of combinations of conditions to test.
        """
        print(
            "\n##\n## Step 1: PC1 algorithm for selecting lagged conditions\n##"
            "\n\nParameters:"
        )
        if link_assumptions is not None:
            print("link_assumptions = %s" % str(link_assumptions))
        print(
            "independence test = %s" % self.cond_ind_test.measure
            + "\ntau_min = %d" % tau_min
            + "\ntau_max = %d" % tau_max
            + "\npc_alpha = %s" % pc_alpha
            + "\nmax_conds_dim = %s" % max_conds_dim
            + "\nmax_combinations = %d" % max_combinations
        )
        print("\n")

    def _reverse_link(self, link):
        """Reverse a given link, taking care to replace > with < and vice versa."""

        if link == "":
            return ""

        if link[2] == ">":
            left_mark = "<"
        else:
            left_mark = link[2]

        if link[0] == "<":
            right_mark = ">"
        else:
            right_mark = link[0]

        return left_mark + link[1] + right_mark

    def _check_cyclic(self, link_dict):
        """Return True if the link_dict has a contemporaneous cycle."""

        path = set()
        visited = set()

        def visit(vertex):
            if vertex in visited:
                return False
            visited.add(vertex)
            path.add(vertex)
            for itaui in link_dict.get(vertex, ()):
                i, taui = itaui
                link_type = link_dict[vertex][itaui]
                if taui == 0 and link_type in ["-->", "-?>"]:
                    if i in path or visit(i):
                        return True
            path.remove(vertex)
            return False

        return any(visit(v) for v in link_dict)

    def _set_link_assumptions(
        self, link_assumptions, tau_min, tau_max, remove_contemp=False
    ):
        """Helper function to set and check the link_assumptions argument

        Parameters
        ----------
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
        tau_mix : int
            Minimum time delay to test.
        tau_max : int
            Maximum time delay to test.
        remove_contemp : bool
            Whether contemporaneous links (at lag zero) should be removed.

        Returns
        -------
        link_assumptions : dict
            Cleaned links.
        """
        # Copy and pass into the function
        _int_link_assumptions = deepcopy(link_assumptions)
        # Set the default selected links if none are set
        _vars = list(range(self.N))
        _lags = list(range(-(tau_max), -tau_min + 1, 1))
        if _int_link_assumptions is None:
            _int_link_assumptions = {}
            # Set the default as all combinations
            for j in _vars:
                _int_link_assumptions[j] = {}
                for i in _vars:
                    for lag in range(tau_min, tau_max + 1):
                        if not (i == j and lag == 0):
                            if lag == 0:
                                _int_link_assumptions[j][(i, 0)] = "o?o"
                            else:
                                _int_link_assumptions[j][(i, -lag)] = "-?>"

        else:

            if remove_contemp:
                for j in _int_link_assumptions.keys():
                    _int_link_assumptions[j] = {
                        link: _int_link_assumptions[j][link]
                        for link in _int_link_assumptions[j]
                        if link[1] != 0
                    }

        # Make contemporaneous assumptions consistent and orient lagged links
        for j in _vars:
            for link in _int_link_assumptions[j]:
                i, tau = link
                link_type = _int_link_assumptions[j][link]
                if tau == 0:
                    if (j, 0) in _int_link_assumptions[i]:
                        if _int_link_assumptions[j][link] != self._reverse_link(
                            _int_link_assumptions[i][(j, 0)]
                        ):
                            raise ValueError(
                                "Inconsistent link assumptions for indices %d - %d "
                                % (i, j)
                            )
                    else:
                        _int_link_assumptions[i][(j, 0)] = self._reverse_link(
                            _int_link_assumptions[j][link]
                        )
                else:
                    # Orient lagged links by time order while leaving the middle mark
                    new_link_type = "-" + link_type[1] + ">"
                    _int_link_assumptions[j][link] = new_link_type

        # Otherwise, check that our assumpions are sane
        # Check that the link_assumptions refer to links that are inside the
        # data range and types
        _key_set = set(_int_link_assumptions.keys())
        valid_entries = _key_set == set(range(self.N))

        valid_types = [
            "o-o",
            "o?o",
            "-->",
            "-?>",
            "<--",
            "<?-",
        ]

        for links in _int_link_assumptions.values():
            if isinstance(links, dict) and len(links) == 0:
                continue
            for var, lag in links:
                if var not in _vars or lag not in _lags:
                    valid_entries = False
                if links[(var, lag)] not in valid_types:
                    valid_entries = False

        if not valid_entries:
            raise ValueError(
                "link_assumptions"
                " must be dictionary with keys for all [0,...,N-1]"
                " variables and contain only links from "
                "these variables in range [tau_min, tau_max] "
                "and with link types in %s" % str(valid_types)
            )

        # Check for contemporaneous cycles
        if self._check_cyclic(_int_link_assumptions):
            raise ValueError("link_assumptions has contemporaneous cycle(s).")

        # Return the _int_link_assumptions
        return _int_link_assumptions

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

    def _print_parents(self, all_parents, val_min, pval_max):
        """Print current parents.

        Parameters
        ----------
        all_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            the conditioning-parents estimated with PC algorithm.
        val_min : dict
            Dictionary of form {0:{(0, -1):float, ...}} containing the minimum
            absolute test statistic value of a link.
        pval_max : dict
            Dictionary of form {0:{(0, -1):float, ...}} containing the maximum
            p-value of a link across different conditions.
        """
        for j in [var for var in list(all_parents)]:
            if val_min is None or pval_max is None:
                self._print_parents_single(j, all_parents[j], None, None)
            else:
                self._print_parents_single(j, all_parents[j], val_min[j], pval_max[j])

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


def _create_nested_dictionary(depth=0, lowest_type=dict):
    """Create a series of nested dictionaries to a maximum depth.  The first
    depth - 1 nested dictionaries are defaultdicts, the last is a normal
    dictionary.

    Parameters
    ----------
    depth : int
        Maximum depth argument.
    lowest_type: callable (optional)
        Type contained in leaves of tree.  Ex: list, dict, tuple, int, float ...
    """
    new_depth = depth - 1
    if new_depth <= 0:
        return defaultdict(lowest_type)
    return defaultdict(lambda: _create_nested_dictionary(new_depth))


def _nested_to_normal(nested_dict):
    """Transforms the nested default dictionary into a standard dictionaries

    Parameters
    ----------
    nested_dict : default dictionary of default dictionaries of ... etc.
    """
    if isinstance(nested_dict, defaultdict):
        nested_dict = {k: _nested_to_normal(v) for k, v in nested_dict.items()}
    return nested_dict
