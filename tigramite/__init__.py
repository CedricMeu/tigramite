from abc import abstractmethod
from typing import Dict, List, Tuple
from collections import defaultdict


class _CDAlgoBase:
    def __init__(self, dataframe, cond_ind_test, verbosity=0):
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

    # @abstractmethod
    # def __call__(
    #     self,
    #     pcmci,
    #     selected_links=None,
    #     link_assumptions=None,
    #     tau_min=1,
    #     tau_max=1,
    #     save_iterations=False,
    #     max_conds_dim=None,
    #     max_combinations=1,
    # ) -> Dict[int, List[Tuple[int, int]]]:
    #     pass

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
