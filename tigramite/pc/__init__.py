from abc import abstractmethod
from typing import Dict, List, Tuple
from tigramite import _CDAlgoBase
import numpy as np


class _PCbase(_CDAlgoBase):
    @abstractmethod
    def __call__(
        self,
        pcmci,
        selected_links=None,
        link_assumptions=None,
        tau_min=1,
        tau_max=1,
        save_iterations=False,
        pc_alpha=0.2,
        max_conds_dim=None,
        max_combinations=1,
    ) -> Dict[int, List[Tuple[int, int]]]:
        pass

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
