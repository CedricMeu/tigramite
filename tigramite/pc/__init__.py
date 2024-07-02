from abc import abstractmethod
from typing import Dict, List, Tuple
from tigramite import _CDAlgoBase
import numpy as np


class _PCBase(_CDAlgoBase):
    @abstractmethod
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
        pass
