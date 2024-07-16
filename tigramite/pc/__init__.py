from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing_extensions import Self
from tigramite import CDAlgoBase
import numpy as np
import pandas as pd

from tigramite.independence_tests.independence_tests_base import CondIndTest


class PCBase(CDAlgoBase):
    @abstractmethod
    def __call__(
        self,
        selected_links=None,
        link_assumptions=None,
        tau_min=1,
        tau_max=1,
        save_iterations=False,
        pc_alpha=0.02,
        max_conds_dim: Optional[int] = None,
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


class PCParallelBase(PCBase):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cond_ind_test: CondIndTest,
        verbosity: int = 0,
        processes: Optional[int] = None,
    ):
        super().__init__(dataframe, cond_ind_test, verbosity)
        self.processes = processes

    @classmethod
    def with_params(
        cls: type[Self], processes: Optional[int] = None
    ) -> Callable[[pd.DataFrame, CondIndTest, int], Self]:
        def _with(
            dataframe: pd.DataFrame, cond_ind_test: CondIndTest, verbosity: int
        ) -> Self:
            obj = cls(dataframe, cond_ind_test, verbosity, processes)
            return obj

        return _with
