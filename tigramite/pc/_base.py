from abc import abstractmethod


class _PCbase:
    @abstractmethod
    def __call__(
        self,
        selected_links=None,
        link_assumptions=None,
        tau_min=1,
        tau_max=1,
        save_iterations=False,
        pc_alpha=0.2,
        max_conds_dim=None,
        max_combinations=1,
    ) -> dict:
        pass
