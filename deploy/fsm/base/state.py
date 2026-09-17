"""Base class for deployment states."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from common.control import ControlTarget, RobotState, StateName


class State(ABC):
    name: StateName

    def enter(self, robot: RobotState) -> None:
        del robot

    @abstractmethod
    def step(
        self,
        robot: RobotState,
        velocity_command: NDArray[np.float32],
    ) -> ControlTarget:
        """Compute one policy-period target."""

    def exit(self) -> None:
        pass
