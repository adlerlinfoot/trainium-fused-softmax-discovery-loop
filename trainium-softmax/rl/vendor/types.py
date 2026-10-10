from dataclasses import dataclass, field
from typing import Any
Action = list[int]
@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: Any = None
    next_stop_condition: Any = None
    metrics: dict = field(default_factory=dict)
