from enum import Enum

from .skill_type import SkillType
from .skill_plan import SkillStep


class ExecutionBackend(str, Enum):
    CLASSICAL = "classical"
    LEARNED = "learned"


class SkillRouter:
    def route(self, step: SkillStep) -> ExecutionBackend:
        match step.skill:
            case SkillType.PICK | SkillType.PLACE:
                return ExecutionBackend.LEARNED
            case _:
                raise ValueError(f"Unsupported skill: {step.skill}")
