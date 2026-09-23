from pydantic import BaseModel
from enum import Enum
from .skill_result import SkillResult, SkillResultReason, SkillResultStatus
from .world_state import WorldState


class RecoveryAction(str, Enum):
    RETRY = "retry"
    REPOSITION = "reposition"
    REOBSERVE_AND_REGROUND = "reobserve_and_reground"
    ASK_SYSTEM1 = "ask_system1"
    ASK_SYSTEM2 = "ask_system2"
    ABORT = "abort"


class RecoveryContext(BaseModel):
    world_state: WorldState
    skill_result: SkillResult
    retry_count: int


class RecoveryRouter:
    """Select the next action after a skill execution result."""

    def decide_recovery(self, result: SkillResult) -> RecoveryAction | None:
        if result.status in {
            SkillResultStatus.PRECONDITION_FAILED,
            SkillResultStatus.FAILED,
            SkillResultStatus.TIMEOUT,
        }:
            match result.reason:
                case SkillResultReason.TARGET_UNAVAILABLE:
                    return RecoveryAction.REOBSERVE_AND_REGROUND
                case SkillResultReason.SAFETY_VIOLATION:
                    return RecoveryAction.ABORT
                case SkillResultReason.ACTION_FAILED:
                    return RecoveryAction.ASK_SYSTEM1
                case _:
                    return RecoveryAction.ASK_SYSTEM2

        return None
