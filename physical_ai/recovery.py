from enum import Enum

from pydantic import BaseModel, Field

from .skill_result import SkillResult, SkillResultReason, SkillResultStatus
from .world_state import WorldState


class RecoveryAction(str, Enum):
    RETRY = "retry"
    REPOSITION = "reposition"
    REOBSERVE_AND_REGROUND = "reobserve_and_reground"
    ASK_SYSTEM1 = "ask_system1"
    ASK_SYSTEM2 = "ask_system2"
    ABORT = "abort"


class ExecutionObservation(BaseModel):
    target_visible: bool
    target_position_changed: bool
    target_reachable: bool
    contact_detected: bool
    grasp_quality: float = Field(ge=0.0, le=1.0)


class RecoveryContext(BaseModel):
    world_state: WorldState
    skill_result: SkillResult
    retry_count: int = Field(ge=0)
    observation: ExecutionObservation


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
