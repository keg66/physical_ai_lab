from pydantic import BaseModel
from .recovery import ExecutionObservation

from .skill_request import SkillRequest
from .skill_result import SkillResult, SkillResultReason, SkillResultStatus
from .skill_type import SkillType


class SkillExecutionOutcome(BaseModel):
    result: SkillResult
    observation: ExecutionObservation | None = None


class RobotRuntime:
    def __init__(self):
        self.pick_attempt_count: int = 0

    def execute(self, request: SkillRequest) -> SkillExecutionOutcome:
        if request.step.skill == SkillType.PICK:
            self.pick_attempt_count += 1
            if self.pick_attempt_count == 1:  # fake a failure on the first pick attempt
                return SkillExecutionOutcome(
                    result=SkillResult(
                        status=SkillResultStatus.FAILED,
                        reason=SkillResultReason.ACTION_FAILED,
                    ),
                    observation=ExecutionObservation(
                        target_visible=True,
                        target_position_changed=False,
                        target_reachable=True,
                        contact_detected=True,
                        grasp_quality=0.8,
                    ),
                )
            return SkillExecutionOutcome(result=self._succeeded())

        if request.step.skill == SkillType.PLACE:
            return SkillExecutionOutcome(result=self._succeeded())

        return SkillExecutionOutcome(
            result=SkillResult(
                status=SkillResultStatus.FAILED,
                reason=SkillResultReason.UNKNOWN,
            )
        )

    @staticmethod
    def _succeeded() -> SkillResult:
        return SkillResult(
            status=SkillResultStatus.SUCCEEDED,
            reason=SkillResultReason.NONE,
        )
