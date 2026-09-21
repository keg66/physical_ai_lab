from .skill_request import SkillRequest
from .skill_result import SkillResult, SkillResultReason, SkillResultStatus
from .skill_type import SkillType


class RobotRuntime:
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.step.skill == SkillType.PICK:
            return self._succeeded()

        if request.step.skill == SkillType.PLACE:
            return self._succeeded()

        return SkillResult(
            status=SkillResultStatus.FAILED,
            reason=SkillResultReason.UNKNOWN,
        )

    @staticmethod
    def _succeeded() -> SkillResult:
        return SkillResult(
            status=SkillResultStatus.SUCCEEDED,
            reason=SkillResultReason.NONE,
        )
