from pydantic import BaseModel

from .skill_type import SkillType

from .skill_request import SkillRequest
from .world_state import WorldState


class SafetyDecision(BaseModel):
    allowed: bool
    reason: str | None = None


class SafetySupervisor:
    def check(
        self,
        request: SkillRequest,
        world_state: WorldState,
    ) -> SafetyDecision:
        # example safety check: if the skill is PICK, ensure the target object is present in the world state
        if request.step.skill == SkillType.PICK:
            target_id = request.step.parameter.target_id
            if not any(obj.id == target_id for obj in world_state.objects):
                return SafetyDecision(
                    allowed=False,
                    reason="target_not_found",
                )

        return SafetyDecision(allowed=True)
