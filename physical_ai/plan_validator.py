from physical_ai.skill_plan import SkillPlan
from physical_ai.skill_type import SkillType
from physical_ai.world_state import WorldState

from pydantic import BaseModel


class RobotState(BaseModel):
    holding_object_id: str | None = None


def validate_plan(
    plan: SkillPlan,
    world_state: WorldState,
) -> bool:
    """Validate object references and robot-state transitions in ``plan``.

    The robot starts with empty hands.  The virtual state is updated after each
    valid step so that later steps are checked against the state they would
    actually encounter.
    """
    world_object_ids = {
        world_object.id
        for world_object in world_state.objects
        if world_object.id is not None
    }

    robot_state = RobotState()

    for step in plan.steps:
        parameters = step.parameter.model_dump()
        referenced_ids = (
            value for name, value in parameters.items() if name.endswith("_id")
        )
        if any(object_id not in world_object_ids for object_id in referenced_ids):
            return False

        if step.skill == SkillType.PICK:
            # PICK requires empty hands.
            if robot_state.holding_object_id is not None:
                return False
            robot_state.holding_object_id = step.parameter.target_id

        elif step.skill == SkillType.PLACE:
            # PLACE requires holding the object being placed.
            if robot_state.holding_object_id != step.parameter.object_id:
                return False
            robot_state.holding_object_id = None

    return True
