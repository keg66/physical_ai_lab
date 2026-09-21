from pathlib import Path

from .grounder import Grounder
from .grounding_result import GroundingResults, GroundingStatus
from .perception import Perception
from .plan_validator import PlanValidator
from .skill_plan import SkillPlan
from .skill_planner import SkillPlanner
from .task_parser import TaskParser
from .task_spec import TaskSpec
from .world_state import WorldState


class Agent:
    def __init__(self, image_path: str | Path):
        self.task_parser = TaskParser()
        self.perception = Perception(image_path)
        self.grounder = Grounder()
        self.skill_planner = SkillPlanner()
        self.plan_validator = PlanValidator()

    def run(
        self, instruction: str
    ) -> tuple[TaskSpec, WorldState, GroundingResults, SkillPlan | None, bool | None]:
        task_spec = self.task_parser.parse(instruction)
        world_state = self.perception.observe()
        grounding_results = self.grounder.ground(task_spec, world_state)

        # Do not plan when any reference is unresolved or ambiguous.
        if not grounding_results.results or not all(
            result.status == GroundingStatus.RESOLVED
            for result in grounding_results.results
        ):
            return task_spec, world_state, grounding_results, None, None

        skill_plan = self.skill_planner.plan(task_spec, world_state, grounding_results)
        is_valid = self.plan_validator.validate(skill_plan, world_state)
        return task_spec, world_state, grounding_results, skill_plan, is_valid
