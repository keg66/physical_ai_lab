from pathlib import Path

from .grounder import Grounder
from .grounding_result import GroundingResults, GroundingStatus
from .perception import Perception
from .plan_validator import PlanValidator
from .robot_runtime import RobotRuntime
from .skill_plan import SkillPlan
from .skill_planner import SkillPlanner
from .skill_request import SkillRequest
from .skill_result import SkillResultStatus
from .task_parser import TaskParser
from .task_spec import TaskSpec
from .world_state import WorldState

import sys


class Agent:
    def __init__(self, image_path: str | Path):
        self.task_parser = TaskParser()
        self.perception = Perception(image_path)
        self.grounder = Grounder()
        self.skill_planner = SkillPlanner()
        self.plan_validator = PlanValidator()
        self.robot_runtime = RobotRuntime()

    def run(self, instruction: str) -> bool:
        task_spec = self.task_parser.parse(instruction)
        print(task_spec)
        print()
        world_state = self.perception.observe()
        print(world_state)
        print()
        grounding_results = self.grounder.ground(task_spec, world_state)
        print(grounding_results)
        print()

        # Do not plan when any reference is unresolved or ambiguous.
        if not grounding_results.results or not all(
            result.status == GroundingStatus.RESOLVED
            for result in grounding_results.results
        ):
            print("[Agent] Failed grounding", file=sys.stderr)
            return False

        skill_plan = self.skill_planner.plan(task_spec, world_state, grounding_results)
        print(skill_plan)
        print()

        is_valid = self.plan_validator.validate(skill_plan, world_state)
        if not is_valid:
            print("[Agent] Failed to create skill plan", file=sys.stderr)
            return False

        for skill_step in skill_plan.steps:
            request = SkillRequest(
                step=skill_step,
                timeout_sec=5.0,
            )
            result = self.robot_runtime.execute(request)

            if result.status != SkillResultStatus.SUCCEEDED:
                print("[Agent] Failed to execute plan", file=sys.stderr)
                return False

        print("[Agent] Succeeded")
        return True
