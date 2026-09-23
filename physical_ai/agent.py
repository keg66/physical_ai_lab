from pathlib import Path

from .grounder import Grounder
from .grounding_result import GroundingResults, GroundingStatus
from .perception import Perception
from .plan_validator import PlanValidator
from .robot_runtime import RobotRuntime
from .recovery import RecoveryAction, RecoveryContext, RecoveryRouter
from .skill_plan import SkillPlan
from .skill_planner import SkillPlanner
from .skill_request import SkillRequest
from .skill_result import SkillResultStatus
from .system1_recovery import System1Recovery
from .task_parser import TaskParser
from .task_spec import TaskSpec
from .world_state import WorldState

import sys


class Agent:
    def __init__(self, image_path: str | Path, max_retries: int = 1):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.task_parser = TaskParser()
        self.perception = Perception(image_path)
        self.grounder = Grounder()
        self.skill_planner = SkillPlanner()
        self.plan_validator = PlanValidator()
        self.robot_runtime = RobotRuntime()
        self.recovery_router = RecoveryRouter()
        self.system1_recovery = System1Recovery()
        self.max_retries = max_retries

    def run(self, instruction: str) -> bool:
        task_spec = self.task_parser.parse(instruction)
        print(task_spec)
        print()
        retry_count = 0

        while True:
            # A retry starts at perception and reruns grounding, planning, and
            # validation against the newly observed world state.
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

            skill_plan = self.skill_planner.plan(
                task_spec, world_state, grounding_results
            )
            print(skill_plan)
            print()

            is_valid = self.plan_validator.validate(skill_plan, world_state)
            if not is_valid:
                print("[Agent] Failed to create skill plan", file=sys.stderr)
                return False

            should_retry = False
            execution_failed = False
            for skill_step in skill_plan.steps:
                request = SkillRequest(
                    step=skill_step,
                    timeout_sec=5.0,
                )
                while True:
                    outcome = self.robot_runtime.execute(request)

                    if outcome.result.status == SkillResultStatus.SUCCEEDED:
                        execution_failed = False
                        break

                    execution_failed = True
                    print("[Agent] Failed to execute plan", file=sys.stderr)
                    recovery_action = self.recovery_router.decide_recovery(
                        outcome.result
                    )
                    if recovery_action is not None:
                        print(f"[Agent] Recovery action: {recovery_action.value}")

                    if recovery_action == RecoveryAction.ASK_SYSTEM1:
                        if outcome.observation is None:
                            print(
                                "[Agent] Missing execution observation for System 1 recovery",
                                file=sys.stderr,
                            )
                            break

                        recovery_action = self.system1_recovery.decide(
                            RecoveryContext(
                                world_state=world_state,
                                skill_result=outcome.result,
                                retry_count=retry_count,
                                observation=outcome.observation,
                            )
                        )
                        print(
                            f"[Agent] System 1 recovery action: {recovery_action.value}"
                        )

                    if recovery_action not in {
                        RecoveryAction.RETRY,
                        RecoveryAction.REPOSITION,
                        RecoveryAction.REOBSERVE_AND_REGROUND,
                    }:
                        break

                    if retry_count >= self.max_retries:
                        print("[Agent] Retry limit reached", file=sys.stderr)
                        break

                    retry_count += 1
                    print(f"[Agent] Retrying ({retry_count}/{self.max_retries})")

                    if recovery_action == RecoveryAction.REOBSERVE_AND_REGROUND:
                        should_retry = True
                        break

                    if recovery_action == RecoveryAction.RETRY:
                        print("[Agent] Retrying the same SkillRequest")
                        continue

                    if recovery_action == RecoveryAction.REPOSITION:
                        print("TODO: Reposition skill/action here")
                        continue

                if should_retry or execution_failed:
                    break

            if should_retry:
                continue

            if not execution_failed:
                print("[Agent] Succeeded")
                return True

            print("[Agent] Failed", file=sys.stderr)
            return False
