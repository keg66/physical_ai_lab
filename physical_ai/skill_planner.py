from pathlib import Path

from google import genai
from google.genai import types

from .grounding_result import GroundingResults
from .skill_plan import SkillPlan
from .task_spec import TaskSpec
from .world_state import WorldState


class SkillPlanner:
    def __init__(
        self, client: genai.Client | None = None, model: str = "gemini-3.5-flash-lite"
    ):
        self.client = client or genai.Client()
        self.model = model
        self.skills = Path(__file__).with_name("skills.md").read_text()

    def plan(
        self,
        task_spec: TaskSpec,
        world_state: WorldState,
        grounding_results: GroundingResults,
    ) -> SkillPlan:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                f"""# Task
{task_spec.model_dump_json()}
# World State
{world_state.model_dump_json()}
# Grounding Results
{grounding_results.model_dump_json()}
# Available Skills
{self.skills}
# Instruction
上記のAvailable Skillsだけを使って目的を達成するSkillPlanを作成してください。"""
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=SkillPlan
            ),
        )
        return response.parsed
