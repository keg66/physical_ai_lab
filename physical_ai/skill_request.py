from pydantic import BaseModel
from .skill_plan import SkillStep


class SkillRequest(BaseModel):
    step: SkillStep
    timeout_sec: float
