from pydantic import BaseModel

from .skill_type import PickParameters, PlaceParameters, SkillType


class SkillStep(BaseModel):
    skill: SkillType
    parameter: PickParameters | PlaceParameters


class SkillPlan(BaseModel):
    steps: list[SkillStep]
