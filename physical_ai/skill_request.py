from pydantic import BaseModel
from .skill_type import PickParameters, PlaceParameters, SkillType


class SkillRequest(BaseModel):
    skill: SkillType
    parameter: PickParameters | PlaceParameters
    precondition: str
    success_condition: str
    timeout_sec: float
