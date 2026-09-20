from pydantic import BaseModel
from enum import Enum


class SkillType(str, Enum):
    PICK = "pick"


class SkillRequest(BaseModel):
    skill: SkillType
    target_id: str
    precondition: str
    success_condition: str
    timeout_sec: float
