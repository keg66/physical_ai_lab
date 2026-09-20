from typing import Literal

from pydantic import BaseModel

from .skill_type import PickParameters, PlaceParameters, SkillType


class PickSkillStep(BaseModel):
    skill: Literal[SkillType.PICK]
    parameter: PickParameters


class PlaceSkillStep(BaseModel):
    skill: Literal[SkillType.PLACE]
    parameter: PlaceParameters


SkillStep = PickSkillStep | PlaceSkillStep


class SkillPlan(BaseModel):
    steps: list[SkillStep]
