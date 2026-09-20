from pydantic import BaseModel
from enum import Enum


class SkillType(str, Enum):
    PICK = "pick"
    PLACE = "place"


class PickParameters(BaseModel):
    target_id: str


class RelationType(str, Enum):
    NEXT_TO = "next_to"


class PlaceParameters(BaseModel):
    object_id: str
    relation: RelationType
    reference_id: str
