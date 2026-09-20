from enum import Enum

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    PLACE = "place"


class RelationType(str, Enum):
    NEXT_TO = "next_to"


class TaskSpec(BaseModel):
    action: ActionType
    object_ref: str
    target_ref: str
    relation: RelationType
