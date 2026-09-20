from enum import Enum

from pydantic import BaseModel, Field

from world_state import WorldObject


class GroundingStatus(str, Enum):
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


class GroundingResult(BaseModel):
    status: GroundingStatus
    object: WorldObject | None
    candidates: list[WorldObject] = Field(default_factory=list)
