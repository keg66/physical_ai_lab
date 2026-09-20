from enum import Enum

from pydantic import BaseModel, Field


class ObjectType(str, Enum):
    BOX = "box"
    OBSTACLE = "obstacle"


class ObjectColor(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    BLACK = "black"


class Position(BaseModel):
    """A point in image coordinates."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class BoundingBox(BaseModel):
    """A bounding box represented by its center and its dimensions."""

    center: Position
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ObjectProperties(BaseModel):
    color: ObjectColor


class WorldObject(BaseModel):
    id: str | None = None
    type: ObjectType
    bbox: BoundingBox
    properties: ObjectProperties


class WorldState(BaseModel):
    objects: list[WorldObject] = Field(default_factory=list)
