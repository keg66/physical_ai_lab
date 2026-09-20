from pydantic import BaseModel
from enum import Enum


class ProgressStatus(str, Enum):
    PROGRESSING = "progressing"
    BLOCKED = "blocked"


class ProgressReason(str, Enum):
    NONE = "none"
    TARGET_LOST = "target_lost"
    PATH_BLOCKED = "path_blocked"


class SkillFeedback(BaseModel):
    status: ProgressStatus
    reason: ProgressReason


class SkillResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    PRECONDITION_FAILED = "precondition_failed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class SkillResultReason(str, Enum):
    NONE = "none"
    TARGET_UNAVAILABLE = "target_unavailable"
    PATH_UNAVAILABLE = "path_unavailable"
    ACTION_FAILED = "action_failed"
    SAFETY_VIOLATION = "safety_violation"
    UNKNOWN = "unknown"


class SkillResult(BaseModel):
    status: SkillResultStatus
    reason: SkillResultReason
