import os
import time

import requests

from physical_ai.system1_recovery import System1Recovery
from physical_ai.recovery import (
    ExecutionObservation,
    RecoveryAction,
    RecoveryContext,
)
from physical_ai.skill_result import (
    SkillResult,
    SkillResultReason,
    SkillResultStatus,
)
from physical_ai.world_state import (
    WorldState,
    WorldObject,
    ObjectType,
    ObjectColor,
    ObjectProperties,
    BoundingBox,
    Position,
)


# Establish the connection/model session before measuring benchmark cases.
# This request is intentionally excluded from the benchmark timings.
WARMUP_API_URL = os.getenv(
    "TYPESAFE_SYSTEMONE_API_URL", "https://api.typesafe.ai/v1/systemone"
)
# Warm up the same model used by System1Recovery.  "noul" is not an accepted
# System One model name and causes the API to return HTTP 400.
WARMUP_MODEL = os.getenv("TYPESAFE_SYSTEMONE_MODEL", "jev-latest")
WARMUP_TIMEOUT_SEC = 30


def warm_up_system1() -> None:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("TYPESAFE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        WARMUP_API_URL,
        headers=headers,
        json={
            "state": "ready?",
            "model": WARMUP_MODEL,
            "questions": {
                "ready": {
                    "type": "choice",
                    "instructions": "準備できているか答えてください",
                    "criteria": {
                        "yes": "準備できている",
                        "no": "まだ準備できていない",
                    },
                }
            },
        },
        timeout=WARMUP_TIMEOUT_SEC,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"System 1 warm-up failed: {response.status_code} {response.text}"
        ) from exc
    print(f"System 1 warm-up completed ({WARMUP_MODEL})")


world_state = WorldState(
    objects=[
        WorldObject(
            id="target-box",
            type=ObjectType.BOX,
            bbox=BoundingBox(
                center=Position(x=320, y=240),
                width=80,
                height=80,
            ),
            properties=ObjectProperties(color=ObjectColor.RED),
        )
    ]
)

failed_result = SkillResult(
    status=SkillResultStatus.FAILED,
    reason=SkillResultReason.ACTION_FAILED,
)


cases = [
    (
        "一時的なgrasp失敗",
        ExecutionObservation(
            target_visible=True,
            target_position_changed=False,
            target_reachable=True,
            contact_detected=True,
            grasp_quality=0.8,
        ),
        0,
        RecoveryAction.RETRY,
    ),
    (
        "対象が動いた",
        ExecutionObservation(
            target_visible=True,
            target_position_changed=True,
            target_reachable=True,
            contact_detected=False,
            grasp_quality=0.2,
        ),
        0,
        RecoveryAction.REOBSERVE_AND_REGROUND,
    ),
    (
        "現在姿勢では届かない",
        ExecutionObservation(
            target_visible=True,
            target_position_changed=False,
            target_reachable=False,
            contact_detected=False,
            grasp_quality=0.0,
        ),
        0,
        RecoveryAction.REPOSITION,
    ),
    (
        "何度も失敗している",
        ExecutionObservation(
            target_visible=True,
            target_position_changed=False,
            target_reachable=True,
            contact_detected=True,
            grasp_quality=0.3,
        ),
        3,
        RecoveryAction.ASK_SYSTEM2,
    ),
]


system1 = System1Recovery()

warm_up_system1()

correct = 0
total_latency_ms = 0.0

for name, observation, retry_count, expected in cases:
    context = RecoveryContext(
        world_state=world_state,
        skill_result=failed_result,
        retry_count=retry_count,
        observation=observation,
    )

    start = time.perf_counter()
    actual = system1.decide(context)
    latency_ms = (time.perf_counter() - start) * 1000

    ok = actual == expected
    correct += int(ok)
    total_latency_ms += latency_ms

    print(f"\n=== {name} ===")
    print(f"expected : {expected.value}")
    print(f"actual   : {actual.value}")
    print(f"match    : {ok}")
    print(f"latency  : {latency_ms:.1f} ms")


print("\n=== Summary ===")
print(f"matched       : {correct}/{len(cases)}")
print(f"mean latency  : {total_latency_ms / len(cases):.1f} ms")
