import time

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
