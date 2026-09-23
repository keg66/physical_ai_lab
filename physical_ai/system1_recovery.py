import os

import requests

from .recovery import RecoveryAction, RecoveryContext


API_URL = os.getenv(
    "TYPESAFE_SYSTEMONE_API_URL", "https://api.typesafe.ai/v1/systemone"
)
MODEL = os.getenv("TYPESAFE_SYSTEMONE_MODEL", "jev-latest")
TIMEOUT_SEC = 30


class System1Recovery:
    """Choose a recovery action using a System 1 model."""

    def decide(self, context: RecoveryContext) -> RecoveryAction:
        payload = {
            "state": self._describe_context(context),
            "model": MODEL,
            "questions": {
                "recovery_action": {
                    "type": "choice",
                    "instructions": (
                        "ロボットの安全とタスクの継続性を優先して、"
                        "最も適切な復旧行動を1つ選んでください"
                    ),
                    "criteria": {
                        "retry": "同じスキルをもう一度実行する。失敗が一時的な可能性がある場合。",
                        "reposition": "ロボットの位置や姿勢を調整してから再実行する。",
                        "reobserve_and_reground": "周囲を再観測し、対象の対応付けをやり直す。",
                        "ask_system2": "より深い計画・推論が必要なため、System 2 に判断を委ねる。",
                    },
                }
            },
        }

        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("TYPESAFE_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SEC,
        )
        response.raise_for_status()

        result = response.json()
        choice = result["answers"]["recovery_action"]["choice"]
        try:
            return RecoveryAction(choice)
        except ValueError as exc:
            raise ValueError(f"Unknown System 1 recovery action: {choice!r}") from exc

    @staticmethod
    def _describe_context(context: RecoveryContext) -> str:
        return (
            "ロボットのスキル実行に失敗した。"
            f"再試行回数は{context.retry_count}回。"
            f"実行結果: status={context.skill_result.status.value}, "
            f"reason={context.skill_result.reason.value}。"
            f"現在のワールド状態: {context.world_state.model_dump_json()}"
        )
