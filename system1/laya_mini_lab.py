import laya


MODEL_NAME = "convaiinnovations/laya-multilingual"


def main() -> None:
    agent = laya.load(MODEL_NAME)

    state = (
        "移動ロボットはオフィスの廊下を走行中。"
        "進行方向上に2人の人が向かい合って会話している。"
        "2人の間にはロボットが安全に通過できる十分な幅がある。"
        "また、2人の外側にも安全に迂回できる十分なスペースがある。"
        "2人は会話に集中しており、ロボットにはまだ気づいていない。"
        "どちらの経路を選んでも目的地には到達できる。"
    )

    questions = {
        "route_choice": {
            "type": "choice",
            "instructions": "社会的に自然な行動を選んでください",
            "criteria": {
                "between_people": (
                    "2人の間を通る。"
                    "最短経路だが、会話を遮る可能性がある。"
                ),
                "go_around": (
                    "2人の外側を迂回する。"
                    "少し遠回りだが、会話を邪魔しにくい。"
                ),
                "wait": (
                    "その場で待つ。"
                    "会話を邪魔しないが、目的地への到着が遅れる。"
                ),
            },
        }
    }

    result = agent.predict(state, questions)

    print("=== Raw result ===")
    print(result)
    print()

    answer = result["answers"]["route_choice"]

    print(f"Model: {MODEL_NAME}")
    print()
    print("=== System 1 route decision ===")
    print(f"Choice     : {answer['choice']}")
    print(f"Confidence : {answer['confidence']:.2f}")
    print()

    print("=== Probabilities ===")
    for name, probability in sorted(
        answer["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name:15}: {probability:.2f}")


if __name__ == "__main__":
    main()

