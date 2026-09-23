English | [日本語](README.ja.md)

# Physical AI Lab

A hands-on textbook and collection of small Python labs for learning Physical AI architecture: Vision-Language Model (VLM) perception and grounding, LLM-based skill planning, Robot Runtime boundaries, robot learning, System 1 recovery, and safety boundaries.

- Textbook: [`doc/textbook.md`](doc/textbook.md)
- End-to-end demo: [`main.py`](main.py)

## How this project was created

The textbook was generated with ChatGPT from the repository owner's history of learning and experimentation, then reviewed and edited for publication. Nearly all code was written by the repository owner, with occasional AI-assisted code completion and generation of simple logic, and the code has been reviewed by the owner. See [How this textbook was created](doc/textbook.md#how-this-textbook-was-created) for details.

The English edition is an adapted, condensed translation produced with assistance from ChatGPT and reviewed and edited by the repository owner. It preserves the full Day 1–5 curriculum and section numbering rather than translating every sentence literally. The [Japanese edition](doc/textbook.ja.md) is the source of truth.

## Important safety notice

This repository contains experimental software for educational and research purposes. It is not a safety-certified control system and provides no assurance of suitability for real robots or other safety-critical applications. Anyone connecting it to physical hardware is responsible for independent validation, appropriate safeguards such as speed, force, and collision limits and emergency stops, and compliance with applicable laws and standards.

The current `RobotRuntime` is a fake implementation. `SafetySupervisor` performs only minimal request-level validation; it does not implement runtime or control safety.

## Curriculum

| Day | Topic | Main code |
|---|---|---|
| 1 | VLMs, structured output, and grounding | `day1/`, `physical_ai/world_state.py`, `physical_ai/grounder.py` |
| 2 | System 2, skill planning, runtime, and timing | `physical_ai/agent.py`, `day2/rt_mini_lab.py` |
| 3 | Behavior Cloning, VLAs, action chunking, and ACT | `day3/` |
| 4 | Skill routing, System 1 recovery, and failure injection | `physical_ai/system1_recovery.py`, `day4/` |
| 5 | Request-level safety, evaluation, and Sim2Real | `physical_ai/safety_supervisor.py` |

## Implementation status

| Status | Scope |
|---|---|
| Implemented | Schemas, task parsing, grounding, skill planning, plan validation, and recovery routing |
| Fake implementation | PICK/PLACE and failure injection in `RobotRuntime` |
| Partial implementation | Request-level safety and a static `SkillRouter` |
| Not implemented | Actual repositioning, ROS 2/Gazebo/real-robot control, and runtime/control safety |
| External-service dependency | Gemini for perception, parsing, and planning; Jev for System 1 recovery |

## Setup

The project has been tested with Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

LeRobot has large dependencies and its examples download external datasets. If you only want to try Days 1, 2, and 4, you may install just the packages required by those labs.

### API keys

Configure credentials only when running examples that call external APIs.

```bash
cp .env.example .env
# Edit .env, then load it.
source source_env.bash
```

The Google GenAI SDK accepts `GEMINI_API_KEY` or `GOOGLE_API_KEY`. The hosted Jev examples require `TYPESAFE_API_KEY`. `.env` is ignored by Git. Never paste API keys into source files, logs, issues, or commits.

## Quick start

Run all commands from the repository root.

### 1. Generate reproducible scenes

`output/` contains generated artifacts and is ignored by Git. Generate the images referenced by the examples with:

```bash
python -m physical_ai.scene_generator --output output/scene_1.png --seed 1
python -m physical_ai.scene_generator --output output/scene_4.png --seed 4
```

The same seed produces the same image and ground-truth JSON. Omit `--seed` to generate a random scene.

### 2. Mini-labs without external APIs

```bash
python day2/rt_mini_lab.py
python day3/bc_mini_lab.py
python day3/bc_mini_lab_action_chunking.py
```

The Behavior Cloning examples open a Matplotlib window after training.

### 3. Day 1 with Gemini

Generate the scenes, configure a Gemini API key, and run:

```bash
python day1/vlm_world_state.py
python day1/vlm_task_spec.py
python day1/vlm_grounding.py
```

Model availability can vary by account, region, and date. If a model name in the code is unavailable, consult the Gemini API documentation for the models available to your account.

### 4. LeRobot and ACT

```bash
python day3/lerobot_dataset_lab.py
python day3/lerobot_act_lab.py
```

These examples download the PushT dataset and related files from Hugging Face. `lerobot_act_lab.py` uses CUDA and requires a compatible GPU and PyTorch environment.

### 5. System 1 recovery with Jev

```bash
python system1/jev_mini_lab.py
python day4/jev_recovery_benchmark.py
```

These commands call the external Jev API. The end-to-end demo calls both Gemini and Jev:

```bash
python main.py
```

Outputs may vary with the model, network, and date. The latency and decisions recorded in the textbook are observations from a small experiment, not performance guarantees.

## External services and data

Some examples use external services:

- [Google Gemini API](https://ai.google.dev/gemini-api/docs): perception, task parsing, grounding, and skill planning
- [TypeSafe AI Jev](https://www.typesafeai.org/tools): System 1 recovery decisions
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot): robot-learning datasets and policies

Images, prompts, structured world states, and other inputs may be sent to these services when an example runs. Before sending confidential, personal, or safety-critical data, review each provider's current terms, pricing, regional availability, and data-handling policies.

These services and datasets are not included in this repository's MIT License and remain subject to their own terms. This project is not affiliated with or endorsed by the providers.

## Third-party dependencies

Python dependencies are listed in [`requirements.txt`](requirements.txt). Each package remains subject to its own copyright and license; this repository's MIT License does not relicense dependencies, external models, datasets, or APIs.

If you distribute binaries, containers, or environments that bundle dependencies, review their licenses and include all required copyright notices and license texts.

## License

Unless otherwise noted, this repository's code and documentation are available under the [MIT License](LICENSE).
