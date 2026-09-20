from google import genai
from google.genai import types
from PIL import Image
from pathlib import Path

from physical_ai.world_state import ObjectType, WorldState
from physical_ai.grounding_result import GroundingResults
from physical_ai.task_spec import TaskSpec
from physical_ai.skill_plan import SkillPlan

client = genai.Client()

image = Image.open("output/scene_1.png")

model = "gemini-3.5-flash-lite"

print("############ TaskSpec #############")
response = client.models.generate_content(
    model=model,
    contents=["赤い箱を青い箱の隣りに置いて"],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TaskSpec,
    ),
)

task_spec = response.parsed
print(task_spec)
print()

print("############ WorldState #############")
response = client.models.generate_content(
    model=model,
    contents=[
        "Detect objects and obstacles (black ones) in image and transform WorldState",
        image,
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=WorldState,
    ),
)

world_state = response.parsed
print(world_state)
print()
print("Simulate the tracker recognizing and assigning an ID.")
for i, obj in enumerate(world_state.objects):
    prefix = "obs" if obj.type == ObjectType.OBSTACLE else "obj"
    obj.id = f"{prefix}_{i}"
print(world_state)
print()

print("############ Grounding #############")
world_state_json = response.parsed.model_dump_json()
response = client.models.generate_content(
    model=model,
    contents=[
        world_state_json,
        task_spec.model_dump_json(),
        "関心のある物体をすべて抽出して",
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GroundingResults,
    ),
)

grounding_result = response.parsed
print(grounding_result)
print()

print("############ Skill Planner #############")
skills = Path("physical_ai/skills.md").read_text()
response = client.models.generate_content(
    model=model,
    contents=[
        f"""
# Task
{task_spec.model_dump_json()}

# Grounding Results
{grounding_result.model_dump_json()}

# Available Skills
{skills}

# Instruction
上記のAvailable Skillsだけを使って目的を達成するSkillPlanを作成してください。
"""
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SkillPlan,
    ),
)

print(response.parsed)
