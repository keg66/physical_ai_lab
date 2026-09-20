from google import genai
from google.genai import types
from PIL import Image

from world_state import ObjectType, WorldState
from grounding_result import GroundingResult

client = genai.Client()

image = Image.open("output/scene_4.png")

model = "gemini-3.5-flash-lite"

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

print("Simulate the tracker recognizing and assigning an ID.")
for i, obj in enumerate(world_state.objects):
    prefix = "obs" if obj.type == ObjectType.OBSTACLE else "obj"
    obj.id = f"{prefix}_{i}"
print(world_state)

world_state_json = response.parsed.model_dump_json()
response = client.models.generate_content(
    model=model,
    contents=[world_state_json, "一番左の赤い箱を抽出して"],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GroundingResult,
    ),
)

print(response.parsed)
