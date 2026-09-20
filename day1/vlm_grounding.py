from google import genai
from google.genai import types
from PIL import Image

from world_state import WorldState
from grounding_result import GroundingResult

client = genai.Client()

image = Image.open("output/scene_4.png")

model = "gemini-3.5-flash-lite"

response = client.models.generate_content(
    model=model,
    contents=["Detect objects and obstacles in image and transform WorldState", image],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=WorldState,
    ),
)

world_state = response.parsed.model_dump_json()
print(world_state)

response = client.models.generate_content(
    model=model,
    contents=[world_state, "一番左の赤い箱を抽出して"],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GroundingResult,
    ),
)

print(response.parsed)
