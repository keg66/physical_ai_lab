from google import genai
from google.genai import types
from PIL import Image

from world_state import WorldState

client = genai.Client()

image = Image.open("output/scene_1.png")

response = client.models.generate_content(
    # model="gemini-3.8-flash",
    # model="gemini-3.5-flash",
    model="gemini-3.5-flash-lite",
    contents=["Detect objects and obstacles in image and transform WorldState", image],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=WorldState,
    ),
)

print(response.parsed)