from google import genai
from google.genai import types

from physical_ai.task_spec import TaskSpec

client = genai.Client()

response = client.models.generate_content(
    # model="gemini-3.8-flash",
    # model="gemini-3.5-flash",
    model="gemini-3.5-flash-lite",
    contents=["赤い箱を青い箱の隣りに置いて"],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TaskSpec,
    ),
)

print(response.parsed)
