from google import genai
from google.genai import types

from .task_spec import TaskSpec


class TaskParser:
    def __init__(
        self, client: genai.Client | None = None, model: str = "gemini-3.5-flash-lite"
    ):
        self.client = client or genai.Client()
        self.model = model

    def parse(self, instruction: str) -> TaskSpec:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[instruction],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=TaskSpec
            ),
        )
        return response.parsed
