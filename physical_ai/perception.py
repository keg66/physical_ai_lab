from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from .world_state import ObjectType, WorldState


class Perception:
    def __init__(
        self,
        image_path: str | Path,
        client: genai.Client | None = None,
        model: str = "gemini-3.5-flash-lite",
    ):
        self.client = client or genai.Client()
        self.model = model
        self.image_path = Path(image_path)

    def observe(self) -> WorldState:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                "Detect objects and obstacles (black ones) in image and transform WorldState",
                Image.open(self.image_path),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=WorldState
            ),
        )
        world_state: WorldState = response.parsed
        for i, obj in enumerate(world_state.objects):
            obj.id = f"{'obs' if obj.type == ObjectType.OBSTACLE else 'obj'}_{i}"
        return world_state
