from google import genai
from google.genai import types

from .grounding_result import GroundingResult, GroundingResults, GroundingStatus
from .task_spec import TaskSpec
from .world_state import WorldState


class Grounder:
    def __init__(
        self, client: genai.Client | None = None, model: str = "gemini-3.5-flash-lite"
    ):
        self.client = client or genai.Client()
        self.model = model

    def ground(
        self, object_refs: TaskSpec, world_state: WorldState
    ) -> GroundingResults:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                world_state.model_dump_json(),
                object_refs.model_dump_json(),
                "関心のある物体をすべて抽出して",
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=GroundingResults
            ),
        )
        grounding_results: GroundingResults = response.parsed

        # Do not treat the grounding as successful when the LLM returns only one reference.
        # Add missing references as NOT_FOUND so that Agent stops before planning.
        required_refs = {object_refs.object_ref, object_refs.target_ref}
        returned_refs = {result.ref for result in grounding_results.results}
        missing_refs = required_refs - returned_refs
        for ref in missing_refs:
            grounding_results.results.append(
                GroundingResult(
                    ref=ref,
                    status=GroundingStatus.NOT_FOUND,
                    object=None,
                )
            )

        return grounding_results
