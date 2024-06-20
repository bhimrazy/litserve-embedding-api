from litserve.specs.base import LitSpec
from fastapi import Request, Response


# TODO: Implement the OpenAIEmbeddingSpec class
class OpenAIEmbeddingSpec(LitSpec):
    def __init__(self):
        super().__init__()
        # register the endpoint
        self.add_endpoint("/v1/embeddings", self.embedding_generation, ["POST"])
        self.add_endpoint(
            "/v1/embeddings", self.options_embedding_generation, ["OPTIONS"]
        )

    async def options_chat_completions(self, request: Request):
        return Response(status_code=200)
