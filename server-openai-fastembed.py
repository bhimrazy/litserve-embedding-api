# server.py
import logging
from typing import List, Literal, Union

import litserve as ls
from fastembed import TextEmbedding
from pydantic import BaseModel
from transformers import AutoTokenizer

# Define allowed embedding models using Literal and dictionary keys
SUPPORTED_MODELS = [model["model"] for model in TextEmbedding.list_supported_models()]
EMBEDDING_MODELS = Literal[tuple(SUPPORTED_MODELS)]  # type: ignore


# Request model for embedding
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]  # Input text or list of texts to be embedded
    model: EMBEDDING_MODELS  # Model to use for embedding
    encoding_format: Literal["float"]  # Format of the embedding


# Model to represent a single embedding
class Embedding(BaseModel):
    embedding: List[float]  # Embedding vector
    index: int  # Index of the embedding in the input list
    object: Literal["embedding"] = "embedding"  # Type of object


# Model to represent usage statistics
class Usage(BaseModel):
    prompt_tokens: int  # Number of tokens in the prompt
    total_tokens: int  # Total number of tokens processed


# Response model for embedding request
class EmbeddingResponse(BaseModel):
    data: List[Embedding]  # List of embeddings
    model: EMBEDDING_MODELS  # Model used for generating embeddings
    object: Literal["list"] = "list"  # Type of object
    usage: Usage  # Usage statistics


class EmbeddingAPI(ls.LitAPI):
    def setup(self, device, model_name="jinaai/jina-embeddings-v2-small-en"):
        """Setup the model and tokenizer."""
        logging.info(f"Loading model: {model_name}")
        self.model_name = model_name
        providers = ["CUDAExecutionProvider"] if device == "cuda" else None
        self.model = TextEmbedding(model_name=self.model_name, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def decode_request(self, request: EmbeddingRequest, context) -> List[str]:
        """Decode the incoming request and prepare it for prediction."""
        context["model"] = request.model

        # load model if different from the active model
        if request.model != self.model_name:
            self.setup(self.device, request.model)

        documents = [request.input] if isinstance(request.input, str) else request.input

        context["total_tokens"] = sum(
            len(self.tokenizer.encode(text)) for text in documents
        )
        return documents

    def predict(self, documents) -> List[List[float]]:
        return list(self.model.embed(documents))  # type: ignore

    def encode_response(self, output, context) -> EmbeddingResponse:
        """Encode the embedding output into the response model."""
        embeddings = [
            Embedding(embedding=embedding, index=i)
            for i, embedding in enumerate(output)
        ]
        return EmbeddingResponse(
            data=embeddings,
            model=context["model"],
            usage=Usage(
                prompt_tokens=context["total_tokens"],
                total_tokens=context["total_tokens"],
            ),
        )


if __name__ == "__main__":
    api = EmbeddingAPI()
    server = ls.LitServer(api, accelerator="auto", api_path="/v1/embeddings")

    # add soute to display supported models
    server.app.add_api_route(
        "/v1/embeddings/models",
        TextEmbedding.list_supported_models,  # type: ignore
        methods=["GET"],
    )
    server.run(port=8000)
