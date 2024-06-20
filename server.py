# server.py
import logging
from typing import List, Literal, Union

import torch
import litserve as ls
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# Define allowed embedding models
EMBEDDING_MODELS = Literal["jina-embeddings-v2-small-en", "jina-embeddings-v2-base-en"]

MODEL_MAPPING = {
    "jina-embeddings-v2-small-en": "jinaai/jina-embeddings-v2-small-en",  # seq len = 8192, dim = 768
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",  # seq len = 8192, dim = 768
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",  # seq len = 512, dim = 384
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",  # seq len = 512, dim = 384
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",  # seq len = 512, dim = 768
    "nomic-embed-text-v1": "nomic-ai/nomic-embed-text-v1",  # seq len = 8192, dim = 768
}


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


BERT_CLASSES = ["NomicBertModel", "BertModel"]


class OpeanAIEmbeddingAPI(ls.LitAPI):
    def setup(self, device, model_id="jina-embeddings-v2-small-en"):
        """Setup the model and tokenizer."""
        logging.info(f"Loading model: {model_id}")
        self.model_id = model_id
        self.model_name = MODEL_MAPPING[model_id]
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name).to(device)

    def decode_request(self, request: EmbeddingRequest, context) -> List[str]:
        """Decode the incoming request and prepare it for prediction."""
        context["model"] = request.model

        # load model if different from the active model
        if request.model != self.model_id:
            self.setup(self.device, request.model)

        sentences = [request.input] if isinstance(request.input, str) else request.input
        context["total_tokens"] = sum(
            len(self.tokenizer.encode(text)) for text in sentences
        )
        return sentences

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def predict(self, x) -> List[List[float]]:
        is_bert_instance = self.model.__class__.__name__ in BERT_CLASSES
        if is_bert_instance:
            encoded_input = self.tokenizer(
                x, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                return (
                    self.mean_pooling(model_output, encoded_input["attention_mask"])
                    .cpu()
                    .numpy()
                )

        return self.model.encode(x)

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
    # TODO: Convert the API to OpenAIEmbedding Spec
    api = OpeanAIEmbeddingAPI()
    server = ls.LitServer(api, accelerator="auto", api_path="/v1/embeddings")
    server.run(port=8000)
