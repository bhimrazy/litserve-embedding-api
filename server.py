import litserve as ls
from typing import List
from fastembed import TextEmbedding


class TextEmbeddingAPI(ls.LitAPI):
    def setup(self, device):
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.embedding_model = TextEmbedding()

    def decode_request(self, request) -> List[str]:
        request_input = request["input"]
        documents = [request_input] if isinstance(request_input, str) else request_input
        return documents

    def predict(self, documents) -> List[List[float]]:
        return [
            embedding.tolist() for embedding in self.embedding_model.embed(documents)
        ]

    def encode_response(self, output) -> dict:
        return {"embeddings": output, "model": self.model_name}


if __name__ == "__main__":
    embedding_api = TextEmbeddingAPI()
    server = ls.LitServer(embedding_api, api_path="/embeddings")
    server.run(port=8000)
