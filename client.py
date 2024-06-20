"""
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The food was delicious and the waiter...",
    "model": "jina-embeddings-v2-small-en",
    "encoding_format": "float"
  }'
"""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1/", api_key="lit")

response = client.embeddings.create(
    model="jina-embeddings-v2-small-en",
    input="The food was delicious and the waiter...",
    encoding_format="float",
)

print(response)
