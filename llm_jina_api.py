import base64
import httpx
import llm
from PIL import Image
import io



@llm.hookimpl
def register_embedding_models(register):
    register(JinaClipEmbeddingModel())


class JinaClipEmbeddingModel(llm.EmbeddingModel):
    model_id = "jina-clip-v1-api"
    needs_key = "jina"
    key_env_var = "JINA_API_KEY"
    supports_binary = True
    supports_text = True

    def embed_batch(self, items):
        input = []

        for item in items:
            if isinstance(item, bytes):
                # If the item is a byte string, resize the image and add the base64 encoded string to the list

                # Open the image from the byte string
                image = Image.open(io.BytesIO(item))

                # Resize the image to a maximum of 896 high or 896 wide
                image.thumbnail((896, 896))

                # Convert the resized image to bytes
                resized_image_bytes = io.BytesIO()
                image.save(resized_image_bytes, format='JPEG')
                resized_image_bytes.seek(0)

                # Encode the resized image as base64
                encoded = base64.b64encode(resized_image_bytes.read()).decode()
                input.append({"image": encoded})
            elif isinstance(item, str):
                input.append({"text": item})

        response = httpx.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(self.get_key())
            },
            json={
                "model": "jina-clip-v1",
                "normalized": False,
                "embedding_type": "float",
                "input": input
            }
        )
        print(input)
        response.raise_for_status()
        embeddings = response.json()["data"]
        return [embedding["embedding"] for embedding in embeddings]
