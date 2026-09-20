"""Nebius Token Factory client (OpenAI-compatible)."""

import base64
import io
from functools import lru_cache

from openai import OpenAI
from PIL import Image

from kin import config


@lru_cache(maxsize=1)
def client() -> OpenAI:
    return OpenAI(api_key=config.nebius_api_key(), base_url=config.nebius_base_url())


def list_models() -> list[str]:
    return sorted(m.id for m in client().models.list())


def embed(text: str, is_query: bool = False) -> list[float]:
    """Embedding for semantic memory search. Queries get the instruction prefix Qwen3-Embedding expects."""
    if is_query:
        text = f"Instruct: Given a question, retrieve personal facts that help answer it\nQuery: {text}"
    return client().embeddings.create(model=config.embed_model(), input=text).data[0].embedding


def _to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def describe_image(image: Image.Image, prompt: str, max_tokens: int = 200) -> str:
    response = client().chat.completions.create(
        model=config.vlm_model(),
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()
