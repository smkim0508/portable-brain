# test response models

from pydantic import BaseModel
from typing import Optional

class TestResponse(BaseModel):
    """
    An example response model to test Pydantic.
    """
    message: str
    list_msg: list[str]

class SimilarEmbeddingResponse(BaseModel):
    """
    Response model for the closest embedding similarity search.
    """
    closest_text: str
    cosine_similarity_distance: float
    target_embedding: list[float]
    closest_embedding: list[float]
