# test request body

from pydantic import BaseModel
from typing import Optional

class TestRequest(BaseModel):
    """
    An example request body model to test Pydantic.
    """
    request_msg: str
    requested_num: Optional[int] = None # this field can be omitted

class TestEmbeddingRequest(BaseModel):
    """
    Request body for testing embedding model.
    """
    embedding_text: str
    observation_id: str

class SimilarEmbeddingRequest(BaseModel):
    """
    Request body for finding the most similar embedding in the DB.
    """
    target_text: str

class SaveObservationRequest(BaseModel):
    """
    Request body for saving a mocked observation to structured memory.
    """
    observation_node: str
