# test response models

from pydantic import BaseModel
from typing import Optional

class TestResponse(BaseModel):
    """
    An example response model to test Pydantic.
    """
    message: str
    list_msg: list[str]
