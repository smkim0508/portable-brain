# test request body

from pydantic import BaseModel
from typing import Optional

class TestRequest(BaseModel):
    """
    An example request body model to test Pydantic.
    """
    request_msg: str
    requested_num: Optional[int] = None # this field can be omitted
