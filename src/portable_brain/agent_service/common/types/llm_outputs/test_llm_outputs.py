# simple test llm outputs to verify functionality

from pydantic import BaseModel
from typing import Optional

# simply outputs whether connection is true
class TestLLMOutput(BaseModel):
    connection: bool