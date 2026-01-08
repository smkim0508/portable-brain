from pydantic import BaseModel
from typing import Union, Literal, Optional
from enum import Enum
from datetime import datetime, timezone, timedelta
import time

class UIState(BaseModel):
    """
    Canonical representation of Android UI State.
    - A11Y tree output translated into portable format.
    """
    pass