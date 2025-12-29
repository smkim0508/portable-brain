from typing import Any, Optional
from portable_brain.common.logging.logger import logger
from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncEngine

# This is the location to conveniently return any app lifetime dependencies to be used in routes
# TODO: add more dependencies as needed
def get_main_db_engine(request: Request) -> AsyncEngine:
    """
    FastAPI dependency to get the shared main DB engine from the application state.
    """
    return request.app.state.main_db_engine
