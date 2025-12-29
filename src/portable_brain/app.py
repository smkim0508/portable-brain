from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Optional, Dict
from portable_brain.common.logging.logger import logger
from portable_brain.config.app_config import get_service_settings
from portable_brain.core.lifespan import lifespan
from fastapi.middleware.cors import CORSMiddleware

# disable FastAPI docs for production/deployment
is_local = get_service_settings().INCLUDE_DOCS
logger.info(f"is_local (include FastAPI docs?): {is_local}")

docs_config: dict[str, Any] = {
    "docs_url": "/docs" if is_local else None,
    "redoc_url": "/redoc" if is_local else None,
    "openapi_url": "/openapi.json" if is_local else None,
}

# main app, asgi entrypoint
app = FastAPI(
    title="Portable Brain Service",
    description="Experimental service for testing different memroy structures to observe HCI data",
    version="0.1.0",
    lifespan=lifespan,
    **docs_config,  
)

# add CORS middleware
# TODO: make this more restrictive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# test endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}
