from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Optional, Dict
from portable_brain.common.logging.logger import logger
from portable_brain.config.app_config import get_service_settings
from portable_brain.core.lifespan import lifespan
from fastapi.middleware.cors import CORSMiddleware
from portable_brain.common.db.session import get_async_session_maker
from sqlalchemy import text
from portable_brain.agent_service.api.routes.test_route import router as test_router

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

# health endpoint
@app.get("/health")
async def health():
    # TODO: add more health checks w/ services

    main_db_engine = app.state.main_db_engine
    main_session_maker = get_async_session_maker(main_db_engine)

    try:
        # simple test query to verify db connection
        async with main_session_maker() as session:
            await session.execute(text("SELECT 1"))
    except Exception as e:
        logger.info(f"error: {e}")
        return {"status": "error", "database": "unable to connect to main database"}

    return {"status": "ok", "database": "connected to main database"}

app.include_router(test_router)