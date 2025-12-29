# test route for router + dependency injection

from fastapi import APIRouter, Depends
from portable_brain.common.logging.logger import logger
from portable_brain.common.db.session import get_async_session_maker
from portable_brain.core.dependencies import get_main_db_engine
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

router = APIRouter(prefix="/test", tags=["Tests"])

@router.get("/first-test")
async def first_test():
    logger.info("first test route")
    return {"message": "this is the first test route"}

@router.get("/second-test")
async def second_test(main_db_engine: AsyncEngine = Depends(get_main_db_engine)):
    logger.info("second test route, trying to inject db session")
    main_session_maker = get_async_session_maker(main_db_engine)

    try:
        async with main_session_maker() as session:
            await session.execute(text("SELECT 1"))
    except Exception as e:
        logger.info(f"error: {e}")
        return {"message": "unable to inject db session"}

    logger.info(f"db session successfully injected, dummy query passed")
    return {"message": "db session successfully injected, dummy query passed"}
