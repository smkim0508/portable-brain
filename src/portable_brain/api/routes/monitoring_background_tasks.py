# background async tasks for monitoring
import time
import asyncio
from fastapi import APIRouter, Depends
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.monitoring.background_tasks.observation_tracker import ObservationTracker
from portable_brain.common.logging.logger import logger
from portable_brain.core.dependencies import get_droidrun_client, get_observation_tracker

router = APIRouter(prefix="/monitoring/background-tasks", tags=["Monitoring Background Tasks"])

@router.get("/start")
async def start_observation_tracking(
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
):
    try:
        tracking_task = observation_tracker.start_background_tracking()
        logger.info(f"Observation tracking task started: {tracking_task}")
        return {"message": "successfully started background observation tracking!"}
    except Exception as e:
        logger.error(f"Error starting observation tracking task: {e}")
        return {"message": f"Error starting observation tracking task: {e}"}, 500
