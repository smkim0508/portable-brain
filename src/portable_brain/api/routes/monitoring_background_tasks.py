# background async tasks for monitoring
import time
import asyncio
from fastapi import APIRouter, Depends, Query
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.monitoring.background_tasks.observation_tracker import ObservationTracker
from portable_brain.common.logging.logger import logger
from portable_brain.core.dependencies import get_droidrun_client, get_observation_tracker

router = APIRouter(prefix="/monitoring/background-tasks", tags=["Monitoring Background Tasks"])

@router.get("/start")
async def start_observation_tracking(
    poll_interval: float = Query(default=1.0, gt=0.0),
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker)
):
    try:
        tracking_task = observation_tracker.start_background_tracking(poll_interval=poll_interval)
        logger.info(f"Observation tracking task started: {tracking_task}")
        return {"message": "successfully started background observation tracking!"}
    except Exception as e:
        logger.error(f"Error starting observation tracking task: {e}")
        return {"message": f"Error starting observation tracking task: {e}"}, 500

@router.get("/stop")
async def stop_observation_tracking(
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
):
    try:
        await observation_tracker.stop_tracking()
        return {"message": "successfully stopped background observation tracking!"}
    except Exception as e:
        logger.error(f"Error stopping observation tracking task: {e}")
        return {"message": f"Error stopping observation tracking task: {e}"}, 500

@router.get("/clear")
def clear_observations(
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
):
    try:
        observation_tracker.clear_observations()
        return {"message": "successfully cleared observation history!"}
    except Exception as e:
        logger.error(f"Error clearing observation history: {e}")
        return {"message": f"Error clearing observation history: {e}"}, 500
    
@router.get("/get-observations")
def retrieve_observations(
    limit: int = Query(default=5, ge=1, le=100),
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
):
    try:
        # NOTE: only retrieve the most recent observations by limit
        logger.info(f"Retrieving observation history with limit: {limit}")
        observations = observation_tracker.get_observations(limit=limit)
        return {"observations": observations}, 200
    except Exception as e:
        logger.error(f"Error retrieving observation history: {e}")
        return {"message": f"Error retrieving observation history: {e}"}, 500
