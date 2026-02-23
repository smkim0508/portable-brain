# background async tasks for monitoring
import time
import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, Query
from portable_brain.common.services.droidrun_tools.droidrun_client import DroidRunClient
from portable_brain.monitoring.background_tasks.observation_tracker import ObservationTracker
from portable_brain.common.logging.logger import logger
from portable_brain.core.dependencies import get_droidrun_client, get_observation_tracker

# monitoring DTOs
from portable_brain.monitoring.background_tasks.types.ui_states.state_changes import UIStateChange
from portable_brain.monitoring.background_tasks.types.action.actions import Action

router = APIRouter(prefix="/monitoring/background-tasks", tags=["Monitoring Background Tasks"])

@router.post("/start")
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

@router.post("/stop")
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

# observation history
@router.post("/clear-observations")
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
    limit: Optional[int] = Query(default=None, ge=1, le=100),
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

# recent UI state change history
@router.post("/clear-state-changes")
def clear_state_changes(
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
):
    try:
        observation_tracker.clear_state_changes()
        return {"message": "successfully cleared recent UI state change history!"}
    except Exception as e:
        logger.error(f"Error clearing UI state change history: {e}")
        return {"message": f"Error clearing UI state change history: {e}"}, 500
    
@router.get("/get-recent-state-changes")
def retrieve_recent_state_changes(
    limit: Optional[int] = Query(default=None, ge=1, le=10),
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
):
    try:
        # NOTE: only retrieve the most recent state changes by limit
        logger.info(f"Retrieving recent UI state change history with limit: {limit}")
        state_changes = observation_tracker.get_state_changes(limit=limit)
        return {"state_changes": state_changes}, 200
    except Exception as e:
        logger.error(f"Error retrieving recent state change history: {e}")
        return {"message": f"Error retrieving recent state change history: {e}"}, 500

# state snapshots history
@router.post("/clear-state-snapshots")
def clear_state_snapshots(
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
):
    try:
        observation_tracker.clear_state_snapshots()
        return {"message": "successfully cleared state snapshots history!"}
    except Exception as e:
        logger.error(f"Error clearing state snapshots history: {e}")
        return {"message": f"Error clearing state snapshots history: {e}"}, 500
    
@router.get("get-state-snapshots")
def retrieve_state_snapshots(
    limit: Optional[int] = Query(default=None, ge=1, le=10),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
):
    try:
        # NOTE: only retrieve the most recent observations by limit
        logger.info(f"Retrieving state snapshots with limit: {limit}")
        snapshots = observation_tracker.get_state_snapshots(limit=limit)
        return {"snapshots": snapshots}, 200
    except Exception as e:
        logger.error(f"Error retrieving state snapshots: {e}")
        return {"message": f"Error retrieving state snapshots: {e}"}, 500

@router.get("/monitoring-overview")
def retrieve_monitoring_overview(
    droidrun_client: DroidRunClient = Depends(get_droidrun_client),
    observation_tracker: ObservationTracker = Depends(get_observation_tracker),
):
    try:
        overview = observation_tracker.get_monitoring_overview()
        return {"overview": overview}, 200
    except Exception as e:
        logger.error(f"Error retrieving monitoring overview: {e}")
        return {"message": f"Error retrieving monitoring overview: {e}"}, 500
