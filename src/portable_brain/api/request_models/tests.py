# test request body

from pydantic import BaseModel
from typing import Optional
from enum import Enum

class TestRequest(BaseModel):
    """
    An example request body model to test Pydantic.
    """
    request_msg: str
    requested_num: Optional[int] = None # this field can be omitted

class TestEmbeddingRequest(BaseModel):
    """
    Request body for testing embedding model.
    """
    embedding_text: str
    observation_id: str

class SimilarEmbeddingRequest(BaseModel):
    """
    Request body for finding the most similar embedding in the DB.
    """
    target_text: str

class SaveObservationRequest(BaseModel):
    """
    Request body for saving a mocked observation to structured memory.
    """
    observation_node: str

class ScenarioName(str, Enum):
    INSTAGRAM_CLOSE_FRIEND_MESSAGING = "instagram_close_friend_messaging"
    MORNING_WORK_APP_ROUTINE = "morning_work_app_routine"
    CROSS_PLATFORM_CONTACT_COMMUNICATION = "cross_platform_contact_communication"
    INSTAGRAM_FITNESS_CONTENT_BROWSING = "instagram_fitness_content_browsing"
    ONE_OFF_FOOD_DELIVERY = "one_off_food_delivery"

class ReplayScenarioRequest(BaseModel):
    """
    Request body for replaying a predefined state snapshot scenario through the observation tracker.
    """
    scenario_name: ScenarioName

class ToolCallRequest(BaseModel):
    """
    Request body for a natural language query to be executed on the device via tool calling.
    """
    user_request: str
