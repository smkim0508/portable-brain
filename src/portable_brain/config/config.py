# main app settings/configs
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from portable_brain.config.settings_mixins import (
    GoogleGenAISettingsMixin,
    MainDBSettingsMixin
)
from portable_brain.common.logging.logger import logger

# Determine which environment we're in. Default to 'dev'.
# NOTE: for now, only dev is used, but subject to expand as service matures.
APP_ENV = os.getenv("APP_ENV", "dev")

# Define the path to the .env file relative to this config file's location.
# This file is in portable_brain/config/, so we go up three levels to portable_brain/
# NOTE: the .env file names must match the APP_ENV config.
SERVICE_ROOT = Path(__file__).resolve().parents[3]
env_file_path = SERVICE_ROOT / f".env.{APP_ENV}"
logger.info(f"APP_ENV: {APP_ENV}")

class DefaultSettings(BaseSettings):
    """
    The baseline, default settings that govern common functionalities.
    Universal, low-levell settings and pydantic config for parsing .env files.
    Passed in last to set low priority (allows overrides)
    """
    # This base config ensures that if a service-specific settings class
    # doesn't define its own model_config, it will still have these safe defaults.
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")
    
    # The application environment is the only truly universal setting.
    APP_ENV: str = os.getenv("APP_ENV", "dev")

class MainSettings(
    MainDBSettingsMixin,
    GoogleGenAISettingsMixin,
    DefaultSettings # passed in last to set low priority
):
    """
    The main app settings.
    Setting mix-ins are passed in for different services/clients.
    TODO: as the service expands, set global-scope configs here.
    """

    # generic rate limit settings, not tied to any LLM client
    RATE_LIMITS_ENABLED: bool = True

    model_config = SettingsConfigDict(
        env_file=env_file_path, env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )
