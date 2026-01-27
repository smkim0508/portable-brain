# main app settings/configs
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from portable_brain.config.settings_mixins import (
    GoogleGenAISettingsMixin,
    AmazonNovaSettingsMixin,
    MainDBSettingsMixin
)
from portable_brain.common.logging.logger import logger
from functools import lru_cache

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
    # NOTE: this should be set in terminal to flexibly switch between different envs
    APP_ENV: str = os.getenv("APP_ENV", "dev")

class ServiceSettings(
    MainDBSettingsMixin, # could be split into a generic db classs and wrapper using db name (delimiter)
    GoogleGenAISettingsMixin,
    AmazonNovaSettingsMixin,
    DefaultSettings # passed in last to set low priority
):
    """
    The main service settings.
    Setting mix-ins are passed in for different services/clients.
    TODO: as the service expands, set global-scope configs here.
    """

    # generic rate limit settings, not tied to any LLM client
    RATE_LIMITS_ENABLED: bool = True

    # FastAPI docs settings
    INCLUDE_DOCS: bool = False # by default disable, only enable in dev

    # Health check settings
    HEALTH_CHECK_LLM: bool = False # disable expensive LLM checks in prod by default

    # default setting with env file path
    model_config = SettingsConfigDict(
        env_file=env_file_path, env_file_encoding="utf-8", extra="ignore"
        # NOTE: use env_nested_delimiter="__" to allow nested env vars in future
    )

# use lru cache to return a cached instance of service settings
# NOTE: makes settings accessible from anywhere in the app, without being request-scope
@lru_cache()
def get_service_settings() -> ServiceSettings:
    return ServiceSettings() # type: ignore
