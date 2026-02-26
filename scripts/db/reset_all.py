from portable_brain.common.db.models.base import MainDB_Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time
# TODO: import all table models as expanded to register them with MainDB_Base.metadata
from portable_brain.common.db.models.memory.structured_storage import StructuredMemory
from portable_brain.common.db.models.memory.text_embeddings import TextEmbeddingLogs
from portable_brain.common.db.models.memory.people import InterpersonalRelationship

from dotenv import load_dotenv
import os
from pathlib import Path

# import creation and deletion scripts
from scripts.db.create_tables import create_all_tables
from scripts.db.delete_tables import delete_all_tables

# NOTE: trouble shooting: if script imports do not work, use PYTHONPATH=. to explicitly include root of project
if __name__ == "__main__":
    # one-off script to reset db, by deleting then creating all tables

    # load in the proper .env file, defaulted to .env.dev
    APP_ENV = os.getenv("APP_ENV", "dev")
    # Define the path to the .env file relative to this config file's location.
    # This file is in scripts/db/, so we go up two levels to project root
    SERVICE_ROOT = Path(__file__).resolve().parents[2]
    env_file_path = SERVICE_ROOT / f".env.{APP_ENV}"

    # Load the .env file manually
    print(f"Loading env file from: {env_file_path}")
    load_dotenv(dotenv_path=env_file_path)

    # one-off script to reset all tables
    MAIN_DB_USER = os.getenv("MAIN_DB_USER")
    MAIN_DB_PW = os.getenv("MAIN_DB_PW")
    MAIN_DB_HOST = os.getenv("MAIN_DB_HOST")
    MAIN_DB_PORT = os.getenv("MAIN_DB_PORT")
    MAIN_DB_NAME = os.getenv("MAIN_DB_NAME")

    # MAIN_DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PW}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}"
    MAIN_DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PW}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}?sslmode=require"

    assert MAIN_DB_URL, "MAIN_DB_URL is not set"

    try:
        engine = create_engine(MAIN_DB_URL)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        exit(1)

    # NOTE: this is the actual logic to reset: edit as needed
    delete_all_tables(engine)
    create_all_tables(engine)
    print(f"All tables reset successfully!")
