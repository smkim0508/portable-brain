# test to connect Amazon NOVA API

import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

APP_ENV = os.getenv("APP_ENV", "dev")

# Define the path to the .env file relative to this config file's location.
# This file is in tests/, so we go up one level to project root
SERVICE_ROOT = Path(__file__).resolve().parents[1]
env_file_path = SERVICE_ROOT / f".env.{APP_ENV}"

# Load the .env file manually
print(f"Loading env file from: {env_file_path}")
load_dotenv(dotenv_path=env_file_path)

# Get the API key
nova_api_key = os.getenv("NOVA_API_KEY")
assert nova_api_key, "NOVA_API_KEY is not set"

# Guide provided by Amazon's documentation
client = OpenAI(
    api_key=nova_api_key,
    base_url="https://api.nova.amazon.com/v1"
)

response = client.chat.completions.create(
    model="nova-2-lite-v1",
    messages=[{"role": "user", "content": "Hello! How are you?"}]
)

print(response.choices[0].message.content)
