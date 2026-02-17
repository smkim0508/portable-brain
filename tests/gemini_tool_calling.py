from google import genai
from google.genai import types
import os
from pathlib import Path
from dotenv import load_dotenv

"""
Test script to verify gemini tool calling capabilities.
"""

# Usual set up to parse env vars
APP_ENV = os.getenv("APP_ENV", "dev")

# Define the path to the .env file relative to this config file's location.
# This file is in tests/, so we go up one level to project root
SERVICE_ROOT = Path(__file__).resolve().parents[1]
env_file_path = SERVICE_ROOT / f".env.{APP_ENV}"

# Load the .env file manually
print(f"Loading env file from: {env_file_path}")
load_dotenv(dotenv_path=env_file_path)

# Get the API key
google_api_key = os.getenv("GOOGLE_GENAI_API_KEY")

# example usage provided by Google GenAI documentation
# https://ai.google.dev/gemini-api/docs/function-calling?example=meeting

# Define a function that the model can call to control smart lights
# NOTE: the declaration telling LLM what/how to use this external function
set_light_values_declaration = {
    "name": "set_light_values",
    "description": "Sets the brightness and color temperature of a light.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100. Zero is off and 100 is full brightness",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}

# This is the actual function that would be called based on the model's suggestion
def set_light_values(brightness: int, color_temp: str) -> dict[str, int | str]:
    """Set the brightness and color temperature of a room light. (mock API).

    Args:
        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness
        color_temp: Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    # some work can be done here
    print(f"Successfully set brightness to {brightness} and color temperature to {color_temp}.")
    return {"brightness": brightness, "colorTemperature": color_temp}

# Configure the client and tools
client = genai.Client(api_key=google_api_key)
tools = types.Tool(function_declarations=[set_light_values_declaration]) # type: ignore
config = types.GenerateContentConfig(tools=[tools])

# Define user prompt
contents = [
    types.Content(
        role="user", parts=[types.Part(text="Turn the lights down to a romantic level")]
    )
]

# Send request with function declarations and user prompt
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents, # type: ignore
    config=config, # NOTE: this is where the actual declaration is wrapped and passed.
)

# Debug
print(response.candidates[0].content.parts[0].function_call) # type: ignore

# Extract tool call details, it may not be in the first part.
tool_call = response.candidates[0].content.parts[0].function_call # type: ignore

if tool_call.name == "set_light_values": # type: ignore
    result = set_light_values(**tool_call.args) # type: ignore
    print(f"Function execution result: {result}")


# Create a function response part
function_response_part = types.Part.from_function_response(
    name=tool_call.name, # type: ignore
    response={"result": result},
)

# Append function call and result of the function execution to contents
contents.append(response.candidates[0].content) # Append the content from the model's response. # type: ignore
contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

# Create final response w/ user-friendly specifications.
client = genai.Client(api_key=google_api_key)
final_response = client.models.generate_content(
    model="gemini-3-flash-preview",
    config=config,
    contents=contents, # type: ignore
)

print(final_response.text)
