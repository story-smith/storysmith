# config.py

import os

from dotenv import load_dotenv

load_dotenv()

# API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# JSON-LD context and base URI
CONTEXT_URL = "https://raw.githubusercontent.com/story-smith/storysmith/refs/heads/main/storysmith/context.jsonld"
CHARACTER_BASE_URI = "https://story-smith.github.io/storysmith/characters/"

# Directory paths
EPISODE_DIR = "episodes"
CHARACTERS_PER_EPISODE_DIR = "data/characters"
CHARACTERS_INDIVIDUAL_DIR = "storysmith/characters"
