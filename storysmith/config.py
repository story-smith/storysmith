import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CONTEXT_URL = "https://raw.githubusercontent.com/story-smith/storysmith/refs/heads/main/storysmith/context.jsonld"
CHARACTER_BASE_URI = "https://story-smith.github.io/storysmith/characters/"
PLACE_BASE_URI = "https://story-smith.github.io/storysmith/places/"

EPISODE_DIR = "episodes"
CHARACTERS_PER_EPISODE_DIR = "data/characters"
CHARACTERS_INDIVIDUAL_DIR = "storysmith/characters"

PLACES_PER_EPISODE_DIR = "data/places"
PLACES_INDIVIDUAL_DIR = "storysmith/places"
