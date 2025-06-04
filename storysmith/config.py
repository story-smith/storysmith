import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CONTEXT_URL = "https://raw.githubusercontent.com/story-smith/storysmith/refs/heads/main/storysmith/context.jsonld"
CHARACTER_BASE_URI = "https://story-smith.github.io/storysmith/characters/"
PLACE_BASE_URI = "https://story-smith.github.io/storysmith/places/"
SCENE_BASE_URI = "https://story-smith.github.io/storysmith/scenes/"
TIMEPOINT_BASE_URI = "https://story-smith.github.io/storysmith/timepoints/"

EPISODE_DIR = "episodes"
CHARACTERS_PER_EPISODE_DIR = "data/characters"
PLACES_PER_EPISODE_DIR = "data/places"
SCENES_PER_EPISODE_DIR = "data/scenes"
TIMEPOINTS_PER_EPISODE_DIR = "data/timepoints"

CHARACTERS_INDIVIDUAL_DIR = "storysmith/characters"
PLACES_INDIVIDUAL_DIR = "storysmith/places"
TIMEPOINTS_INDIVIDUAL_DIR = "storysmith/timepoints"
