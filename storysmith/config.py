# config.py

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CONTEXT_URL = "https://raw.githubusercontent.com/story-smith/storysmith/refs/heads/main/storysmith/context.jsonld"

CHARACTER_BASE_URI = "https://story-smith.github.io/storysmith/characters/"
PLACE_BASE_URI = "https://story-smith.github.io/storysmith/places/"
SCENE_BASE_URI = "https://story-smith.github.io/storysmith/scenes/"
TIMEPOINT_BASE_URI = "https://story-smith.github.io/storysmith/timepoints/"
EVENT_BASE_URI = "https://story-smith.github.io/storysmith/events/"

EPISODE_DIR = "episodes"
DATA_DIR = "data"

# Canonical entity directories (after integration)
CHARACTER_DIR = os.path.join(DATA_DIR, "characters")
PLACE_DIR = os.path.join(DATA_DIR, "places")
TIMEPOINT_DIR = os.path.join(DATA_DIR, "timepoints")
SCENE_DIR = os.path.join(DATA_DIR, "scenes")
EVENT_DIR = os.path.join(DATA_DIR, "events")

# Raw extracted entity directories (before integration)
CHARACTER_DIR_RAW = os.path.join(DATA_DIR, "raw", "characters")
PLACE_DIR_RAW = os.path.join(DATA_DIR, "raw", "places")
TIMEPOINT_DIR_RAW = os.path.join(DATA_DIR, "raw", "timepoints")
EVENT_DIR_RAW = os.path.join(DATA_DIR, "raw", "events")
