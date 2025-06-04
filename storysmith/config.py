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

# 保存先をすべて data フォルダ内に統一
DATA_DIR = "data"
CHARACTER_DIR = os.path.join(DATA_DIR, "characters")
PLACE_DIR = os.path.join(DATA_DIR, "places")
TIMEPOINT_DIR = os.path.join(DATA_DIR, "timepoints")
SCENE_DIR = os.path.join(DATA_DIR, "scenes")
