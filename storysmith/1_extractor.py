import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Base URIs
CONTEXT_URL = "https://raw.githubusercontent.com/story-smith/storysmith/refs/heads/main/storysmith/context.jsonld"
CHARACTER_BASE_URI = "https://story-smith.github.io/storysmith/characters/"
PLACE_BASE_URI = "https://story-smith.github.io/storysmith/places/"
SCENE_BASE_URI = "https://story-smith.github.io/storysmith/scenes/"
TIMEPOINT_BASE_URI = "https://story-smith.github.io/storysmith/timepoints/"
EVENT_BASE_URI = "https://story-smith.github.io/storysmith/events/"


# -------- Output Parser --------
class OutputParser(BaseOutputParser):
    def parse(self, text: str) -> List[dict]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        print("⚠️ Failed to parse JSON from GPT output.")
        return []


# -------- Chain Builders --------
def build_chain(context_url: str, target_type: str):
    system_template = f"""
    You are a JSON-LD generator. Extract only {target_type}s from this episode, following context {context_url}.
    Output a JSON array. Only include @id, @type, and optional label.
    """.strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
    return prompt | model | OutputParser()


def summarize_features(label: str, raw_text: str, target_type: str) -> str:
    system_prompt = f"""
    You are an assistant summarizing distinguishing features of a {target_type} entity.
    Your output should be:
    - Extremely concise (max 1 sentence, ~20 words)
    - Optimized for use in similarity comparison using embeddings
    - Focused on traits or actions that make the {target_type} unique

    Avoid generic descriptions. Be precise.
    """.strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("""
        Entity label: {label}

        Context excerpt:
        {raw_text}
        """),
        ]
    )

    model = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY)
    return (
        (prompt | model).invoke({"label": label, "raw_text": raw_text}).content.strip()
    )


# -------- Entity Extraction --------
def extract_entities_from_episodes(
    episodes: List[Path], target_type: str, base_uri: str, output_dir: Path
):
    chain = build_chain(CONTEXT_URL, target_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entities = []

    for ep in episodes:
        epname = ep.stem
        print(f"🔍 Extracting {target_type} from {ep.name}")
        content = ep.read_text(encoding="utf-8", errors="replace").strip()

        try:
            extracted = chain.invoke({"input": content})
            for ent in extracted:
                label = ent.get("label", "")
                slug = (
                    re.sub(r"[^\w]+", "_", label.strip().lower())
                    or uuid.uuid4().hex[:8]
                )
                raw_excerpt = content[:1000].encode("utf-8", "replace").decode("utf-8")
                features_summary = summarize_features(label, raw_excerpt, target_type)

                ent["@id"] = f"{base_uri}{slug}"
                ent["@type"] = target_type
                ent["@context"] = CONTEXT_URL
                ent["episode"] = epname
                ent["_features"] = {
                    "label": label,
                    "episode": epname,
                    "summary": features_summary,
                }

                outpath = output_dir / f"{slug}.jsonld"
                with open(outpath, "w", encoding="utf-8", errors="replace") as f:
                    safe_ent = json.loads(json.dumps(ent, ensure_ascii=False))
                    json.dump(safe_ent, f, ensure_ascii=False, indent=2)
                print(f"📄 Saved: {outpath.name}")
                all_entities.append(ent)
        except Exception as e:
            print(f"❌ Error processing {ep.name}: {e}")

    return all_entities


# -------- Path Configuration --------
def get_paths(category: str):
    base_dir = Path("data") / category

    return {
        "episode_dir": base_dir / "episodes",
        "character_dir_raw": base_dir / "raw" / "characters",
        "place_dir_raw": base_dir / "raw" / "places",
        "timepoint_dir_raw": base_dir / "raw" / "timepoints",
        "event_dir_raw": base_dir / "raw" / "events",
    }


# -------- CLI Entry Point --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        nargs="+",
        default=["main", "fanfic"],
        help="Target categories (e.g., main fanfic). Defaults to both.",
    )
    args = parser.parse_args()

    for category in args.category:
        print(f"\n📂 Processing category: {category}")
        paths = get_paths(category)
        episodes = sorted(paths["episode_dir"].glob("*.txt"))

        if not episodes:
            print(f"⚠️ No episodes found in: {paths['episode_dir']}")
            continue

        extract_entities_from_episodes(
            episodes, "Character", CHARACTER_BASE_URI, paths["character_dir_raw"]
        )
        extract_entities_from_episodes(
            episodes, "Place", PLACE_BASE_URI, paths["place_dir_raw"]
        )
        extract_entities_from_episodes(
            episodes, "TimePoint", TIMEPOINT_BASE_URI, paths["timepoint_dir_raw"]
        )
        extract_entities_from_episodes(
            episodes, "Event", EVENT_BASE_URI, paths["event_dir_raw"]
        )

        print(f"✅ Done with category: {category}")

    print("\n🎉 All extraction complete.")
