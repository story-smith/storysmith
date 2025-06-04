import difflib
import json
import re
import uuid
from pathlib import Path
from typing import List, Optional

from config import (
    CHARACTER_BASE_URI,
    CHARACTER_DIR,
    CONTEXT_URL,
    EPISODE_DIR,
    OPENAI_API_KEY,
    PLACE_BASE_URI,
    PLACE_DIR,
    SCENE_BASE_URI,
    SCENE_DIR,
    TIMEPOINT_BASE_URI,
    TIMEPOINT_DIR,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI


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


def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name)
    return name


def load_existing_entities_from_dir(directory: Path) -> List[dict]:
    entities = []
    for file in directory.glob("*.jsonld"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                entities.append(json.load(f))
        except json.JSONDecodeError:
            print(f"⚠️ Skipping invalid JSON: {file}")
    return entities


def find_matching_entity(
    label: str, existing_entities: List[dict], threshold: float = 0.85
) -> Optional[dict]:
    label_lower = label.strip().lower()
    for ent in existing_entities:
        existing_label = ent.get("label", "").strip().lower()
        if existing_label:
            ratio = difflib.SequenceMatcher(None, label_lower, existing_label).ratio()
            if ratio >= threshold:
                return ent
    return None


def build_chain(context_url: str, target_type: str) -> ChatOpenAI:
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

    model = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)
    return prompt | model | OutputParser()


def build_scene_chain(context_url: str, characters, places, timepoints):
    character_ids = [c["@id"] for c in characters]
    place_ids = [p["@id"] for p in places]
    timepoint_ids = [t["@id"] for t in timepoints]

    system_template = f"""
    You are a JSON-LD generator. Extract Scene objects from this episode using context {context_url}.
    Each Scene must include:
    - "@id": random unique identifier (you can leave blank)
    - "@type": must be "Scene"
    - "@context": must be "{context_url}"
    - "c": characters involved (only use from: {character_ids})
    - "p": places involved (only use from: {place_ids})
    - "t": timepoint (only use from: {timepoint_ids})
    - "seq": unique string for ordering

    Do not invent or create new values.
    Output must be a JSON array of Scenes.
    """.strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    model = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)
    return prompt | model | OutputParser()


def extract_entities(
    episodes: List[Path],
    context_url: str,
    target_type: str,
    base_uri: str,
    output_dir: Path,
    existing_entities: Optional[List[dict]] = None,
) -> List[dict]:
    chain = build_chain(context_url, target_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = {e["@id"]: True for e in existing_entities} if existing_entities else {}
    all_entities = existing_entities.copy() if existing_entities else []

    for ep in episodes:
        epname = ep.stem
        print(f"🔍 Extracting {target_type}: {ep.name}")
        content = ep.read_text(encoding="utf-8").strip()

        try:
            extracted = chain.invoke({"input": content})
            for ent in extracted:
                label = ent.get("label", "")
                if not label:
                    continue

                matched = find_matching_entity(label, all_entities)
                if matched:
                    ent["@id"] = matched["@id"]
                    ent["@type"] = matched["@type"]
                    ent["@context"] = matched["@context"]
                    ent["episode"] = epname
                    print(
                        f"🔁 Reusing existing entity for label '{label}': {ent['@id']}"
                    )
                else:
                    slug = slugify(label or uuid.uuid4().hex[:8])
                    ent["@id"] = f"{base_uri}{slug}"
                    ent["@type"] = target_type
                    ent["@context"] = context_url
                    ent["episode"] = epname

                    if ent["@id"] in seen:
                        continue
                    seen[ent["@id"]] = True

                    outpath = output_dir / f"{slug}.jsonld"
                    with open(outpath, "w", encoding="utf-8") as f:
                        json.dump(ent, f, ensure_ascii=False, indent=2)
                    print(f"📄 Saved: {outpath.name}")
                    all_entities.append(ent)
        except Exception as e:
            print(f"❌ Error processing {ep.name}: {e}")

    return all_entities


def extract_scenes(
    episodes: List[Path], chain, context_url: str, output_dir: Path, merged_path: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scenes = []

    for ep in episodes:
        epname = ep.stem
        print(f"🎬 Extracting Scene: {ep.name}")
        content = ep.read_text(encoding="utf-8").strip()

        try:
            extracted = chain.invoke({"input": content})
            for ent in extracted:
                ent["@id"] = f"{SCENE_BASE_URI}{uuid.uuid4().hex[:8]}"
                ent["@type"] = "Scene"
                ent["@context"] = context_url
                ent["episode"] = epname

            ep_path = output_dir / f"{epname}.jsonld"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(extracted, f, ensure_ascii=False, indent=2)
            print(f"✅ Scene saved: {ep_path.name}")

            all_scenes.extend(extracted)
        except Exception as e:
            print(f"❌ Error processing {ep.name}: {e}")

    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    print(f"📦 Merged all scenes: {merged_path}")


if __name__ == "__main__":
    episodes = sorted(Path(EPISODE_DIR).glob("*.txt"))
    episode1 = [ep for ep in episodes if ep.name == "episode1.txt"]
    episode2 = [ep for ep in episodes if ep.name == "episode2.txt"]

    characters1 = extract_entities(
        episode1, CONTEXT_URL, "Character", CHARACTER_BASE_URI, Path(CHARACTER_DIR)
    )
    places1 = extract_entities(
        episode1, CONTEXT_URL, "Place", PLACE_BASE_URI, Path(PLACE_DIR)
    )
    timepoints1 = extract_entities(
        episode1, CONTEXT_URL, "TimePoint", TIMEPOINT_BASE_URI, Path(TIMEPOINT_DIR)
    )

    characters_existing = load_existing_entities_from_dir(Path(CHARACTER_DIR))
    places_existing = load_existing_entities_from_dir(Path(PLACE_DIR))
    timepoints_existing = load_existing_entities_from_dir(Path(TIMEPOINT_DIR))

    characters2 = extract_entities(
        episode2,
        CONTEXT_URL,
        "Character",
        CHARACTER_BASE_URI,
        Path(CHARACTER_DIR),
        existing_entities=characters_existing,
    )
    places2 = extract_entities(
        episode2,
        CONTEXT_URL,
        "Place",
        PLACE_BASE_URI,
        Path(PLACE_DIR),
        existing_entities=places_existing,
    )
    timepoints2 = extract_entities(
        episode2,
        CONTEXT_URL,
        "TimePoint",
        TIMEPOINT_BASE_URI,
        Path(TIMEPOINT_DIR),
        existing_entities=timepoints_existing,
    )

    all_characters = characters_existing + characters2
    all_places = places_existing + places2
    all_timepoints = timepoints_existing + timepoints2

    scene_chain = build_scene_chain(
        CONTEXT_URL, all_characters, all_places, all_timepoints
    )
    extract_scenes(
        episode1 + episode2,
        scene_chain,
        CONTEXT_URL,
        Path(SCENE_DIR),
        Path(SCENE_DIR) / "scene-all.jsonld",
    )

    print("🎉 Done. All episodes processed.")
