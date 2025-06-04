import json
import re
import uuid
from pathlib import Path
from typing import List

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
    episodes_dir, context_url: str, target_type: str, base_uri: str, output_dir: Path
) -> List[dict]:
    episodes = sorted(Path(episodes_dir).glob("*.txt"))
    chain = build_chain(context_url, target_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = {}
    all_entities = []

    for ep in episodes:
        epname = Path(ep.name).stem
        print(f"🔍 Extracting {target_type}: {ep.name}")
        content = ep.read_text(encoding="utf-8").strip()

        try:
            extracted = chain.invoke({"input": content})
            for ent in extracted:
                label = ent.get("label", "")
                slug = slugify(label or uuid.uuid4().hex[:8])
                ent["@id"] = f"{base_uri}{slug}"
                ent["@type"] = target_type
                ent["@context"] = context_url
                ent["episode"] = epname

                if ent["@id"] in seen:
                    continue
                seen[ent["@id"]] = True

                # Save each entity to individual file
                outpath = output_dir / f"{slug}.jsonld"
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(ent, f, ensure_ascii=False, indent=2)
                print(f"📄 Saved: {outpath.name}")
                all_entities.append(ent)
        except Exception as e:
            print(f"❌ Error processing {ep.name}: {e}")

    return all_entities


def extract_scenes(
    episodes_dir, chain, context_url: str, output_dir: Path, merged_path: Path
):
    episodes = sorted(Path(episodes_dir).glob("*.txt"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scenes = []

    for ep in episodes:
        epname = Path(ep.name).stem
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
    characters = extract_entities(
        EPISODE_DIR, CONTEXT_URL, "Character", CHARACTER_BASE_URI, Path(CHARACTER_DIR)
    )

    places = extract_entities(
        EPISODE_DIR, CONTEXT_URL, "Place", PLACE_BASE_URI, Path(PLACE_DIR)
    )

    timepoints = extract_entities(
        EPISODE_DIR, CONTEXT_URL, "TimePoint", TIMEPOINT_BASE_URI, Path(TIMEPOINT_DIR)
    )

    scene_chain = build_scene_chain(CONTEXT_URL, characters, places, timepoints)

    extract_scenes(
        EPISODE_DIR,
        scene_chain,
        CONTEXT_URL,
        Path(SCENE_DIR),
        Path(SCENE_DIR) / "scene-all.jsonld",
    )

    print("🎉 Done. Processed all episodes.")
