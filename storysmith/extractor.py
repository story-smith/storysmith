import json
import re
import uuid
from pathlib import Path
from typing import Dict, List

from config import (
    CHARACTER_BASE_URI,
    CHARACTERS_INDIVIDUAL_DIR,
    CHARACTERS_PER_EPISODE_DIR,
    CONTEXT_URL,
    EPISODE_DIR,
    OPENAI_API_KEY,
    PLACE_BASE_URI,
    PLACES_INDIVIDUAL_DIR,
    PLACES_PER_EPISODE_DIR,
    SCENE_BASE_URI,
    SCENES_PER_EPISODE_DIR,
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
    system_template = {
        "Character": f"You are a JSON-LD generator. Extract only Characters from this episode, following context {context_url}. Output a JSON array. Only include @id, @type, and optional label.",
        "Place": f"You are a JSON-LD generator. Extract only Places from this episode, following context {context_url}. Output a JSON array. Only include @id, @type, and optional label.",
    }[target_type]

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )

    return prompt | model | OutputParser()


def build_scene_chain(context_url: str, characters: List[dict], places: List[dict]):
    character_ids = [c["@id"] for c in characters]
    place_ids = [p["@id"] for p in places]

    system_template = f"""
You are a JSON-LD generator. Extract Scene objects from this episode using context {context_url}.
Each Scene must include:
- "@id": random unique identifier (you can leave blank)
- "@type": must be "Scene"
- "@context": must be "{context_url}"
- "c": characters involved in the scene (only use from the following list): {character_ids}
- "p": places involved (only use from the following list): {place_ids}
- "seq": unique string for ordering

Only use the provided @id values for c and p. Do not create new ones.
Output must be a JSON array of Scenes.
""".strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )

    return prompt | model | OutputParser()


def extract_entities(
    episodes_dir: str | Path, context_url: str, target_type: str
) -> Dict[str, List[dict]]:
    episodes = sorted(Path(episodes_dir).glob("*.txt"))
    chain = build_chain(context_url, target_type)
    all_entities = {}

    for ep in episodes:
        print(f"🔍 Extracting {target_type}: {ep.name}")
        content = ep.read_text(encoding="utf-8").strip()

        try:
            extracted = chain.invoke({"input": content})
            seen = set()
            episode_entities = []

            for ent in extracted:
                label = ent.get("label", "")
                if target_type == "Character":
                    slug = slugify(label or uuid.uuid4().hex[:8])
                    ent["@id"] = f"{CHARACTER_BASE_URI}{slug}"
                elif target_type == "Place":
                    slug = slugify(label or uuid.uuid4().hex[:8])
                    ent["@id"] = f"{PLACE_BASE_URI}{slug}"
                ent["@type"] = target_type
                ent["@context"] = context_url

                if ent["@id"] in seen:
                    continue
                seen.add(ent["@id"])
                episode_entities.append(ent)

            all_entities[ep.name] = episode_entities
        except Exception as e:
            print(f"❌ Error processing {ep.name}: {e}")

    return all_entities


def extract_scenes_with_chain(
    episodes_dir: str | Path, chain, context_url: str
) -> Dict[str, List[dict]]:
    episodes = sorted(Path(episodes_dir).glob("*.txt"))
    all_entities = {}

    for ep in episodes:
        print(f"🎬 Extracting Scene: {ep.name}")
        content = ep.read_text(encoding="utf-8").strip()

        try:
            extracted = chain.invoke({"input": content})
            seen = set()
            episode_entities = []

            for ent in extracted:
                ent["@id"] = f"{SCENE_BASE_URI}{uuid.uuid4().hex[:8]}"
                ent["@type"] = "Scene"
                ent["@context"] = context_url
                if ent["@id"] in seen:
                    continue
                seen.add(ent["@id"])
                episode_entities.append(ent)

            all_entities[ep.name] = episode_entities
        except Exception as e:
            print(f"❌ Error processing {ep.name}: {e}")

    return all_entities


def save_entities_to_dir_per_episode(
    entity_map: Dict[str, List[dict]], output_dir: str | Path
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname, entities in entity_map.items():
        outpath = output_dir / f"{Path(fname).stem}.jsonld"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {outpath.name} ({len(entities)} items)")


def save_individual_entities(entity_map: Dict[str, List[dict]], output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for entities in entity_map.values():
        for ent in entities:
            cid = ent.get("@id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            slug = Path(cid).name
            outpath = output_dir / f"{slug}.jsonld"
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(ent, f, ensure_ascii=False, indent=2)
            print(f"📄 Saved individual: {outpath.name}")


if __name__ == "__main__":
    # Extract characters
    char_map = extract_entities(EPISODE_DIR, CONTEXT_URL, "Character")
    save_entities_to_dir_per_episode(char_map, CHARACTERS_PER_EPISODE_DIR)
    save_individual_entities(char_map, CHARACTERS_INDIVIDUAL_DIR)

    # Extract places
    place_map = extract_entities(EPISODE_DIR, CONTEXT_URL, "Place")
    save_entities_to_dir_per_episode(place_map, PLACES_PER_EPISODE_DIR)
    save_individual_entities(place_map, PLACES_INDIVIDUAL_DIR)

    # Flatten for scene chain
    all_characters = [c for clist in char_map.values() for c in clist]
    all_places = [p for plist in place_map.values() for p in plist]

    scene_chain = build_scene_chain(CONTEXT_URL, all_characters, all_places)
    scene_map = extract_scenes_with_chain(EPISODE_DIR, scene_chain, CONTEXT_URL)
    save_entities_to_dir_per_episode(scene_map, SCENES_PER_EPISODE_DIR)

    print(f"🎉 Done. Processed {len(char_map)} episode(s).")
