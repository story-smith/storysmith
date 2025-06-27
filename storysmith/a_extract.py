import json
import os
import re
import uuid
from pathlib import Path
from typing import List

import pandas as pd
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

# Constants
CONTEXT_URL = "https://raw.githubusercontent.com/story-smith/storysmith/refs/heads/main/storysmith/context.jsonld"
ENTITY_BASE_URI = "https://story-smith.github.io/storysmith/entities/"


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
    You are a JSON-LD generator that extracts only {target_type} entities from natural language stories. 
    Use the context {context_url} to guide your extraction.

    IMPORTANT RULES:
    - Only extract entities that are clearly {target_type}s.
    - DO NOT include other entity types.
    - Use common sense and role/context in the story to decide if something is a {target_type}.
    - Output a JSON array of objects with @id, @type, and optional name.
    """.strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
    return prompt | model | OutputParser()


# -------- Feature Summarizer --------
def summarize_features(name: str, raw_text: str, target_type: str) -> str:
    system_prompt = f"""
    You are an assistant summarizing distinguishing features of a {target_type} entity.
    Output must be:
    - Very concise (≤ 1 sentence)
    - Embedding-friendly (semantic similarity optimized)
    - Focused on unique traits or actions
    """.strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("""
        Entity name: {name}

        Context excerpt:
        {raw_text}
        """),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)
    return (prompt | model).invoke({"name": name, "raw_text": raw_text}).content.strip()


# -------- Entity Extractor --------
def extract_entities(
    content: str, excerpt: str, target_type: str, episode_id: str
) -> List[dict]:
    chain = build_chain(CONTEXT_URL, target_type)
    try:
        raw_entities = chain.invoke({"input": content})
    except Exception as e:
        print(f"❌ Error extracting {target_type}s: {e}")
        return []

    schema_type = {
        "Character": "Person",
        "Place": "Place",
        "Item": "Product",
    }[target_type]

    entities = []
    for ent in raw_entities:
        name = ent.get("name", "")
        if not name:
            continue
        slug = re.sub(r"[^\w]+", "_", name.strip().lower()) or uuid.uuid4().hex[:8]
        summary = summarize_features(name, excerpt, target_type)

        entities.append(
            {
                "@id": f"{ENTITY_BASE_URI}{slug}",
                "@type": schema_type,
                "name": name,
                "episode": episode_id,
                "_features": {
                    "name": name,
                    "episode": episode_id,
                    "summary": summary,
                },
            }
        )
    return entities


# -------- ShortStory Builder --------
def build_shortstory_with_entities(
    episode_path: Path, output_path: Path, story_title: str
):
    content = episode_path.read_text(encoding="utf-8", errors="replace").strip()
    excerpt = content[:1000].encode("utf-8", "replace").decode("utf-8")
    episode_id = episode_path.stem

    story_obj = {
        "@context": "https://schema.org",
        "@type": "ShortStory",
        "name": story_title,
        "text": content,
    }

    # Extract characters
    characters = extract_entities(content, excerpt, "Character", episode_id)
    if characters:
        story_obj["character"] = characters

    # Extract place (first only, as contentLocation is singular)
    places = extract_entities(content, excerpt, "Place", episode_id)
    if places:
        story_obj["contentLocation"] = places[0]

    # Extract items (as mentions list)
    items = extract_entities(content, excerpt, "Item", episode_id)
    if items:
        story_obj["mentions"] = items

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(story_obj, f, ensure_ascii=False, indent=2)

    print(f"📘 JSON-LD saved: {output_path.name}")


# -------- CSV Loader --------
def load_single_csv_as_texts() -> tuple[dict[int, Path], str]:
    csv_path = Path(
        "data/data-by-train-split/section-stories/all/a-fish-story-story.csv"
    )
    if not csv_path.exists():
        print(f"⚠️ Missing CSV: {csv_path}")
        return {}, ""

    basename = csv_path.stem
    tmp_txt_dir = Path("temp/episodes")
    tmp_txt_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "section" not in df.columns or "text" not in df.columns:
        print("⚠️ Missing required columns in CSV.")
        return {}, ""

    episode_paths = {}
    for section_id in [2, 3, 4]:
        section_df = df[df["section"] == section_id]
        if section_df.empty:
            continue
        merged_text = "\n".join(section_df["text"].dropna().astype(str).map(str.strip))
        tmp_path = tmp_txt_dir / f"{basename}_section_{section_id}.txt"
        tmp_path.write_text(merged_text, encoding="utf-8")
        episode_paths[section_id] = tmp_path

    print(f"📦 Loaded {len(episode_paths)} episodes from CSV")
    return episode_paths, basename


# -------- Main --------
if __name__ == "__main__":
    section_episodes, csv_basename = load_single_csv_as_texts()

    if not section_episodes:
        print("⚠️ No episodes found.")
        exit()

    for section_id, episode_path in section_episodes.items():
        output_path = (
            Path("output/shortstories")
            / f"{csv_basename}_section_{section_id}.shortstory.jsonld"
        )
        story_title = csv_basename.replace("-", " ").title()
        build_shortstory_with_entities(episode_path, output_path, story_title)

    print("\n✅ All done.")
