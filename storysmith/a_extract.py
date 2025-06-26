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

# Base URIs and context
CONTEXT_URL = "https://raw.githubusercontent.com/story-smith/storysmith/refs/heads/main/storysmith/context.jsonld"
CHARACTER_BASE_URI = "https://story-smith.github.io/storysmith/characters/"


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
        print("\u26a0\ufe0f Failed to parse JSON from GPT output.")
        return []


# -------- Chain Builders --------
def build_chain(context_url: str, target_type: str):
    system_template = f"""
    You are a JSON-LD generator that extracts only {target_type} entities from natural language stories. 
    Use the context {context_url} to guide your extraction.

    IMPORTANT RULES:
    - Only extract entities that are clearly {target_type}s.
    - DO NOT include other entity types. For example, do not include characters when extracting places.
    - Use common sense and role/context in the story to decide if something is a {target_type}.
    - Include group characters (e.g., tribes, families, or animal groups) if they act together or show signs of agency (e.g., speech, action, emotion).
    - Output a JSON array of objects with @id, @type, and optional label.

    Be conservative: if you are unsure about the type, do not include the entity.
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

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)
    return (
        (prompt | model).invoke({"label": label, "raw_text": raw_text}).content.strip()
    )


# -------- Entity Extraction and ShortStory Assembly --------
def extract_characters_and_build_shortstory(
    episode_path: Path, base_uri: str, output_path: Path, story_title: str
):
    chain = build_chain(CONTEXT_URL, "Character")
    content = episode_path.read_text(encoding="utf-8", errors="replace").strip()
    raw_excerpt = content[:1000].encode("utf-8", "replace").decode("utf-8")

    characters = []
    try:
        extracted = chain.invoke({"input": content})
        for ent in extracted:
            label = ent.get("label", "")
            slug = re.sub(r"[^\w]+", "_", label.strip().lower()) or uuid.uuid4().hex[:8]
            features_summary = summarize_features(label, raw_excerpt, "Character")

            characters.append(
                {
                    "@id": f"{base_uri}{slug}",
                    "@type": "Person",
                    "label": label,
                    "episode": episode_path.stem,
                    "_features": {
                        "label": label,
                        "episode": episode_path.stem,
                        "summary": features_summary,
                    },
                }
            )
    except Exception as e:
        print(f"❌ Error extracting characters from {episode_path.name}: {e}")

    story_obj = {
        "@context": "https://schema.org",
        "@type": "ShortStory",
        "name": story_title,
        "text": content,
        "character": characters,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(story_obj, f, ensure_ascii=False, indent=2)

    print(f"📘 ShortStory saved: {output_path.name}")


# -------- Episode Loader --------
def load_single_csv_as_texts() -> tuple[dict[int, Path], str]:
    csv_path = Path(
        "data/data-by-train-split/section-stories/all/a-fish-story-story.csv"
    )
    if not csv_path.exists():
        print(f"⚠️ Specified CSV not found: {csv_path}")
        return {}, ""

    basename = csv_path.stem
    tmp_txt_dir = Path("temp/episodes")
    tmp_txt_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = {}
    df = pd.read_csv(csv_path)

    if "section" not in df.columns or "text" not in df.columns:
        print("⚠️ Missing required columns: 'section' and 'text'")
        return {}, ""

    for section_id in [2, 3]:
        section_df = df[df["section"] == section_id]
        if section_df.empty:
            continue

        merged_text = "\n".join(section_df["text"].dropna().astype(str).map(str.strip))
        tmp_path = tmp_txt_dir / f"{basename}_section_{section_id}.txt"
        tmp_path.write_text(merged_text, encoding="utf-8")
        episode_paths[section_id] = tmp_path

    print(f"📦 Loaded {len(episode_paths)} sections from: {csv_path.name}")
    return episode_paths, basename


# -------- Main --------
if __name__ == "__main__":
    section_episodes, csv_basename = load_single_csv_as_texts()

    if not section_episodes:
        print("⚠️ No episodes to process.")
        exit()

    for section_id, episode_path in section_episodes.items():
        output_path = (
            Path("output/shortstories")
            / f"{csv_basename}_section_{section_id}.shortstory.jsonld"
        )
        story_title = csv_basename.replace("-", " ").title()
        extract_characters_and_build_shortstory(
            episode_path, CHARACTER_BASE_URI, output_path, story_title
        )

    print("\n✅ All ShortStory documents complete.")
