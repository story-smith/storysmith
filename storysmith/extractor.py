import json
import os
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# GitHub上のJSON-LDコンテキストURL
CONTEXT_URL = "https://raw.githubusercontent.com/story-smith/storysmith/refs/heads/main/storysmith/context.jsonld"

# GitHub Pages URI base
CHARACTER_BASE_URI = "https://story-smith.github.io/storysmith/characters/"


def safe_json_parse(response_content: str) -> list:
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        match = re.search(
            r"```(?:json)?\s*(\[.*?\])\s*```", response_content, re.DOTALL
        )
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


def fix_character_id(char: dict) -> None:
    """
    Fix or generate @id for a character based on label.
    """
    label = char.get("label")
    cid = char.get("@id")

    if label:
        slug = slugify(label)
        if not cid or "example.org" in cid:
            char["@id"] = f"{CHARACTER_BASE_URI}{slug}"


def extract_characters(
    episodes_dir: str | Path,
    context_url: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Dict[str, List[dict]]:
    ep_dir = Path(episodes_dir)
    files = sorted(ep_dir.glob("*.txt"))

    all_characters = {}

    for f in files:
        print(f"🔍 Processing: {f.name}")
        text = f.read_text(encoding="utf-8").strip()

        prompt = [
            {
                "role": "system",
                "content": (
                    f"You are a JSON-LD generator. Output only Characters from this episode "
                    f"following context {context_url}. No natural language. "
                    "Use only keys: @id, @type. Optionally label. "
                    "Only output a JSON array of Character objects."
                ),
            },
            {"role": "user", "content": text},
        ]

        try:
            res = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=prompt,
            )
            raw = res.choices[0].message.content.strip()
            extracted = safe_json_parse(raw)

            seen_ids = set()
            episode_chars = []

            for char in extracted:
                fix_character_id(char)

                cid = char.get("@id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    char["@context"] = context_url
                    char["@type"] = "Character"
                    episode_chars.append(char)

            all_characters[f.name] = episode_chars

        except Exception as e:
            print(f"❌ Error processing {f.name}: {e}")

    return all_characters


def save_characters_to_dir_per_episode(
    char_map: Dict[str, List[dict]], output_dir: str | Path
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for episode_filename, characters in char_map.items():
        base_name = Path(episode_filename).stem
        filepath = output_path / f"{base_name}.jsonld"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(characters, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {filepath.name} ({len(characters)} characters)")


def save_individual_characters(
    char_map: Dict[str, List[dict]],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seen = set()
    for episode_chars in char_map.values():
        for char in episode_chars:
            cid = char.get("@id")
            if not cid or cid in seen:
                continue
            seen.add(cid)

            slug = Path(cid).name
            filepath = output_path / f"{slug}.jsonld"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(char, f, ensure_ascii=False, indent=2)
            print(f"📄 Saved individual character: {filepath.name}")


if __name__ == "__main__":
    episodes_dir = "episodes"
    episode_output_dir = "data/characters"
    individual_output_dir = "storysmith/characters"

    char_map = extract_characters(episodes_dir, CONTEXT_URL)
    save_characters_to_dir_per_episode(char_map, episode_output_dir)
    save_individual_characters(char_map, individual_output_dir)

    print(f"🎉 Done. Processed {len(char_map)} episode(s).")
