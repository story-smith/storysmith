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


def safe_json_parse(response_content: str) -> list:
    """
    Try to parse JSON. If response is wrapped in code block, extract and parse.
    """
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


def extract_characters(
    episodes_dir: str | Path,
    context_path: str | Path,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Dict[str, List[dict]]:
    """
    Extract character list per episode file.
    Returns dict: { episode_filename (str): [Character dicts] }
    """
    ep_dir = Path(episodes_dir)
    ctx_path = str(context_path)
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
                    f"following context {ctx_path}. No natural language. "
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
                cid = char.get("@id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    char.setdefault("@context", ctx_path)
                    char.setdefault("@type", "Character")
                    episode_chars.append(char)

            all_characters[f.name] = episode_chars

        except Exception as e:
            print(f"❌ Error processing {f.name}: {e}")

    return all_characters


def save_characters_to_dir_per_episode(
    char_map: Dict[str, List[dict]], output_dir: str | Path
) -> None:
    """
    Save each episode's character list as a JSON-LD file using episode name.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for episode_filename, characters in char_map.items():
        base_name = Path(episode_filename).stem
        filepath = output_path / f"{base_name}.jsonld"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(characters, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {filepath.name} ({len(characters)} characters)")


if __name__ == "__main__":
    episodes_dir = "episodes"
    context_path = "context.jsonld"
    output_dir = "data/characters"

    char_map = extract_characters(episodes_dir, context_path)
    save_characters_to_dir_per_episode(char_map, output_dir)
    print(f"🎉 Done. Processed {len(char_map)} episode(s).")
