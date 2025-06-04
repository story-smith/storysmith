import json
import os
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser

# 環境変数読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 定数
CONTEXT_URL = "https://raw.githubusercontent.com/story-smith/storysmith/refs/heads/main/storysmith/context.jsonld"
CHARACTER_BASE_URI = "https://story-smith.github.io/storysmith/characters/"


class CharacterOutputParser(BaseOutputParser):
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


def fix_character_id(char: dict) -> None:
    label = char.get("label")
    cid = char.get("@id")
    if label:
        slug = slugify(label)
        if not cid or "example.org" in cid:
            char["@id"] = f"{CHARACTER_BASE_URI}{slug}"


def build_chain(context_url: str) -> ChatOpenAI:
    system_template = (
        f"You are a JSON-LD generator. Output only Characters from this episode "
        f"following context {context_url}. No natural language. "
        "Use only keys: @id, @type. Optionally label. "
        "Only output a JSON array of Character objects."
    )

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

    return prompt | model | CharacterOutputParser()


def extract_characters(
    episodes_dir: str | Path, context_url: str
) -> Dict[str, List[dict]]:
    episodes = sorted(Path(episodes_dir).glob("*.txt"))
    chain = build_chain(context_url)
    all_characters = {}

    for ep in episodes:
        print(f"🔍 Processing: {ep.name}")
        content = ep.read_text(encoding="utf-8").strip()

        try:
            extracted = chain.invoke({"input": content})
            seen = set()
            episode_chars = []

            for char in extracted:
                fix_character_id(char)
                cid = char.get("@id")
                if cid and cid not in seen:
                    seen.add(cid)
                    char["@context"] = context_url
                    char["@type"] = "Character"
                    episode_chars.append(char)

            all_characters[ep.name] = episode_chars
        except Exception as e:
            print(f"❌ Error processing {ep.name}: {e}")

    return all_characters


def save_characters_to_dir_per_episode(
    char_map: Dict[str, List[dict]], output_dir: str | Path
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname, chars in char_map.items():
        outpath = output_dir / f"{Path(fname).stem}.jsonld"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(chars, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {outpath.name} ({len(chars)} characters)")


def save_individual_characters(char_map: Dict[str, List[dict]], output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for chars in char_map.values():
        for char in chars:
            cid = char.get("@id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            slug = Path(cid).name
            outpath = output_dir / f"{slug}.jsonld"
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(char, f, ensure_ascii=False, indent=2)
            print(f"📄 Saved individual character: {outpath.name}")


if __name__ == "__main__":
    ep_dir = "episodes"
    per_episode_out = "data/characters"
    individual_out = "storysmith/characters"

    char_map = extract_characters(ep_dir, CONTEXT_URL)
    save_characters_to_dir_per_episode(char_map, per_episode_out)
    save_individual_characters(char_map, individual_out)

    print(f"🎉 Done. Processed {len(char_map)} episode(s).")
