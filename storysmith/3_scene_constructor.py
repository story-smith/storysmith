import json
import re
from pathlib import Path
from typing import Dict, List

from config import OPENAI_API_KEY
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

CHARACTER_DIR = Path("data/integrated/characters")
EPISODE_DIR = Path("episodes")
OUTPUT_DIR = Path("data/kg/character_relations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY,
)


def load_character_label_uri() -> Dict[str, str]:
    result = {}
    for file in CHARACTER_DIR.glob("*.jsonld"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            label = data.get("_features", {}).get("label", "").strip().lower()
            uri = data.get("@id")
            if label and uri:
                result[label] = uri
        except Exception as e:
            print(f"❌ Failed to parse {file.name}: {e}")
    return result


def build_gpt_prompt(char_labels: List[str], episode_text: str) -> str:
    return f"""
You are a helpful assistant that extracts interactions between characters from a story.

Extract triples in the form of:
- subject (a character from the provided list)
- predicate (an open natural-language verb or phrase)
- object (a character from the same list)

Rules:
- Only use characters from the list below as subject and object.
- Output a list of JSON triples.
- Predicate can be any descriptive phrase (e.g. "helps", "confesses love to", "fights").

Characters:
{", ".join(char_labels)}

Story:
{episode_text}
""".strip()


def clean_json_block(raw: str) -> str:
    # Remove triple backticks or "```json"
    return re.sub(r"```(?:json)?", "", raw).strip()


def extract_character_triples(character_uri_map: Dict[str, str]):
    character_labels = list(character_uri_map.keys())

    for file in EPISODE_DIR.glob("*.txt"):
        print(f"\n📝 Reading episode file: {file.name}")
        try:
            episode_text = file.read_text(encoding="utf-8")
            print("📖 Episode preview:\n", episode_text[:300], "...\n")
        except Exception as e:
            print(f"❌ Failed to read {file.name}: {e}")
            continue

        # Prompt GPT
        prompt_text = build_gpt_prompt(character_labels, episode_text)
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You extract character-to-character interaction triples."
                ),
                HumanMessagePromptTemplate.from_template("{prompt}"),
            ]
        )
        chain = prompt | model

        try:
            print("🤖 Sending prompt to GPT...")
            response = chain.invoke({"prompt": prompt_text})
            raw_content = response.content
            print("📦 GPT raw response:\n", raw_content)

            cleaned = clean_json_block(raw_content)
            triples = json.loads(cleaned)
        except Exception as e:
            print(f"⚠️ GPT parse error for {file.name}: {e}")
            continue

        # URI 変換
        converted = []
        for t in triples:
            subj_label = t.get("subject", "").lower().strip()
            obj_label = t.get("object", "").lower().strip()
            predicate = t.get("predicate", "").strip()

            if (
                subj_label in character_uri_map
                and obj_label in character_uri_map
                and predicate
            ):
                converted.append(
                    {
                        "subject": character_uri_map[subj_label],
                        "predicate": predicate,
                        "object": character_uri_map[obj_label],
                    }
                )
            else:
                print(f"⚠️ Skipping unmatched triple: {t}")

        if converted:
            outpath = OUTPUT_DIR / (file.stem + ".json")
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(converted, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved: {outpath.name}")
        else:
            print(f"❌ No valid triples extracted from {file.name}")


if __name__ == "__main__":
    character_uri_map = load_character_label_uri()
    extract_character_triples(character_uri_map)
    print("\n🏁 All episodes processed.")
