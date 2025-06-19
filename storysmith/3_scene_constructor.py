import json
import os
import re
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=OPENAI_API_KEY,
)

BASE_DIR = Path("output/raw/a-fish-story-story")
OUTPUT_DIR = Path("output/triples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_TYPES = ["characters", "places", "events"]


def load_entities(section: str) -> Dict[str, Dict[str, str]]:
    result = {"characters": {}, "places": {}, "events": []}
    for etype in ENTITY_TYPES:
        path = BASE_DIR / etype / section
        if not path.exists():
            continue
        for file in path.glob("*.jsonld"):
            data = json.loads(file.read_text(encoding="utf-8"))
            if etype == "events":
                summary = data.get("_features", {}).get("summary", "").strip()
                if summary:
                    result["events"].append(summary)
            else:
                label = data.get("_features", {}).get("label", "").strip().lower()
                if label:
                    result[etype][label] = label  # use label instead of URI
    return result


def build_event_prompt(entities: Dict[str, str], summary: str) -> str:
    return f"""
You are an assistant that generates RDF-style interaction triples from event summaries.

Generate a JSON list of triples with:
- subject: entity label from the list below
- predicate: inferred from the event summary
- object: entity label from the list below

Available entities:
{", ".join(entities.keys())}

Event summary:
"{summary}"

Rules:
- Only use the above labels for subject/object.
- Predicate should be natural language, e.g., "meets", "runs away from", "protects".
""".strip()


def clean_json_block(raw: str) -> str:
    return re.sub(r"```(?:json)?", "", raw).strip()


def process_section(section_name: str):
    print(f"\n📂 Processing {section_name}")
    entity_data = load_entities(section_name)

    label_to_label = {**entity_data["characters"], **entity_data["places"]}
    if not label_to_label or not entity_data["events"]:
        print("⚠️ Skipping due to missing data.")
        return

    triples = []
    for summary in entity_data["events"]:
        prompt = build_event_prompt(label_to_label, summary)
        chat = (
            ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        "You generate RDF-like triples from event summaries."
                    ),
                    HumanMessagePromptTemplate.from_template("{prompt}"),
                ]
            )
            | model
        )
        try:
            response = chat.invoke({"prompt": prompt})
            cleaned = clean_json_block(response.content)
            generated = json.loads(cleaned)
        except Exception as e:
            print(f"⚠️ Failed to parse for summary: {summary[:30]}...: {e}")
            continue

        for t in generated:
            s = t.get("subject", "").lower().strip()
            o = t.get("object", "").lower().strip()
            p = t.get("predicate", "").strip()
            if s in label_to_label and o in label_to_label and p:
                triples.append(
                    {
                        "subject": s,
                        "predicate": p,
                        "object": o,
                    }
                )

    if triples:
        outpath = OUTPUT_DIR / f"{section_name}.json"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {outpath}")
    else:
        print(f"❌ No valid triples for {section_name}")


if __name__ == "__main__":
    for sec in ["2", "3"]:  # extend list as needed
        process_section(f"section_{sec}")
