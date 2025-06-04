import json
import re
from pathlib import Path
from typing import Dict, Optional

from config import CONTEXT_URL, OPENAI_API_KEY
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

# === CONFIG ===
INTEGRATED_DIRS = {
    "Character": Path("data/integrated/characters"),
    "Place": Path("data/integrated/places"),
    "TimePoint": Path("data/integrated/timepoints"),
}
EVENT_DIR = Path("data/raw/events")
KG_OUTPUT_DIR = Path("data/kg/events")
EVENT_BASE_URI = "https://story-smith.github.io/storysmith/events/"

# === SETUP ===
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)


# === 1. Build label → URI map ===
def build_label_uri_map() -> Dict[str, str]:
    label_uri = {}
    for path in INTEGRATED_DIRS.values():
        for file in path.glob("*.jsonld"):
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
                label = data.get("_features", {}).get("label", "").lower().strip()
                if label:
                    label_uri[label] = data["@id"]
    return label_uri


# === 2. Extract triple elements from summary (GPT) ===
def extract_triple_elements(summary: str) -> Optional[Dict[str, str]]:
    system_prompt = """
You are an assistant that extracts knowledge triples from short event summaries.

Extract:
- actor (the subject/agent who did something)
- target (optional: who or what it was done to)
- location (optional: where it happened)
- time (optional: when it happened)

Return JSON like:
{{
  "actor": "Akira",
  "target": "Kaiju",
  "location": "Tokyo Tower",
  "time": "August 6, 1945"
}}
""".strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    chain = prompt | model
    try:
        response = chain.invoke({"input": summary}).content
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"{.*}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    print("⚠️ Failed to parse GPT output:", response)
    return None


# === 3. Build and save event triples ===
def create_event_triples(label_uri_map: Dict[str, str]):
    KG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file in sorted(EVENT_DIR.glob("*.jsonld")):
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        label = data.get("label", "")
        summary = data.get("_features", {}).get("summary", "")
        eid = data.get("@id")

        # ない場合は自動生成
        if not eid:
            slug = re.sub(r"[^\w]+", "_", label.strip().lower())
            eid = f"{EVENT_BASE_URI}{slug}"

        print(f"🧠 Processing event: {label}")

        elements = extract_triple_elements(summary)
        if not elements:
            print("❌ Skipping due to parse failure.")
            continue

        def map_uri(key: str) -> Optional[str]:
            value = elements.get(key)
            if value:
                return label_uri_map.get(value.lower().strip())
            return None

        triple = {
            "@id": eid,
            "@type": "Event",
            "@context": CONTEXT_URL,
            "hasActor": map_uri("actor"),
            "hasTarget": map_uri("target") or None,
            "hasLocation": map_uri("location"),
            "hasTime": map_uri("time"),
        }

        outpath = KG_OUTPUT_DIR / file.name
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(triple, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved triple: {outpath.name}")


# === MAIN ===
if __name__ == "__main__":
    label_uri_map = build_label_uri_map()
    create_event_triples(label_uri_map)
    print("🏁 All event triples generated.")
