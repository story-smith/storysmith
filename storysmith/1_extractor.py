# extract_entities_from_episodes.py

import json
import re
import uuid
from pathlib import Path
from typing import List

from config import (
    CHARACTER_BASE_URI,
    CHARACTER_DIR_RAW,
    CONTEXT_URL,
    EPISODE_DIR,
    EVENT_BASE_URI,
    EVENT_DIR_RAW,
    OPENAI_API_KEY,
    PLACE_BASE_URI,
    PLACE_DIR_RAW,
    TIMEPOINT_BASE_URI,
    TIMEPOINT_DIR_RAW,
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


def build_chain(context_url: str, target_type: str):
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


def summarize_features(label: str, raw_text: str, target_type: str) -> str:
    system_prompt = f"""
    You are an assistant summarizing distinguishing features of a {target_type} entity.
    Summarize as simply and distinctly as possible, in 1 or 2 concise sentences.
    Include specific traits or actions that can help differentiate it from others in the story.
    The summary should be optimized for use in embedding-based similarity matching (e.g. BERT).
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

    model = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY)
    return (
        (prompt | model).invoke({"label": label, "raw_text": raw_text}).content.strip()
    )


def extract_entities_from_episodes(
    episodes: List[Path], target_type: str, base_uri: str, output_dir: Path
):
    chain = build_chain(CONTEXT_URL, target_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entities = []

    for ep in episodes:
        epname = ep.stem
        print(f"🔍 Extracting {target_type} from {ep.name}")
        content = ep.read_text(encoding="utf-8", errors="replace").strip()

        try:
            extracted = chain.invoke({"input": content})
            for ent in extracted:
                label = ent.get("label", "")
                slug = (
                    re.sub(r"[^\w]+", "_", label.strip().lower())
                    or uuid.uuid4().hex[:8]
                )
                raw_excerpt = content[:1000].encode("utf-8", "replace").decode("utf-8")
                features_summary = summarize_features(label, raw_excerpt, target_type)

                ent["@id"] = f"{base_uri}{slug}"
                ent["@type"] = target_type
                ent["@context"] = CONTEXT_URL
                ent["episode"] = epname
                ent["_features"] = {
                    "label": label,
                    "episode": epname,
                    "summary": features_summary,
                }

                outpath = output_dir / f"{slug}.jsonld"
                with open(outpath, "w", encoding="utf-8", errors="replace") as f:
                    safe_ent = json.loads(json.dumps(ent, ensure_ascii=False))
                    json.dump(safe_ent, f, ensure_ascii=False, indent=2)
                print(f"📄 Saved: {outpath.name}")
                all_entities.append(ent)
        except Exception as e:
            print(f"❌ Error processing {ep.name}: {e}")

    return all_entities


if __name__ == "__main__":
    episodes = sorted(Path(EPISODE_DIR).glob("*.txt"))

    extract_entities_from_episodes(
        episodes, "Character", CHARACTER_BASE_URI, Path(CHARACTER_DIR_RAW)
    )
    extract_entities_from_episodes(
        episodes, "Place", PLACE_BASE_URI, Path(PLACE_DIR_RAW)
    )
    extract_entities_from_episodes(
        episodes, "TimePoint", TIMEPOINT_BASE_URI, Path(TIMEPOINT_DIR_RAW)
    )
    extract_entities_from_episodes(
        episodes, "Event", EVENT_BASE_URI, Path(EVENT_DIR_RAW)
    )

    print("✅ Extraction complete.")
