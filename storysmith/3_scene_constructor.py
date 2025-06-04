import json
import re
from pathlib import Path
from typing import List

from config import (
    CHARACTER_DIR,
    CONTEXT_URL,
    EPISODE_DIR,
    PLACE_DIR,
    TIMEPOINT_DIR,
)

SCENE_DIR = Path("data/scenes")
SCENE_DIR.mkdir(parents=True, exist_ok=True)


def load_entities(directory: Path) -> dict:
    uri_to_label = {}
    for file in directory.glob("*.jsonld"):
        with open(file, "r", encoding="utf-8") as f:
            ent = json.load(f)
            uri = ent.get("@id")
            label = ent.get("_features", {}).get("label") or ent.get("label", "")
            if uri and label:
                uri_to_label[label.lower()] = uri
    return uri_to_label


def detect_mentions(text: str, label_to_uri: dict) -> List[str]:
    found_uris = set()
    for label, uri in label_to_uri.items():
        if re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE):
            found_uris.add(uri)
    return sorted(found_uris)


def create_scene_id(epname: str, index: int) -> str:
    return f"https://story-smith.github.io/storysmith/scenes/scene_{epname}_{index:03}"


def build_scenes():
    char_ents = load_entities(Path(CHARACTER_DIR))
    place_ents = load_entities(Path(PLACE_DIR))
    time_ents = load_entities(Path(TIMEPOINT_DIR))

    episodes = sorted(Path(EPISODE_DIR).glob("*.txt"))

    for ep in episodes:
        epname = ep.stem
        text = ep.read_text(encoding="utf-8", errors="replace")

        # シンプルに N 分割して Scene を仮生成（改行や長さで分割も応用可）
        chunks = [text[i : i + 800] for i in range(0, len(text), 800)]
        for idx, chunk in enumerate(chunks):
            chars = detect_mentions(chunk, char_ents)
            places = detect_mentions(chunk, place_ents)
            times = detect_mentions(chunk, time_ents)

            if not (chars or places or times):
                continue  # 空のSceneはスキップ

            scene = {
                "@context": CONTEXT_URL,
                "@id": create_scene_id(epname, idx),
                "@type": "Scene",
                "c": chars,
                "p": places,
                "t": times,
                "seq": chunk.strip()[:300],
                "episode": epname,
            }

            outpath = SCENE_DIR / f"scene_{epname}_{idx:03}.jsonld"
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(scene, f, ensure_ascii=False, indent=2)
            print(f"🎬 Scene saved: {outpath.name}")


if __name__ == "__main__":
    build_scenes()
    print("✅ Scene construction complete.")
