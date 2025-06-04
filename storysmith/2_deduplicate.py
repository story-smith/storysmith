import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === 設定 ===
ENTITY_TYPES = {
    "characters": ("data/raw/characters", "data/integrated/characters"),
    "places": ("data/raw/places", "data/integrated/places"),
    "timepoints": ("data/raw/timepoints", "data/integrated/timepoints"),
    "events": ("data/raw/events", "data/integrated/events"),
}
SIMILARITY_THRESHOLD = 0.90

# === モデルロード ===
model = SentenceTransformer("intfloat/multilingual-e5-large")


def load_entities(entity_dir: Path) -> List[Tuple[dict, str]]:
    entities = []
    for path in entity_dir.glob("*.jsonld"):
        with open(path, "r", encoding="utf-8") as f:
            ent = json.load(f)
            label = ent.get("_features", {}).get("label", "").strip()
            if label:
                entities.append((ent, label))
    return entities


def deduplicate_entities_by_label_embedding(entities, threshold: float):
    labels = [label for _, label in entities]
    embeddings = model.encode(
        [f"query: {label}" for label in labels], normalize_embeddings=True
    )

    clusters = []
    used = set()

    for i in tqdm(range(len(entities)), desc="🔗 意味的ラベル統合"):
        if i in used:
            continue
        cluster = [entities[i][0]]
        used.add(i)

        for j in range(i + 1, len(entities)):
            if j in used:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                cluster.append(entities[j][0])
                used.add(j)

        clusters.append(cluster)

    return clusters


def save_clusters(clusters, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster in clusters:
        rep = cluster[0]
        rep_id = rep["@id"]
        rep["sameAs"] = [c["@id"] for c in cluster[1:]] if len(cluster) > 1 else []

        slug = rep_id.rsplit("/", 1)[-1]
        outpath = output_dir / f"{slug}.jsonld"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)

        print(f"✅ 統合: {outpath.name} （同一: {len(cluster)}）")


# === 実行 ===
if __name__ == "__main__":
    for type_name, (raw_dir, out_dir) in ENTITY_TYPES.items():
        print(f"\n📦 統合中: {type_name}")
        entities = load_entities(Path(raw_dir))
        clusters = deduplicate_entities_by_label_embedding(
            entities, SIMILARITY_THRESHOLD
        )
        save_clusters(clusters, Path(out_dir))

    print("\n🏁 全てのエンティティ統合完了！")
