import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === Configurable Parameters ===
SIMILARITY_THRESHOLD = 0.90
LOG_FILE = "deduplication.log"

ENTITY_TYPES = {
    "characters": ("data/raw/characters", "data/characters"),
    "places": ("data/raw/places", "data/places"),
    "timepoints": ("data/raw/timepoints", "data/timepoints"),
}

# === Set up logging ===
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# === Load embedding model ===
model = SentenceTransformer("intfloat/multilingual-e5-large")


def get_embedding(text: str):
    return model.encode(f"query: {text}", normalize_embeddings=True)


def load_entities(entity_dir: Path):
    entities = []
    for path in entity_dir.glob("*.jsonld"):
        with open(path, "r", encoding="utf-8") as f:
            ent = json.load(f)
            label = ent.get("_features", {}).get("label", "")
            if label:
                entities.append((ent, label.strip()))
    logging.info(f"Loaded {len(entities)} entities from {entity_dir}")
    return entities


def deduplicate_entities_by_e5(entities, threshold: float):
    labels = [label for _, label in entities]
    embeddings = model.encode(
        [f"query: {label}" for label in labels], normalize_embeddings=True
    )

    clusters = []
    used = set()

    for i in tqdm(range(len(entities)), desc="🔗 Merging similar entities"):
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

    logging.info(f"Generated {len(clusters)} clusters from {len(entities)} entities")
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

        msg = f"✅ Saved merged entity: {outpath.name} (merged: {len(cluster)})"
        print(msg)
        logging.info(msg)


# === Run for all types ===
if __name__ == "__main__":
    for entity_type, (input_dir, output_dir) in ENTITY_TYPES.items():
        print(f"\n📦 Processing {entity_type}")
        raw_entities = load_entities(Path(input_dir))
        clusters = deduplicate_entities_by_e5(raw_entities, SIMILARITY_THRESHOLD)
        save_clusters(clusters, Path(output_dir))

    print("\n🏁 All entity types deduplicated successfully.")
