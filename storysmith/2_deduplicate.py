import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment variables
load_dotenv()

# === 設定 ===
ENTITY_TYPES = {
    "characters": (
        "output/raw/a-fish-story-story/characters",  # raw_dir: セクション別エンティティがここにある
        "output/integrated/a-fish-story-story/characters",  # out_dir: 統合結果の保存先も同じ構造で使う
    ),
    "places": (
        "output/raw/a-fish-story-story/places",
        "output/integrated/a-fish-story-story/places",
    ),
}

SIMILARITY_THRESHOLD = 0.90

# === モデルロード ===
model = SentenceTransformer("intfloat/multilingual-e5-large")


def load_all_entities(entity_dir: Path) -> List[Tuple[dict, str, str]]:
    """全セクションを横断してエンティティを読み込む（セクション情報付き）"""
    entities = []
    for path in entity_dir.glob("section_*/**/*.jsonld"):
        with open(path, "r", encoding="utf-8") as f:
            ent = json.load(f)
            label = ent.get("_features", {}).get("label", "").strip()
            section = ent.get("episode", "").rsplit("_", 1)[-1]
            if label and section:
                entities.append((ent, label, section))
    return entities


def deduplicate_entities_by_label_embedding(entities, threshold: float):
    labels = [label for _, label, _ in entities]
    embeddings = model.encode(
        [f"query: {label}" for label in labels], normalize_embeddings=True
    )

    clusters = []
    used = set()

    for i in tqdm(range(len(entities)), desc="🔗 セクション横断ラベル統合"):
        if i in used:
            continue
        cluster = [entities[i]]
        used.add(i)

        for j in range(i + 1, len(entities)):
            if j in used:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                cluster.append(entities[j])
                used.add(j)

        clusters.append(cluster)

    return clusters


def save_cluster_to_sections(cluster: List[Tuple[dict, str, str]], output_base: Path):
    rep = cluster[0][0]
    rep_id = rep["@id"]
    rep["sameAs"] = [ent[0]["@id"] for ent in cluster[1:]] if len(cluster) > 1 else []
    slug = rep_id.rsplit("/", 1)[-1]

    sections = {section for _, _, section in cluster}
    for section in sections:
        output_dir = output_base / f"section_{section}"
        output_dir.mkdir(parents=True, exist_ok=True)
        outpath = output_dir / f"{slug}.jsonld"
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"✅ 保存: {outpath.name}（section {section}, 同一:{len(cluster)}）")


# === 実行 ===
if __name__ == "__main__":
    for type_name, (raw_dir, out_dir) in ENTITY_TYPES.items():
        print(f"\n📦 統合中: {type_name}")
        entities = load_all_entities(Path(raw_dir))
        clusters = deduplicate_entities_by_label_embedding(
            entities, SIMILARITY_THRESHOLD
        )
        for cluster in clusters:
            save_cluster_to_sections(cluster, Path(out_dir))

    print("\n🏁 統合＋出現セクションごとの保存完了！")
