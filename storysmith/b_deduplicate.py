import json
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment variables
load_dotenv()

# === 設定 ===
SHORTSTORY_DIR = Path("output/shortstories")  # 出力済み shortstory のディレクトリ
INTEGRATED_DIR = Path("output/integrated/characters")  # 正規化後の character 統合先
SIMILARITY_THRESHOLD = 0.90

# === モデルロード ===
model = SentenceTransformer("intfloat/multilingual-e5-large")


def load_all_characters_from_shortstories(
    shortstory_dir: Path,
) -> List[Tuple[dict, str, Path]]:
    entities = []
    for path in shortstory_dir.glob("*.jsonld"):
        with open(path, "r", encoding="utf-8") as f:
            story = json.load(f)
            characters = story.get("character", [])
            for char in characters:
                label = char.get("label", "").strip()
                if label:
                    entities.append((char, label, path))
    return entities


def deduplicate_entities_by_label_embedding(entities, threshold: float):
    labels = [label for _, label, _ in entities]
    embeddings = model.encode(
        [f"query: {label}" for label in labels], normalize_embeddings=True
    )

    clusters = []
    used = set()

    for i in tqdm(range(len(entities)), desc="🔗 キャラクター名の統合"):
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


def save_cluster_to_integrated_dir(cluster: List[Tuple[dict, str, Path]], outdir: Path):
    rep = cluster[0][0]
    rep_id = rep.get("@id")
    rep["sameAs"] = [ent[0]["@id"] for ent in cluster[1:]] if len(cluster) > 1 else []
    slug = rep_id.rsplit("/", 1)[-1] if rep_id else f"char_{uuid.uuid4().hex[:8]}"

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{slug}.jsonld"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"✅ 保存: {outpath.name}（統合数: {len(cluster)}）")


# === 実行 ===
if __name__ == "__main__":
    print("\n📦 ShortStory からキャラクター統合処理を開始")
    entities = load_all_characters_from_shortstories(SHORTSTORY_DIR)
    clusters = deduplicate_entities_by_label_embedding(entities, SIMILARITY_THRESHOLD)
    for cluster in clusters:
        save_cluster_to_integrated_dir(cluster, INTEGRATED_DIR)

    print("\n🏁 キャラクター統合と保存が完了しました！")
