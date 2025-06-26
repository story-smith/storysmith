import json
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === 環境変数読み込み ===
load_dotenv()

# === ディレクトリ設定 ===
SHORTSTORY_DIR = Path("output/shortstories")
INTEGRATED_DIR = Path("output/integrated/characters")
UPDATED_SHORTSTORY_DIR = Path("output/shortstories_integrated")

# 類似度の閾値
SIMILARITY_THRESHOLD = 0.90

# === モデルロード ===
model = SentenceTransformer("intfloat/multilingual-e5-large")


# === Step 1: ShortStoryからキャラクター抽出 ===
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


# === Step 2: 類似キャラクターをクラスタリング ===
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


# === Step 3: 統合済みキャラを保存 ===
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


# === Step 4: sameAs マップを構築（old_id → rep_id） ===
def build_sameas_map(integrated_dir: Path) -> dict:
    id_map = {}
    for file in integrated_dir.glob("*.jsonld"):
        with open(file, "r", encoding="utf-8") as f:
            rep = json.load(f)
            rep_id = rep.get("@id")
            for old_id in rep.get("sameAs", []):
                id_map[old_id] = rep_id
    return id_map


# === Step 5: ShortStory を更新して再保存 ===
def update_shortstory_with_integrated_ids(
    story_path: Path, sameas_map: dict, outdir: Path
):
    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    updated = False
    for char in story.get("character", []):
        old_id = char.get("@id")
        if old_id in sameas_map:
            char["@id"] = sameas_map[old_id]
            updated = True

    if updated:
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / story_path.name
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(story, f, ensure_ascii=False, indent=2)
        print(f"🔄 ShortStory 更新保存: {outpath.name}")


# === 実行本体 ===
if __name__ == "__main__":
    print("\n📦 ShortStory からキャラクター統合処理を開始")

    # キャラクター抽出と統合
    entities = load_all_characters_from_shortstories(SHORTSTORY_DIR)
    clusters = deduplicate_entities_by_label_embedding(entities, SIMILARITY_THRESHOLD)
    for cluster in clusters:
        save_cluster_to_integrated_dir(cluster, INTEGRATED_DIR)

    print("\n🔁 ShortStory のキャラクターIDを統合済みに差し替え中...")
    sameas_map = build_sameas_map(INTEGRATED_DIR)
    for path in SHORTSTORY_DIR.glob("*.jsonld"):
        update_shortstory_with_integrated_ids(path, sameas_map, UPDATED_SHORTSTORY_DIR)

    print("\n🏁 統合処理と ShortStory 更新が完了しました！")
