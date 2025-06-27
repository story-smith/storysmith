import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === 環境変数読み込み ===
load_dotenv()

# === ディレクトリ設定 ===
SHORTSTORY_DIR = Path("output/shortstories")
INTEGRATED_BASE_DIR = Path("output/integrated")
UPDATED_SHORTSTORY_DIR = Path("output/shortstories_integrated")

# 類似度の閾値
SIMILARITY_THRESHOLD = 0.90

# 統合対象スロットとスキーマタイプ
TARGET_SLOTS = {
    "character": "Person",
    "spatial": "Place",
    "mentions": "Product",
}

# === SBERTモデル読み込み ===
model = SentenceTransformer("intfloat/multilingual-e5-large")


# === Step 1: ShortStoryからエンティティ抽出 ===
def load_entities_from_shortstories(
    slot: str, shortstory_dir: Path
) -> List[Tuple[dict, str, Path]]:
    entities = []
    for path in shortstory_dir.glob("*.jsonld"):
        with open(path, "r", encoding="utf-8") as f:
            story = json.load(f)

        for ent in story.get(slot, []):
            name = ent.get("name", "").strip()
            if name:
                entities.append((ent, name, path))
    return entities


# === Step 2: 類似クラスタリング ===
def deduplicate_entities(entities: List[Tuple[dict, str, Path]], threshold: float):
    labels = [label for _, label, _ in entities]
    embeddings = model.encode(
        [f"query: {label}" for label in labels], normalize_embeddings=True
    )

    clusters = []
    used = set()
    for i in tqdm(range(len(entities)), desc="🔗 エンティティ統合"):
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


# === Step 3: 統合クラスタ保存 ===
def save_cluster(cluster: List[Tuple[dict, str, Path]], outdir: Path):
    rep = cluster[0][0]
    rep_id = rep.get("@id")
    rep["sameAs"] = [ent[0]["@id"] for ent in cluster[1:]] if len(cluster) > 1 else []
    slug = rep_id.rsplit("/", 1)[-1] if rep_id else f"ent_{uuid.uuid4().hex[:8]}"

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{slug}.jsonld"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"✅ 統合保存: {outpath.name}（{len(cluster)} 件）")


# === Step 4: sameAsマップ構築 ===
def build_sameas_map(integrated_dir: Path) -> Dict[str, str]:
    id_map = {}
    for path in integrated_dir.glob("*.jsonld"):
        with open(path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        rep_id = rep.get("@id")
        for old_id in rep.get("sameAs", []):
            id_map[old_id] = rep_id
    return id_map


# === Step 5: JSON内の @id をマップで更新 ===
def update_ids_in_slot(story: dict, slot: str, sameas_map: dict) -> bool:
    updated = False
    for ent in story.get(slot, []):
        if ent.get("@id") in sameas_map:
            ent["@id"] = sameas_map[ent["@id"]]
            updated = True
    return updated


# === Step 6: ShortStoryを更新して保存 ===
def update_shortstory_ids(
    story_path: Path, id_maps: Dict[str, Dict[str, str]], outdir: Path
):
    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    updated = False
    for slot in TARGET_SLOTS:
        updated |= update_ids_in_slot(story, slot, id_maps[slot])

    if updated:
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / story_path.name
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(story, f, ensure_ascii=False, indent=2)
        print(f"🔄 ShortStory更新: {outpath.name}")


# === メイン処理 ===
if __name__ == "__main__":
    print("🚀 統合処理開始\n")

    # スロットごとに処理
    id_maps = {}
    for slot, type_name in TARGET_SLOTS.items():
        print(f"\n📂 {slot} の処理中...")
        ent_dir = INTEGRATED_BASE_DIR / slot
        entities = load_entities_from_shortstories(slot, SHORTSTORY_DIR)
        clusters = deduplicate_entities(entities, SIMILARITY_THRESHOLD)
        for cluster in clusters:
            save_cluster(cluster, ent_dir)
        id_maps[slot] = build_sameas_map(ent_dir)

    # ShortStoryの更新
    print("\n🛠 ShortStory の @id を統合済みに差し替え中...")
    for story_path in SHORTSTORY_DIR.glob("*.jsonld"):
        update_shortstory_ids(story_path, id_maps, UPDATED_SHORTSTORY_DIR)

    print("\n✅ 統合処理と ShortStory 更新が完了しました！")
