import json
from datetime import datetime
from pathlib import Path
from typing import Dict

# === ディレクトリ設定 ===
INTEGRATED_SHORTSTORY_DIR = Path("output/shortstories_integrated")
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

# === 対象スロット定義 ===
ENTITY_SLOTS = {"character": "characters", "spatial": "places", "mentions": "items"}


def extract_entities(jsonld_path: Path, slot: str) -> Dict[str, dict]:
    """指定スロットのエンティティを辞書形式で抽出"""
    with open(jsonld_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get(slot, [])
    if isinstance(entries, dict):  # spatialが1件だけだった過去形式への対応
        entries = [entries]

    return {entry["@id"]: entry for entry in entries if "@id" in entry}


def analyze_latest_episode(story_dir: Path, metrics_dir: Path):
    files = sorted(story_dir.glob("*.jsonld"), key=lambda p: p.stat().st_mtime)
    if len(files) < 2:
        print("⚠️ 比較対象が足りません（最低2ファイル必要）")
        return

    latest_file = files[-1]
    previous_files = files[:-1]

    overall_result = {
        "summary": {
            "latest_file": latest_file.name,
            "timestamp": datetime.now().isoformat(),
        },
        "slots": {},
    }

    for slot, label in ENTITY_SLOTS.items():
        # 最新ファイルから抽出
        latest_data = extract_entities(latest_file, slot)
        latest_ids = set(latest_data.keys())

        # 過去すべてから統合
        previous_data = {}
        for file in previous_files:
            previous_data.update(extract_entities(file, slot))
        previous_ids = set(previous_data.keys())

        # 分析
        new_ids = latest_ids - previous_ids
        continued_ids = latest_ids & previous_ids
        disappeared_ids = previous_ids - latest_ids
        all_ids = latest_ids | previous_ids

        novelty = len(new_ids) / len(latest_ids) if latest_ids else 0
        continuity = len(continued_ids) / len(latest_ids) if latest_ids else 0
        disappearance = len(disappeared_ids) / len(previous_ids) if previous_ids else 0

        overall_result["slots"][slot] = {
            "total_count": len(all_ids),
            "all_ids": sorted(all_ids),
            "latest_entities": [latest_data[_id] for _id in sorted(latest_ids)],
            "new_entities": [latest_data[_id] for _id in sorted(new_ids)],
            "continued_entities": [latest_data[_id] for _id in sorted(continued_ids)],
            "disappeared_entities": [
                previous_data[_id] for _id in sorted(disappeared_ids)
            ],
            "metrics": {
                "novelty": round(novelty, 4),
                "continuity": round(continuity, 4),
                "disappearance_rate": round(disappearance, 4),
            },
        }

    # 保存
    outpath = metrics_dir / "entity_metrics.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(overall_result, f, ensure_ascii=False, indent=2)

    print(f"✅ メトリクス保存完了 → {outpath.resolve()}")


# エントリーポイント
if __name__ == "__main__":
    analyze_latest_episode(INTEGRATED_SHORTSTORY_DIR, METRICS_DIR)
