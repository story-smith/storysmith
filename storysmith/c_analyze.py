import json
from datetime import datetime
from pathlib import Path

# === ディレクトリ設定 ===
INTEGRATED_SHORTSTORY_DIR = Path("output/shortstories_integrated")
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)


def extract_character_data(jsonld_path: Path) -> dict:
    """与えられたJSON-LDファイルからキャラクターのIDとメタデータを辞書で抽出"""
    with open(jsonld_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {char["@id"]: char for char in data.get("character", []) if "@id" in char}


def analyze_latest_episode(story_dir: Path, metrics_dir: Path):
    files = sorted(story_dir.glob("*.jsonld"), key=lambda p: p.stat().st_mtime)
    if len(files) < 2:
        print("⚠️ 比較対象が足りません（最低2ファイル必要）")
        return

    latest_file = files[-1]
    previous_files = files[:-1]

    # 最新ファイルのキャラクター抽出
    latest_data = extract_character_data(latest_file)
    latest_ids = set(latest_data.keys())

    # 以前のファイルすべてからキャラクター抽出（統合）
    all_previous_data = {}
    for path in previous_files:
        all_previous_data.update(extract_character_data(path))
    all_previous_ids = set(all_previous_data.keys())

    # 全キャラクターID
    all_story_ids = latest_ids | all_previous_ids

    # 分類
    new_ids = latest_ids - all_previous_ids
    continued_ids = latest_ids & all_previous_ids
    disappeared_ids = all_previous_ids - latest_ids

    # メトリクス計算
    novelty = len(new_ids) / len(latest_ids) if latest_ids else 0
    continuity = len(continued_ids) / len(latest_ids) if latest_ids else 0
    disappearance = (
        len(disappeared_ids) / len(all_previous_ids) if all_previous_ids else 0
    )

    # 結果構造体の構築
    result = {
        "story_summary": {
            "total_characters": len(all_story_ids),
            "all_characters": sorted(all_story_ids),
        },
        "latest_episode": {
            "file": latest_file.name,
            "timestamp": datetime.now().isoformat(),
            "characters": [latest_data[_id] for _id in sorted(latest_ids)],
            "new_characters": [latest_data[_id] for _id in sorted(new_ids)],
            "continued_characters": [latest_data[_id] for _id in sorted(continued_ids)],
            "disappeared_characters": [
                all_previous_data[_id] for _id in sorted(disappeared_ids)
            ],
            "metrics": {
                "novelty": round(novelty, 4),
                "continuity": round(continuity, 4),
                "disappearance_rate": round(disappearance, 4),
            },
        },
    }

    # 保存
    outpath = metrics_dir / "character_metrics.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✅ メトリクス保存完了 →", outpath.resolve())


# エントリーポイント
if __name__ == "__main__":
    analyze_latest_episode(INTEGRATED_SHORTSTORY_DIR, METRICS_DIR)
