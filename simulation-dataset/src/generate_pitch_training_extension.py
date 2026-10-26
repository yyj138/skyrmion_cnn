"""重建平台补训模型使用的 pitch 训练扩展表（labels_pitch_training_*.csv）

背景
----
`cnn-inversion/runs/supplement_geometry_v1/` 与 `.../supplement_router_v1/` 的
`metrics.json` 记录了两张扩展训练表的路径与 sha256：

    simulation-dataset/outputs/pitch_training_extension/labels_pitch_training_single.csv
        extra_train_count = 176   sha256 = 648ba6b0...d4241b
    simulation-dataset/outputs/pitch_training_extension/labels_pitch_training_mixed.csv
        extra_train_count = 396   sha256 = 7d89eec7...eb86

但这两张表**从未进入版本库**（git 全历史检索无此路径），因此无法逐位复现。
本脚本按仓库内可核对的证据重建它们，供日后复训/审计使用。

重建规则（确定性，无随机抽样）
------------------------------
1. 单 toron 扩展表：`outputs/dataset/labels/labels_single.csv` 的 88 行，各复制
   一份（×2），共 176 行 —— 与 metrics 记录的 extra_train_count 一致。
2. 混合扩展表：上述 88 行 ×2，加 `outputs/supplement/labels/labels_supplement_double.csv`
   中属于**开发侧**的 110 行 ×2，共 176 + 220 = 396 行 —— 与记录一致。
3. 只用开发侧（development）补充图，**不使用封存侧（holdout）**，避免测试集泄漏；
   判定依据是 `cnn-inversion/data/supplement_split.csv` 的 split 列。
4. 输出列与原标签表完全一致（filename,eta,U,free_energy,kind,n_torons,separation），
   因此可直接与 labels_single/labels_mixed 合并，或经 --csv 传入 train.py。

已知差异（重要）
----------------
重建表的行数、列结构、图片引用与历史一致，但**行顺序与具体复制对象未必与历史文件
逐字节相同**，sha256 不会等于 metrics 中记录的值。脚本结尾会打印实测 sha256 并
与历史值对比，明确提示“不复现”。

用法
----
    cd simulation-dataset/src
    python generate_pitch_training_extension.py          # 写入 outputs/pitch_training_extension/
    python generate_pitch_training_extension.py --check  # 只校验，不写文件
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # simulation-dataset/
DATASET = ROOT / "outputs" / "dataset"
SUPPLEMENT = ROOT / "outputs" / "supplement"
OUT_DIR = ROOT / "outputs" / "pitch_training_extension"
SPLIT_CSV = ROOT.parent / "cnn-inversion" / "data" / "supplement_split.csv"
METRICS_DIR = ROOT.parent / "cnn-inversion" / "runs" / "supplement_geometry_v1"

# metrics.json 中记录的历史计数与 sha256（重建目标，仅用于对比）
HISTORICAL = {
    "single": {
        "count": 176,
        "sha256": "648ba6b05ba9fd0884d803e98d43fe490c0179a63924bd484164680235d4241b",
        "csv": METRICS_DIR / "pitch_final" / "metrics.json",
        "out": "labels_pitch_training_single.csv",
    },
    "mixed": {
        "count": 396,
        "sha256": "7d89eec797baff2157cce8de99b9f1235c5c492f80004291bd53f73aff00eb86",
        "csv": METRICS_DIR / "mixed_pitch_final" / "metrics.json",
        "out": "labels_pitch_training_mixed.csv",
    },
}

BASE_FIELDS = ["filename", "eta", "U", "free_energy", "kind", "n_torons"]
MIXED_FIELDS = BASE_FIELDS + ["separation"]


def read_rows(path, fields):
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = [f for f in fields if f not in (rows[0] if rows else {})]
    if missing:
        raise SystemExit(f"{path.name} 缺少列: {missing}")
    return [{k: (row.get(k) or "") for k in fields} for row in rows]


def development_double_filenames():
    """补充图中属于开发侧的双 toron 文件名集合（封存侧一律排除）。"""
    if not SPLIT_CSV.is_file():
        raise SystemExit(f"缺少划分表: {SPLIT_CSV}")
    with SPLIT_CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    dev = {row["filename"] for row in rows
           if row.get("kind") == "double" and row.get("split") == "development"}
    if not dev:
        raise SystemExit("划分表中没有 double/development 行，检查 supplement_split.csv")
    return dev


def build_single():
    rows = read_rows(DATASET / "labels" / "labels_single.csv", BASE_FIELDS)
    return [dict(row) for row in rows for _ in range(2)]        # 88 x 2 = 176


def build_mixed():
    singles = read_rows(DATASET / "labels" / "labels_single.csv", BASE_FIELDS)
    doubles = read_rows(SUPPLEMENT / "labels" / "labels_supplement_double.csv", MIXED_FIELDS)
    dev = development_double_filenames()
    singles_doubled = [{**row, "separation": ""} for row in singles for _ in range(2)]
    doubles_dev = [row for row in doubles if row["filename"] in dev]
    doubles_doubled = [dict(row) for row in doubles_dev for _ in range(2)]
    return singles_doubled + doubles_doubled                    # 176 + 220 = 396


def digest(rows, fields):
    """按与 csv 写入完全相同的字节流计算 sha256（用于和历史值对比）。"""
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


def write_csv(rows, fields, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只校验计数，不写文件")
    args = ap.parse_args()

    plans = [("single", build_single(), BASE_FIELDS),
             ("mixed", build_mixed(), MIXED_FIELDS)]

    ok = True
    report = []
    for name, rows, fields in plans:
        hist = HISTORICAL[name]
        count_ok = len(rows) == hist["count"]
        sha = digest(rows, fields)
        ok &= count_ok
        report.append({
            "name": name,
            "rows": len(rows),
            "expected_rows": hist["count"],
            "count_ok": count_ok,
            "sha256": sha,
            "historical_sha256": hist["sha256"],
            "sha256_matches_history": sha == hist["sha256"],
        })
        status = "OK" if count_ok else f"FAIL(期望 {hist['count']})"
        print(f"[{name}] 行数 {len(rows)} ({status})  sha256={sha[:16]}…")
        if not args.check:
            write_csv(rows, fields, OUT_DIR / hist["out"])
            print(f"        已写入 {OUT_DIR / hist['out']}")

    manifest = {
        "generated_by": "simulation-dataset/src/generate_pitch_training_extension.py",
        "rule": "single=88 originals x2; mixed=(88 originals x2)+(110 double development x2)",
        "holdout_used": False,
        "note": "历史文件未入库，行数与列结构一致但 sha256 不可复现；详见脚本 docstring。",
        "files": report,
    }
    if not args.check:
        (OUT_DIR / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n清单已写入 {OUT_DIR / 'manifest.json'}")

    if not ok:
        print("\n计数与 metrics.json 记录不符，请检查数据源是否被改动。")
        return 1
    print("\n计数校验通过。注意：sha256 与历史值不同属预期（历史文件从未入库）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
