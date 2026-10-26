"""Export Geometry of Truth datasets (Marks & Tegmark, 2024) to liu_per_domain_prompts/got_<name>_{true,false}.txt.

git clone https://github.com/saprmarks/geometry-of-truth.git
python export_geometry_of_truth.py --got-dir geometry-of-truth --out-dir liu_per_domain_prompts
"""

import argparse
import csv
import random
from pathlib import Path

DEFAULT_DATASETS = [
    "neg_cities",
    "neg_sp_en_trans",
    "sp_en_trans",
    "larger_than",
    "smaller_than",
    "likely",
]

# Overlap with Liu domains already in use -- not a held-out test.
OVERLAPPING = {"cities", "companies", "counterfact"}


def load_csv(path: Path) -> tuple[list[str], list[str]]:
    """(true statements, false statements)"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")

    cols = set(rows[0])
    stmt_col = next((c for c in ("statement", "claim", "text") if c in cols), None)
    label_col = next((c for c in ("label", "truth", "is_true") if c in cols), None)
    if stmt_col is None or label_col is None:
        raise SystemExit(
            f"{path}: could not find statement/label columns.\n"
            f"Columns present: {sorted(cols)}\n"
            "Adjust the column names in load_csv()."
        )

    true_s: list[str] = []
    false_s: list[str] = []
    for r in rows:
        text = " ".join(str(r[stmt_col]).split())  # collapse any stray newlines
        if not text:
            continue
        lab = str(r[label_col]).strip()
        if lab in ("1", "1.0", "True", "true"):
            true_s.append(text)
        elif lab in ("0", "0.0", "False", "false"):
            false_s.append(text)
    return true_s, false_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--got-dir", required=True, help="Path to a clone of github.com/saprmarks/geometry-of-truth"
    )
    ap.add_argument("--out-dir", required=True, help="Your liu_per_domain_prompts directory")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument(
        "--n-per-side",
        type=int,
        default=15,
        help="True AND false statements per dataset. 15/15=30 matches the "
        "Liu domains, so no dataset dominates the training folds "
        "(the TruthfulQA export previously over-sampled 6.7x).",
    )
    ap.add_argument("--prefix", default="got_")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.got_dir) / "datasets"
    if not data_dir.is_dir():
        raise SystemExit(f"No datasets/ directory under {args.got_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for name in args.datasets:
        if name in OVERLAPPING:
            print(
                f"[skip] {name}: overlaps a Liu domain already in use "
                f"-- would not be a held-out test"
            )
            continue
        path = data_dir / f"{name}.csv"
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue

        true_s, false_s = load_csv(path)
        rng.shuffle(true_s)
        rng.shuffle(false_s)
        n = min(args.n_per_side, len(true_s), len(false_s))
        true_s, false_s = true_s[:n], false_s[:n]

        tag = f"{args.prefix}{name}"
        (out_dir / f"{tag}_true.txt").write_text("\n".join(true_s) + "\n")
        (out_dir / f"{tag}_false.txt").write_text("\n".join(false_s) + "\n")
        print(f"[ok]   {tag}: {len(true_s)} true / {len(false_s)} false")
        print(f"         true : {true_s[0]}")
        print(f"         false: {false_s[0]}")


if __name__ == "__main__":
    main()
