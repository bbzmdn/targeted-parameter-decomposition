"""Export TruthfulQA to liu_per_domain_prompts/truthfulqa_{true,false}.txt as "<question> <answer>." lines.

python export_truthfulqa.py --out-dir liu_per_domain_prompts
"""

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-examples", type=int, default=50)
    args = ap.parse_args()

    ds = load_dataset("truthful_qa", "generation", split="validation")
    assert isinstance(ds, Dataset)
    true_lines: list[str] = []
    false_lines: list[str] = []
    for row in ds.to_list():
        # "I have no comment" answers are refusals, not true claims
        corrects = [
            a.strip() for a in row["correct_answers"] if a.strip() and "no comment" not in a.lower()
        ]
        incorrects = [a.strip() for a in row["incorrect_answers"] if a.strip()]
        if not corrects or not incorrects:
            continue
        question = row["question"].strip()
        true_lines.append(f"{question} {corrects[0].rstrip('.')}.")
        false_lines.append(f"{question} {incorrects[0].rstrip('.')}.")
        if len(true_lines) == args.n_examples:
            break

    out_dir = Path(args.out_dir)
    (out_dir / "truthfulqa_true.txt").write_text("\n".join(true_lines) + "\n")
    (out_dir / "truthfulqa_false.txt").write_text("\n".join(false_lines) + "\n")
    print(f"Wrote {len(true_lines)} true / {len(false_lines)} false statements to {out_dir}")


if __name__ == "__main__":
    main()
