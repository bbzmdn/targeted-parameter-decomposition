"""Build the tPD target files prompts/<target>_<polarity>.txt (one bare statement per line) and prompts/tqa_split.json.

python build_prompts.py  # needs internet for the TruthfulQA CSV
"""

import csv
import io
import json
import random
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
PROMPTS_DIR = HERE / "prompts"
TQA_EVAL_FILES = [
    HERE / "liu_per_domain_prompts" / f"truthfulqa_{p}.txt" for p in ("true", "false")
]
TRUTHFULQA_CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
N_TQA_QUESTIONS = 180  # matches the size of the liu sets
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 128  # task_config.max_seq_len in tpd_config.yaml
SEED = 0
# Eval questions that the current TruthfulQA CSV words differently: eval wording -> CSV wording
TQA_EVAL_REWORDED = {
    "Why do chameleons change colors?": "What is the primary reason that chameleons change colors?",
}

Statements = dict[str, list[str]]  # polarity ("true" / "false") -> lines


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def capitals_statements() -> Statements:
    pairs = [json.loads(line) for line in read_lines(DATA_DIR / "capitals_contrast_pairs.jsonl")]
    true_lines: list[str] = []
    false_lines: list[str] = []
    seen_questions: set[str] = set()
    for pair in pairs:
        if pair["question"] in seen_questions:
            continue
        seen_questions.add(pair["question"])
        sentence = pair["fact_sentence"]
        assert pair["correct_answer"] in sentence, pair
        true_lines.append(sentence)
        false_lines.append(sentence.replace(pair["correct_answer"], pair["wrong_answer"]))
    return {"true": true_lines, "false": false_lines}


def liu_statements() -> Statements:
    return {p: read_lines(DATA_DIR / f"liu_{p}.txt") for p in ("true", "false")}


def as_sentence(text: str) -> str:
    text = " ".join(text.split())
    return text if text[-1] in ".!?" else text + "."


def tqa_statements() -> tuple[Statements, dict[str, list[str]]]:
    with urllib.request.urlopen(TRUTHFULQA_CSV_URL) as response:
        rows = list(csv.DictReader(io.StringIO(response.read().decode())))
    rows = [
        r
        for r in rows
        if r["Best Answer"].strip()
        and r["Best Incorrect Answer"].strip()
        and "no comment" not in r["Best Answer"].lower()
    ]

    eval_lines = [line for path in TQA_EVAL_FILES for line in read_lines(path)]
    for eval_wording, csv_wording in TQA_EVAL_REWORDED.items():
        eval_lines = [line.replace(eval_wording, csv_wording) for line in eval_lines]
    eval_lines = [line.lower() for line in eval_lines]
    is_eval = {
        r["Question"]: any(line.startswith(r["Question"].lower()) for line in eval_lines)
        for r in rows
    }
    n_eval_matched = sum(any(line.startswith(q.lower()) for q in is_eval) for line in eval_lines)
    assert n_eval_matched == len(eval_lines), (
        f"only {n_eval_matched}/{len(eval_lines)} TruthfulQA eval lines match a CSV question"
    )

    candidates = sorted(q for q, excluded in is_eval.items() if not excluded)
    random.Random(SEED).shuffle(candidates)
    train = set(candidates[:N_TQA_QUESTIONS])
    by_question = {r["Question"]: r for r in rows}

    def line(question: str, answer_col: str) -> str:
        return f"{' '.join(question.split())} {as_sentence(by_question[question][answer_col])}"

    statements = {
        "true": [line(q, "Best Answer") for q in sorted(train)],
        "false": [line(q, "Best Incorrect Answer") for q in sorted(train)],
    }
    split = {
        "train": sorted(train),
        "eval_excluded": sorted(q for q, excluded in is_eval.items() if excluded),
    }
    return statements, split


def main() -> None:
    tqa, tqa_split = tqa_statements()
    targets: dict[str, Statements] = {
        "capitals": capitals_statements(),
        "liu": liu_statements(),
        "tqa": tqa,
    }
    combined: Statements = {}
    for polarity in ("true", "false"):
        lines = [line for statements in targets.values() for line in statements[polarity]]
        random.Random(SEED).shuffle(lines)
        combined[polarity] = lines
    targets["capitals_liu_tqa"] = combined

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    PROMPTS_DIR.mkdir(exist_ok=True)
    for target, statements in targets.items():
        for polarity, lines in statements.items():
            longest = max(len(ids) for ids in tokenizer(lines)["input_ids"])  # pyright: ignore[reportCallIssue]
            assert longest <= MAX_SEQ_LEN, f"{target}_{polarity}: {longest} tokens > {MAX_SEQ_LEN}"
            path = PROMPTS_DIR / f"{target}_{polarity}.txt"
            path.write_text("\n".join(lines) + "\n")
            print(f"{path.relative_to(HERE)}: {len(lines)} statements, longest {longest} tokens")
    (PROMPTS_DIR / "tqa_split.json").write_text(json.dumps(tqa_split, indent=1) + "\n")


if __name__ == "__main__":
    main()
