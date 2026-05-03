"""
Convert raw task data into evidence-aware training examples
for the verifier / reward model.

Input:
  data/<lang>/<lang>_train.json

Output:
  output/training_data_for_RM/<lang>_train_with_evidence.jsonl

Each output record has:
  sample_id
  query_id
  input_text : "Claim: ...\nEvidence: ...\nVerdict: ...\nJustification: ..."
  Claim
  Evidence
  Justification
  Label
  Verdict
  Class
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import numpy as np


UNKNOWN_LIMIT = 150
RANDOM_SEED = 42


def remove_label_pattern(text: str) -> str:
    text = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting|Supports|Refutes))",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    return text.replace("\n", " ")


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()

    mapping = {
        "supports": "true",
        "support": "true",
        "refutes": "false",
        "refute": "false",
        "true": "true",
        "false": "false",
        "conflicting": "conflicting",
        "unknown": "unknown",
    }

    return mapping.get(label, label)


def extract_evidence(row: pd.Series) -> str:
    possible_keys = [
        "evidences",
        "evidence",
        "Evidence",
        "relevant_evidence",
        "Relevant_evidence",
        "context",
        "Context",
        "gold_evidence",
        "Gold_evidence",
    ]

    for key in possible_keys:
        if key in row.index and row[key]:
            value = row[key]

            if isinstance(value, list):
                return " ".join(map(str, value))

            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)

            return str(value)

    return ""


def sample_training_examples(row: pd.Series, unknown_state: dict) -> list[int]:
    label = normalize_label(row["label"])
    verdict_list = [normalize_label(v) for v in row["Verdict_list"]]

    correct_indices = [i for i, v in enumerate(verdict_list) if v == label]

    true_indices = [
        i for i, v in enumerate(verdict_list)
        if v == "true" and v != label
    ]

    false_indices = [
        i for i, v in enumerate(verdict_list)
        if v == "false" and v != label
    ]

    conflicting_indices = [
        i for i, v in enumerate(verdict_list)
        if v == "conflicting" and v != label
    ]

    unknown_indices = [
        i for i, v in enumerate(verdict_list)
        if v == "unknown"
    ]

    selected_indices = []

    if len(correct_indices) >= 2:
        selected_indices.extend(random.sample(correct_indices, 2))
        num_remaining = 4
    elif len(correct_indices) == 1:
        selected_indices.append(correct_indices[0])
        num_remaining = 5
    else:
        num_remaining = 6

    wrong_indices = []

    if label != "true" and true_indices:
        wrong_indices.append(random.choice(true_indices))

    if label != "false" and false_indices:
        wrong_indices.append(random.choice(false_indices))

    if label != "conflicting" and conflicting_indices:
        wrong_indices.append(random.choice(conflicting_indices))

    wrong_indices = list(set(wrong_indices))

    needed = num_remaining - len(wrong_indices)

    all_wrong = true_indices + false_indices + conflicting_indices
    random.shuffle(all_wrong)

    wrong_indices.extend(all_wrong[:needed])
    selected_indices.extend(wrong_indices[:num_remaining])

    if not selected_indices and unknown_indices:
        selected_indices = random.sample(
            unknown_indices,
            min(5, len(unknown_indices)),
        )

    elif unknown_indices and unknown_state["counter"] < UNKNOWN_LIMIT:
        if selected_indices:
            selected_indices.pop()

        selected_indices.append(random.choice(unknown_indices))
        unknown_state["counter"] += 1

    return selected_indices


def build_input_text(
    claim: str,
    evidence: str,
    verdict: str,
    justification: str,
) -> str:
    return (
        f"Claim: {claim}\n"
        f"Evidence: {evidence}\n"
        f"Verdict: {verdict}\n"
        f"Justification: {justification}"
    )


def build_training_data(input_path: str, output_path: str) -> None:
    random.seed(RANDOM_SEED)

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    data = pd.DataFrame(raw)

    unknown_state = {"counter": 0}

    data["sampled_indices"] = data.apply(
        lambda row: sample_training_examples(row, unknown_state),
        axis=1,
    )

    final_training_data = []

    for idx in range(len(data)):
        item = data.loc[idx]

        claim = str(item["claim"])
        evidence = extract_evidence(item)
        label = normalize_label(item["label"])

        for decoding_idx, trace_idx in enumerate(item["sampled_indices"]):
            justification = remove_label_pattern(
                item["Reasoning_traces"][trace_idx]
            )

            verdict = normalize_label(
                item["Verdict_list"][trace_idx]
            )

            qid = item["query_id"] if "query_id" in item.index else idx
            sample_id = f"{qid}_{chr(97 + decoding_idx)}"

            if len(justification.split()) < 3:
                continue

            input_text = build_input_text(
                claim=claim,
                evidence=evidence,
                verdict=verdict,
                justification=justification,
            )

            # TeacherScore is the soft distillation target.
            # If the raw data has score_list, use it.
            # Otherwise, fall back to softened hard labels.
            hard_class = 1 if verdict == label else 0

            if "score_list" in item.index:
                teacher_score = float(item["score_list"][trace_idx])

                # Convert any raw score to a 0–1 probability-like value
                teacher_score = 1 / (1 + np.exp(-teacher_score))
            else:
                teacher_score = 0.9 if hard_class == 1 else 0.1

            final_training_data.append(
                {
                    "sample_id": sample_id,
                    "query_id": qid,
                    "input_text": input_text,
                    "Claim": claim,
                    "Evidence": evidence,
                    "Justification": justification,
                    "Label": label,
                    "Verdict": verdict,
                    "Class": hard_class,
                    "TeacherScore": teacher_score,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(final_training_data).to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print(f"Wrote {len(final_training_data)} examples to {output_path}")
    print(f"Unknown traces included: {unknown_state['counter']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build evidence-aware verifier training data from raw task JSON."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw train JSON.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for output JSONL file.",
    )

    args = parser.parse_args()

    build_training_data(args.input, args.output)