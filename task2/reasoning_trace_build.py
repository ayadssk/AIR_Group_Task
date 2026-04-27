"""
Convert the raw task data into training examples for the verifier / reward model.

Input:  output/reasoning_generation/<lang>_train.json
Output: output/training_data_for_RM/<lang>_train.jsonl

Each output record has fields:
  sample_id  : "<query_id>_<a-z suffix>"
  input_text : "Claim: ...\nVerdict: ...\nJustification: ..."
  Label      : ground-truth label (lower-case)
  Verdict    : verdict of this reasoning trace (lower-case)
  Class      : 1 if verdict == label else 0
"""

import json
import random
import re

import pandas as pd

UNKNOWN_LIMIT = 150
unknown_counter = 0


def remove_label_pattern(text: str) -> str:
    text = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting))",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text.replace("\n", " ")


def sample_training_examples(row: pd.Series) -> list[int]:
    global unknown_counter

    label = row["label"].lower()
    verdict_list = [v.lower() for v in row["Verdict_list"]]

    correct_indices = [i for i, v in enumerate(verdict_list) if v == label]
    true_indices = [i for i, v in enumerate(verdict_list) if v == "true" and v != label]
    false_indices = [i for i, v in enumerate(verdict_list) if v == "false" and v != label]
    conflicting_indices = [i for i, v in enumerate(verdict_list) if v == "conflicting" and v != label]
    unknown_indices = [i for i, v in enumerate(verdict_list) if v == "unknown"]

    selected_indices: list[int] = []

    if len(correct_indices) >= 2:
        selected_indices.extend(random.sample(correct_indices, 2))
        num_remaining = 4
    elif len(correct_indices) == 1:
        selected_indices.append(correct_indices[0])
        num_remaining = 5
    else:
        num_remaining = 6

    wrong_indices: list[int] = []
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
        selected_indices = random.sample(unknown_indices, min(5, len(unknown_indices)))
    elif unknown_indices and unknown_counter < UNKNOWN_LIMIT:
        if selected_indices:
            selected_indices.pop()
        selected_indices.append(random.choice(unknown_indices))
        unknown_counter += 1

    return selected_indices


def build_training_data(input_path: str, output_path: str) -> None:
    global unknown_counter
    unknown_counter = 0

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    data = pd.DataFrame(raw)

    data["sampled_indices"] = data.apply(sample_training_examples, axis=1)

    final_training_data = []
    for idx in range(len(data)):
        item = data.loc[idx]
        label = item["label"].lower()

        for decoding_idx, decoding_sample in enumerate(item["sampled_indices"]):
            justification = remove_label_pattern(item["Reasoning_traces"][decoding_sample])
            verdict = item["Verdict_list"][decoding_sample].lower()
            qid = item["query_id"] if "query_id" in item.index else idx
            sample_id = str(qid) + "_" + chr(97 + decoding_idx)

            if len(justification.split()) < 3:
                continue

            final_training_data.append({
                "sample_id": sample_id,
                "input_text": f"Claim: {item['claim']}\nVerdict: {verdict}\nJustification: {justification}",
                "Label": label,
                "Verdict": verdict,
                "Class": 1 if verdict == label else 0,
            })

    pd.DataFrame(final_training_data).to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print(f"Wrote {len(final_training_data)} examples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build verifier training data from raw task JSON.")
    parser.add_argument("--input", required=True, help="Path to raw <lang>_train.json")
    parser.add_argument("--output", required=True, help="Path for output .jsonl file")
    args = parser.parse_args()

    build_training_data(args.input, args.output)
