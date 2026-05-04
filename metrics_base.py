# =========================
# METRIC COLLECTION SKELETON
# =========================

import os
import re
import json
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
import torch


# =========================
# 1. Experiment metadata
# =========================

BASE = "/content/drive/MyDrive/ColabNotebooks/AIR_Group_Task"

MODEL_NAME = "roberta"              # e.g. roberta, distilbert, llama
MODEL_TYPE = "baseline"             # baseline / experiment with distillation
BASE_MODEL = "roberta-base"          # HuggingFace model name
STUDENT_MODEL = BASE_MODEL
TEACHER_MODEL = "none"               # none / llama-3b / etc.
DISTILLATION_STRATEGY = "none"       # none / response_mse / contrastive / grouped_contrastive / feature_based
PREPROCESSING = "with_evidence"      # provided_preprocessing / custom_evidence_preprocessing / etc.
AGGREGATION = "top1"                 # top1 / majority_top3 / majority_top5 / score_weighted

BATCH_SIZE = 4
INFERENCE_BATCH_SIZE = 1
EPOCHS = 3

RESULTS_ROOT = f"{BASE}/output/results"
PRED_ROOT = f"{BASE}/output/RM_prediction"
RUNS_ROOT = f"{BASE}/runs"
CKPT_ROOT = f"{BASE}/output/checkpoints"

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(PRED_ROOT, exist_ok=True)
os.makedirs(RUNS_ROOT, exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. Helper functions
# =========================

def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_hardware_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def get_parameter_counts(model):
    total_params = 0
    trainable_params = 0

    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return total_params, trainable_params


def get_file_size_mb(path):
    if path is None or not os.path.exists(path):
        return np.nan
    return os.path.getsize(path) / (1024 ** 2)


def parse_scorer_result(result_csv_path):
    """
    Parses macro F1 and Recall@5 from task2/scorer.py output.
    Adjust regex if your scorer output format changes.
    """

    with open(result_csv_path, "r", encoding="utf-8") as f:
        content = f.read()

    m_f1 = re.search(
        r"^macro avg,[0-9.]+,[0-9.]+,([0-9.]+)",
        content,
        re.MULTILINE,
    )

    m_r5 = re.search(
        r"^5,([0-9.]+)",
        content,
        re.MULTILINE,
    )

    macro_f1 = float(m_f1.group(1)) if m_f1 else np.nan
    recall_at5 = float(m_r5.group(1)) if m_r5 else np.nan

    return macro_f1, recall_at5


def run_scorer_and_copy_outputs(pred_path, result_path, ir_path):
    """
    scorer.py expects predictions at:
    output/RM_prediction/clef_predictions.json

    This function copies our named prediction file there,
    runs scorer.py, then copies result.csv and per_sample_ir.csv
    to clearly named final locations.
    """

    os.makedirs("output/RM_prediction", exist_ok=True)

    shutil.copy(pred_path, "output/RM_prediction/clef_predictions.json")

    subprocess.run(
        [sys.executable, "task2/scorer.py"],
        check=True,
    )

    shutil.copy("output/RM_prediction/result.csv", result_path)
    shutil.copy("output/RM_prediction/per_sample_ir.csv", ir_path)

    return parse_scorer_result(result_path)


def create_trec_file(predictions, trec_path, run_id):
    """
    Creates a TREC-style ranking file from prediction JSON format.
    Assumes each prediction has:
    - query_id
    - score_list
    """

    with open(trec_path, "w", encoding="utf-8") as out:
        for sample in predictions:
            query_id = sample["query_id"]

            ranked = sorted(
                enumerate(sample["score_list"]),
                key=lambda x: x[1],
                reverse=True,
            )

            for rank, (trace_idx, score) in enumerate(ranked, start=1):
                out.write(
                    f"{query_id}\tQ0\t{query_id}_{trace_idx}\t{rank}\t{score:.6f}\t{run_id}\n"
                )

    return os.path.exists(trec_path)


# =========================
# 3. Main metric-tracking template
# =========================

all_results = []

LANGUAGES = [
    {
        "lang": "english",
        "train_path": f"{BASE}/data/english/english_train.json",
        "val_path": f"{BASE}/data/english/clef2026_gpt4_o_mini_val.json",
    },
    {
        "lang": "spanish",
        "train_path": f"{BASE}/data/spanish/spanish_train.json",
        "val_path": f"{BASE}/data/spanish/spanish_val.json",
    },
    {
        "lang": "arabic",
        "train_path": f"{BASE}/data/arabic/clef2026_gpt4_o_mini_train_arabic.json",
        "val_path": f"{BASE}/data/arabic/clef2026_gpt4_o_mini_val_arabic.json",
    },
]


for lang_cfg in LANGUAGES:
    lang = lang_cfg["lang"]

    print("=" * 80)
    print(f"LANGUAGE: {lang}")
    print("=" * 80)

    # =========================
    # A. Load train / validation data
    # =========================

    train_path = lang_cfg["train_path"]
    val_path = lang_cfg["val_path"]

    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    n_val_claims = len(val_data)
    n_val_reasoning_traces = sum(len(x["Reasoning_traces"]) for x in val_data)
    avg_traces_per_claim = n_val_reasoning_traces / max(n_val_claims, 1)

    # =========================
    # B. Train model and measure training time
    # =========================

    checkpoint_dir = f"{CKPT_ROOT}/{lang}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = f"{checkpoint_dir}/{MODEL_NAME}_{lang}.pt"

    training_status = "trained"
    training_time_sec = np.nan

    sync_if_cuda()
    train_start = time.perf_counter()

    # -------------------------------------------------
    # TODO: replace this block with actual training code
    # -------------------------------------------------
    #
    # model = ...
    # train_loader = ...
    # optimizer = ...
    #
    # for epoch in range(EPOCHS):
    #     train_one_epoch(...)
    #
    # torch.save(model.state_dict(), checkpoint_path)
    #
    # -------------------------------------------------

    sync_if_cuda()
    train_end = time.perf_counter()

    training_time_sec = train_end - train_start

    # If checkpoint already exists and you skipped training:
    # training_status = "skipped_existing_checkpoint"
    # training_time_sec = np.nan

    # =========================
    # C. Load final model
    # =========================

    # -------------------------------------------------
    # TODO: replace this block with actual model loading
    # -------------------------------------------------
    #
    # model = YourModel(...)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # model.to(device)
    # model.eval()
    #
    # -------------------------------------------------

    # Example placeholders. Remove after adding real model.
    model = None

    if model is not None:
        nr_params, trainable_params = get_parameter_counts(model)
    else:
        nr_params, trainable_params = np.nan, np.nan

    model_size_mb = get_file_size_mb(checkpoint_path)

    # =========================
    # D. Run validation inference and measure latency
    # =========================

    scored_samples = []

    sync_if_cuda()
    inference_start = time.perf_counter()

    for idx, sample in enumerate(val_data):
        query_id = sample.get("query_id", idx)
        claim = sample["claim"]

        verdict_list = sample["Verdict_list"]
        reasoning_traces = sample["Reasoning_traces"]

        score_list = []
        cleaned_traces = []

        for trace_idx, trace in enumerate(reasoning_traces):
            verdict = verdict_list[trace_idx]

            # -------------------------------------------------
            # TODO: replace this with actual scoring code
            # -------------------------------------------------
            #
            # input_text = build_input(claim, evidence, verdict, trace)
            # score = model_score(model, tokenizer, input_text)
            #
            # -------------------------------------------------

            score = 0.0  # placeholder
            cleaned_trace = trace

            score_list.append(float(score))
            cleaned_traces.append(cleaned_trace)

        scored_samples.append({
            "query_id": query_id,
            "Claim": claim,
            "Label": sample["label"],
            "verdict_list": verdict_list,
            "score_list": score_list,
            "justification_list": cleaned_traces,
        })

    sync_if_cuda()
    inference_end = time.perf_counter()

    inference_time_sec = inference_end - inference_start

    val_avg_time_claim_ms = (inference_time_sec / max(n_val_claims, 1)) * 1000
    val_avg_time_reasoning_ms = (inference_time_sec / max(n_val_reasoning_traces, 1)) * 1000

    # =========================
    # E. Apply aggregation
    # =========================

    def aggregate_top1(verdict_list, score_list):
        best_idx = int(np.argmax(score_list))
        return verdict_list[best_idx]

    predictions = []

    for s in scored_samples:
        final_verdict = aggregate_top1(
            s["verdict_list"],
            s["score_list"],
        )

        predictions.append({
            "query_id": s["query_id"],
            "Claim": s["Claim"],
            "Label": s["Label"],
            "Verdict_BoN": final_verdict,
            "BoN_Verdict_list": s["verdict_list"],
            "Reasoning_traces": s["justification_list"],
            "score_list": s["score_list"],
        })

    # =========================
    # F. Save files
    # =========================

    run_id = (
        f"{MODEL_NAME}_{MODEL_TYPE}_{DISTILLATION_STRATEGY}_"
        f"{PREPROCESSING}_{lang}_{AGGREGATION}"
    )

    result_dir = f"{RESULTS_ROOT}/{lang}/{run_id}"
    pred_dir = f"{PRED_ROOT}/{lang}"
    runs_dir = f"{RUNS_ROOT}/{lang}"

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    pred_path = f"{pred_dir}/val_predictions_{run_id}.json"
    result_path = f"{result_dir}/val_result_{run_id}.csv"
    ir_path = f"{result_dir}/val_per_sample_ir_{run_id}.csv"
    trec_path = f"{runs_dir}/val_trec_{run_id}.txt"

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    # =========================
    # G. Run scorer
    # =========================

    macro_f1, recall_at5 = run_scorer_and_copy_outputs(
        pred_path=pred_path,
        result_path=result_path,
        ir_path=ir_path,
    )

    # =========================
    # H. Create TREC file
    # =========================

    trec_created = create_trec_file(
        predictions=predictions,
        trec_path=trec_path,
        run_id=run_id,
    )

    # =========================
    # I. Save one checklist row
    # =========================

    all_results.append({
        "Model": MODEL_NAME,
        "Type (baseline/experiment)": MODEL_TYPE,

        "English": "yes" if lang == "english" else "",
        "Spanish": "yes" if lang == "spanish" else "",
        "Arabic": "yes" if lang == "arabic" else "",

        "Language": lang,
        "Base model": BASE_MODEL,
        "Student model": STUDENT_MODEL,
        "Teacher model": TEACHER_MODEL,
        "Distillation strategy": DISTILLATION_STRATEGY,
        "Preprocessing": PREPROCESSING,
        "Aggregation": AGGREGATION,
        "Run ID": run_id,

        "Val results.csv": result_path,
        "Val per_sample_ir.csv": ir_path,
        "Val prediction JSON": pred_path,

        "Nr. Of parameters": nr_params,
        "Trainable parameters": trainable_params,
        "Model size (MB)": round(model_size_mb, 2) if not np.isnan(model_size_mb) else "",

        "Training status": training_status,
        "Training time": round(training_time_sec, 2) if not np.isnan(training_time_sec) else "",

        "Val Avg time/claim": round(val_avg_time_claim_ms, 2),
        "Val Avg. time/reasoning": round(val_avg_time_reasoning_ms, 2),
        "Val Avg traces/claim": round(avg_traces_per_claim, 2),

        "Val Macro F1": macro_f1,
        "Val Recall@5": recall_at5,

        "TREC file created": "yes" if trec_created else "no",
        "TREC file path": trec_path if trec_created else "",

        "Run on hardware (CPU/GPU)": "GPU" if torch.cuda.is_available() else "CPU",
        "Hardware name": get_hardware_name(),
        "Batch Size": BATCH_SIZE,
        "Inference batch size": INFERENCE_BATCH_SIZE,

        "Checkpoint path": checkpoint_path,
        "n_val_claims": n_val_claims,
        "n_val_reasoning_traces": n_val_reasoning_traces,
    })


# =========================
# 4. Export summary
# =========================

results_df = pd.DataFrame(all_results)

summary_path = f"{BASE}/output/experiment_checklist_summary.csv"
results_df.to_csv(summary_path, index=False)

print("=" * 120)
print("EXPERIMENT CHECKLIST SUMMARY")
print("=" * 120)

display_cols = [
    "Model",
    "Type (baseline/experiment)",
    "Language",
    "Preprocessing",
    "Aggregation",
    "Val Macro F1",
    "Val Recall@5",
    "Nr. Of parameters",
    "Model size (MB)",
    "Training time",
    "Val Avg time/claim",
    "Val Avg. time/reasoning",
    "Val Avg traces/claim",
    "TREC file created",
    "Run on hardware (CPU/GPU)",
    "Batch Size",
]

print(results_df[display_cols].to_string(index=False))

print("=" * 120)
print(f"Saved summary to: {summary_path}")
print("=" * 120)