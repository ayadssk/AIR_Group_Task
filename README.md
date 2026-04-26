# CLEF CheckThat! 2026 — Task 2: Fact-Checking Numerical Claims
## Course Project Template

This repository is your team's workspace for the course project. You will participate in
[CLEF CheckThat! Task 2](https://checkthat.gitlab.io/) and develop a reasoning trace
verification system under your assigned methodological track.

**Core problem:** Given a claim containing quantitative and/or temporal expressions,
relevant evidence, and a set of LLM-generated reasoning traces with corresponding verdicts,
train a verifier model that (1) ranks the reasoning traces by their utility in leading to
the correct verdict, and (2) outputs a final verdict derived from the top-ranked traces.

The task spans three languages: **English**, **Spanish**, and **Arabic**.

---

## Getting Started

### 1. Register your team (do this first)

Open [`team.toml`](team.toml) and fill in your details, then verify with:

```bash
python validate_team.py
```

Fields to fill in:
- `team_name` — must exactly match the name under which you registered for CLEF CheckThat! Task 2
- `track` — your assigned track number (1–4)
- `members` — name, immatriculation number, and GitHub username for each of the 5 members

Commit `team.toml` to your repository. Failure to do so means your project cannot be matched
to TUWEL and **cannot be graded**.

### 2. Register for CLEF CheckThat!

Register your team for CLEF CheckThat! Task 2 and note the team name — it must match what
you put in `team.toml`.

### 3. Get the Data

Training and validation data are in the [`data/`](data/) folder:

| Path | Language | Split |
|------|----------|-------|
| `data/english/clef2026_gpt4_o_mini_val.json` | English | val |
| `data/spanish/spanish_train.json` | Spanish | train |
| `data/spanish/spanish_val.json` | Spanish | val |
| `data/arabic/clef2026_gpt4_o_mini_train_arabic.json` | Arabic | train |
| `data/arabic/clef2026_gpt4_o_mini_val_arabic.json` | Arabic | val |

The **English training split** is too large for git and must be downloaded separately:
[English train data (Google Drive)](https://drive.google.com/file/d/11g-LyDVrMP09EimzKQIg0hXBvEXrKCQN/view?usp=sharing)
→ place at `data/english/english_train.json`

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Project Tracks

Each team works on exactly one track. The track is assigned by the course instructors.

### Track 1 — Small & Distilled Verifier Models (< 0.5B parameters)
**Focus:** Efficient and lightweight verifier models under strict size constraints.
- Verifier models must be below ~0.5B parameters
- Model distillation from a larger teacher model is required
- Evaluate inference latency, model size, and performance/efficiency trade-offs (Recall@5, macro F1)
- Compare distillation strategies: response-based, feature-based, contrastive

### Track 2 — Verifier Training Objectives & Model Scale
**Focus:** Systematic comparison of learning-to-rank paradigms across model scales.
- Allowed: pointwise, pairwise, and listwise training objectives; models of varying sizes
- Goal: identify which training objective leads to the best reasoning trace ranker, and
  whether this holds consistently across model sizes (~100M, ~500M, ~1B+ parameters)
- Evaluate interaction effects between objective type and model scale

### Track 3 — Numerical & Temporal Reasoning Signals
**Focus:** Exploiting quantitative and temporal structure of claims to improve trace ranking.
- Directions: structured extraction of numbers/dates, symbolic consistency checking,
  numeric normalization, scoring traces by numerical faithfulness, auxiliary tasks
- Constraint: main contribution must come from reasoning signals, not general model scaling

### Track 4 — Trace Aggregation & Verdict Derivation
**Focus:** How a final verdict is derived from a set of ranked reasoning traces.
- Directions: majority voting over top-k, confidence-weighted aggregation, meta-reasoning,
  uncertainty estimation
- Constraint: keep the verifier/ranker fixed; contribution must be in the aggregation step

---

## Deliverables & Grading

| Component | Weight | Description |
|-----------|--------|-------------|
| Competition Registration | required | Must register — unregistered teams cannot be graded |
| Competitive Baseline | 20% | Fine-tuned verifier + verdict aggregation, evaluated on dev and test with macro F1 and Recall@5 |
| Track-Specific Research | 50% | Multiple meaningful experiments, ablations, literature engagement, error analysis |
| Report | 20% | 2-page ACM-style (double column): system description, experimental setup, results, discussion |
| Run Submission | 10% | TREC-formatted run files submitted to the shared task platform, mirrored in `runs/` |
| Code Submission | required | This repository, including all scripts and reproduction instructions |

### Report Requirements
The 2-page report must include: system description, experimental setup, results, discussion.

**Appendix** (not counted toward page limit): team name, individual contributions, optional
extra tables/runtimes/ablations.

### Code Requirements
Your repository must include:
- Instructions to reproduce all submitted runs
- Environment setup (`requirements.txt` or equivalent)
- Scripts to generate submitted runs
- Model checkpoints or instructions to obtain them
- The code must run with reasonable effort — non-runnable code may lose credit

---

## Repository Structure

Organize your repository as follows (adapt as needed for your track):

```
├── team.toml                # Fill in your team details — commit this
├── validate_team.py         # Run to check team.toml is correct
├── README.md                # This file — replace with a description of your system
├── requirements.txt         # Python dependencies
├── data/                    # All datasets (English val, Spanish, Arabic; see data/README.md)
├── task2/                   # Baseline notebook, scorer, and preprocessing (do not modify)
├── baseline/                # Your baseline pipeline (verifier + aggregation)
├── experiments/             # Track-specific experiments
├── runs/                    # Mirror of the TREC run files you submitted to the shared task
└── report/                  # PDF of your 2-page ACM report
```

> Note: Do not commit raw datasets or model weights. Large files will be rejected.

---

## Baseline

A reference baseline is provided in [`task2/CT26_Task2_baseline.ipynb`](task2/CT26_Task2_baseline.ipynb).
It fine-tunes a LLaMA-3B model with LoRA as a pointwise verifier (binary classifier) that
scores each reasoning trace. The top-scoring trace's verdict is used as the final prediction.

Use it as your starting point. Your baseline must be competitive — check the CLEF 2025
proceedings for reference systems and scores.

Preprocessing (converting raw data to training examples) is in
[`task2/reasoning_trace_build.py`](task2/reasoning_trace_build.py).

---

## Evaluation

Use the provided scorer to evaluate on development data:

```bash
python task2/scorer.py
```

The scorer reads from `output/RM_prediction/clef_predictions.json` and outputs:
- `output/RM_prediction/result.csv` — aggregate metrics (Recall@k, macro F1)
- `output/RM_prediction/per_sample_ir.csv` — per-sample IR metrics

## Run Submission

Submit your TREC-formatted run files to the shared task platform as instructed by the CLEF
organisers. Then copy the exact same files into the `runs/` folder of this repository and
commit them.

---

## Questions?

Post questions in the course forum on TUWEL or contact the course instructors.
