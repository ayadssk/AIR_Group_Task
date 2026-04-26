# Data

LLM-generated reasoning traces and ground-truth labels for CLEF CheckThat! 2026 Task 2.

| File | Language | Split | Size |
|------|----------|-------|------|
| `english/clef2026_gpt4_o_mini_val.json` | English | val | ~33 MB |
| `spanish/spanish_train.json` | Spanish | train | ~39 MB |
| `spanish/spanish_val.json` | Spanish | val | ~10 MB |
| `arabic/clef2026_gpt4_o_mini_train_arabic.json` | Arabic | train | ~68 MB |
| `arabic/clef2026_gpt4_o_mini_val_arabic.json` | Arabic | val | ~17 MB |

**English train split** is too large for git and must be downloaded separately:
[Google Drive link](https://drive.google.com/file/d/11g-LyDVrMP09EimzKQIg0hXBvEXrKCQN/view?usp=sharing)

Place it at `data/english/english_train.json` after downloading.

## Schema

Each JSON file is a list of objects with the following fields:

| Field | Description |
|-------|-------------|
| `query_id` | Unique identifier for the claim |
| `claim` | The numerical/temporal claim to verify |
| `label` | Ground-truth verdict: `true`, `false`, or `conflicting` |
| `Verdict_list` | List of verdicts from each reasoning trace |
| `Reasoning_traces` | List of LLM-generated reasoning trace strings |
| `Questions` | (optional) Decomposition questions used during generation |
