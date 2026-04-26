# Fact-Checking Numerical Claims

The task spans three languages — English, Spanish, and Arabic.

Given a claim and a set of reasoning paths with evidence, your task is to **rank the
reasoning paths** using a verifier model. You can also verify which of your top-ranked
paths has the right verdict.

During evaluation you will be asked to output the top-5 reasoning paths as ranked by
your verifier; these are evaluated with **Recall@5**. Additionally, the top-1 verdict is
used to calculate **Macro F1**.

## Data

| Split | Location |
|-------|----------|
| English train | [Google Drive](https://drive.google.com/file/d/11g-LyDVrMP09EimzKQIg0hXBvEXrKCQN/view?usp=sharing) → place at `data/english/english_train.json` |
| English val | `data/english/clef2026_gpt4_o_mini_val.json` |
| Spanish train | `data/spanish/spanish_train.json` |
| Spanish val | `data/spanish/spanish_val.json` |
| Arabic train | `data/arabic/clef2026_gpt4_o_mini_train_arabic.json` |
| Arabic val | `data/arabic/clef2026_gpt4_o_mini_val_arabic.json` |

## Files in this folder

| File | Description |
|------|-------------|
| `CT26_Task2_baseline.ipynb` | Baseline notebook: LoRA fine-tuning of a LLaMA-3B verifier |
| `reasoning_trace_build.py` | Preprocessing: converts raw JSON data into training examples for the verifier |
| `scorer.py` | Evaluation: computes Recall@k, macro F1, and per-sample metrics |

## Running the scorer

After generating predictions to `output/RM_prediction/clef_predictions.json`:

```bash
python task2/scorer.py
```

Output files:
- `output/RM_prediction/result.csv` — aggregate Recall@k and classification metrics
- `output/RM_prediction/per_sample_ir.csv` — per-sample Recall@k and Precision@k
