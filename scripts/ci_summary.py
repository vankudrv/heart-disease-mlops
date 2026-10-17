"""Print the model comparison as a Markdown table (used for the GitHub Actions job summary)."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "model_comparison.csv"
METADATA = ROOT / "models" / "model_metadata.json"
COLUMNS = ["model", "cv_roc_auc", "cv_recall", "test_accuracy", "test_recall", "test_roc_auc"]


def render() -> str:
    if not RESULTS.exists():
        return "No training results found."
    with RESULTS.open() as handle:
        rows = list(csv.DictReader(handle))
    lines = ["## Model comparison", "",
             "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        lines.append("| " + " | ".join(row[c] for c in COLUMNS) + " |")
    if METADATA.exists():
        meta = json.loads(METADATA.read_text())
        lines += ["", f"**Selected model:** `{meta['model']}` "
                      f"(rule: {meta['selection_rule']}, scikit-learn {meta['sklearn_version']})"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
