"""Train, tune, evaluate and track three classifiers with MLflow.

Run from the project root:
    python -m src.train
    python -m src.train --no-fe --no-save     # ablation run without the engineered feature

For every model one MLflow run is created containing the parameters, the
cross-validated and hold-out metrics, plots (confusion matrix, ROC curve,
feature importance), a text classification report and the fitted pipeline.
Model selection rule: highest cross-validated ROC-AUC wins; models within
TIE_TOLERANCE of the best count as tied, and the tie is broken by cross-validated
recall (in a medical setting a missed patient is worse than a false alarm).
The chosen pipeline is saved to models/best_model.joblib for the API.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402

from src.data_processing import PROJECT_ROOT, load_clean  # noqa: E402
from src.preprocessing import RAW_FEATURES, build_pipeline, feature_names, prepare_xy  # noqa: E402

SEED = 42
TIE_TOLERANCE = 0.01   # ROC-AUC difference treated as "tied"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORT_DIR = PROJECT_ROOT / "reports"

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}

# name -> (estimator, hyper-parameter grid). "model__" targets the classifier step.
MODEL_SPACE = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"model__n_estimators": [100, 300], "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 3]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3]},
    ),
}


def holdout_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc(y_true, y_proba, title):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_importance(pipeline, title, top=15):
    """Coefficients for linear models, impurity importance for tree models."""
    model = pipeline.named_steps["model"]
    names = feature_names(pipeline)
    if hasattr(model, "coef_"):
        values, label = model.coef_.ravel(), "coefficient (standardised scale)"
    else:
        values, label = model.feature_importances_, "importance"
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([names[i] for i in order], values[order], color="#e76f51")
    ax.set_xlabel(label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--experiment-name", default="heart-disease-classification")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--no-fe", action="store_true", help="disable the engineered feature")
    p.add_argument("--no-save", action="store_true", help="do not overwrite models/best_model.joblib")
    return p.parse_args()


def main():
    args = parse_args()
    use_fe = not args.no_fe

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())   # local ./mlruns folder
    mlflow.set_experiment(args.experiment_name)

    df = load_clean()
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)
    print(f"Train rows: {len(X_train)} | test rows: {len(X_test)} | engineered feature: {use_fe}")

    results = []
    for name, (estimator, grid) in MODEL_SPACE.items():
        print(f"\n=== {name} ===")
        pipeline = build_pipeline(estimator, use_engineered=use_fe)
        search = GridSearchCV(pipeline, grid, scoring=SCORING, refit="roc_auc",
                              cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        i = search.best_index_
        cv_scores = {m: search.cv_results_[f"mean_test_{m}"][i] for m in SCORING}
        cv_roc_std = search.cv_results_["std_test_roc_auc"][i]

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        test_scores = holdout_metrics(y_test, y_pred, y_proba)

        run_name = name + ("" if use_fe else "_no_fe")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({"model_family": name, "engineered_feature": str(use_fe)})
            mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_params({
                "cv_folds": args.cv_folds, "test_size": args.test_size, "seed": SEED,
                "grid_size": len(search.cv_results_["params"]), "engineered_feature": use_fe,
                "n_train": len(X_train), "n_test": len(X_test), "n_raw_features": len(RAW_FEATURES),
            })
            mlflow.log_metrics({f"cv_{m}": v for m, v in cv_scores.items()})
            mlflow.log_metric("cv_roc_auc_std", cv_roc_std)
            mlflow.log_metrics({f"test_{m}": v for m, v in test_scores.items()})

            for fig, path in [
                (plot_confusion(y_test, y_pred, f"{name}: confusion matrix"), "plots/confusion_matrix.png"),
                (plot_roc(y_test, y_proba, f"{name}: ROC curve"), "plots/roc_curve.png"),
                (plot_importance(best, f"{name}: top features"), "plots/feature_importance.png"),
            ]:
                mlflow.log_figure(fig, path)
                plt.close(fig)
            mlflow.log_text(
                classification_report(y_test, y_pred, target_names=["No disease", "Disease"]),
                "classification_report.txt")
            mlflow.sklearn.log_model(
                sk_model=best, artifact_path="model",
                signature=infer_signature(X_train, best.predict(X_train)),
                input_example=X_train.head(3))
            run_id = run.info.run_id

        print(f"CV  ROC-AUC {cv_scores['roc_auc']:.3f} (+/- {cv_roc_std:.3f}) | "
              f"test ROC-AUC {test_scores['roc_auc']:.3f} | test recall {test_scores['recall']:.3f}")
        print("Best params:", search.best_params_)
        results.append({"model": name, "run_id": run_id, "pipeline": best, "params": search.best_params_,
                        **{f"cv_{m}": v for m, v in cv_scores.items()},
                        **{f"test_{m}": v for m, v in test_scores.items()}})

    # ---- compare models and pick the best by cross-validated ROC-AUC -------------------
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pipeline", "params")}
                          for r in results]).sort_values("cv_roc_auc", ascending=False)
    REPORT_DIR.mkdir(exist_ok=True)
    table.drop(columns="run_id").round(4).to_csv(
        REPORT_DIR / ("model_comparison.csv" if use_fe else "model_comparison_no_fe.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = table.set_index("model")[["cv_roc_auc", "test_roc_auc", "test_recall", "test_accuracy"]]
    plot_df.plot.bar(ax=ax, rot=0, colormap="viridis")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Model comparison")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    (REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_DIR / "figures" / ("07_model_comparison.png" if use_fe else "07_model_comparison_no_fe.png"),
                dpi=150)

    with mlflow.start_run(run_name="model_comparison" + ("" if use_fe else "_no_fe")):
        mlflow.log_figure(fig, "model_comparison.png")
        mlflow.log_text(table.drop(columns="run_id").round(4).to_csv(index=False), "model_comparison.csv")
    plt.close(fig)

    top_auc = max(r["cv_roc_auc"] for r in results)
    contenders = [r for r in results if top_auc - r["cv_roc_auc"] <= TIE_TOLERANCE]
    winner = max(contenders, key=lambda r: r["cv_recall"])
    print(f"\nContenders within {TIE_TOLERANCE} ROC-AUC of the best: {[r['model'] for r in contenders]}")
    print(f"Selected model (highest CV recall among contenders): {winner['model']}")
    print(table.drop(columns="run_id").round(3).to_string(index=False))

    if args.no_save:
        print("--no-save given: models/best_model.joblib left untouched.")
        return

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(winner["pipeline"], MODEL_PATH)
    metadata = {
        "model": winner["model"], "selection_rule": f"CV ROC-AUC within {TIE_TOLERANCE}, then CV recall",
        "best_params": {k: str(v) for k, v in winner["params"].items()},
        "cv_roc_auc": round(winner["cv_roc_auc"], 4), "test_roc_auc": round(winner["test_roc_auc"], 4),
        "test_recall": round(winner["test_recall"], 4), "test_accuracy": round(winner["test_accuracy"], 4),
        "input_features": RAW_FEATURES, "engineered_feature": use_fe,
        "sklearn_version": sklearn.__version__, "mlflow_run_id": winner["run_id"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    with mlflow.start_run(run_id=winner["run_id"]):
        mlflow.set_tag("selected_for_deployment", "true")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="deployment")
        mlflow.log_artifact(str(METADATA_PATH), artifact_path="deployment")
    print(f"Saved {MODEL_PATH} and {METADATA_PATH}")


if __name__ == "__main__":
    main()
