import pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path

MODELS_DIR = Path("/home/niranjan/ai_ids_project/models/saved")
DATA_DIR   = Path("/home/niranjan/ai_ids_project/data/processed")


class ExplainabilityEngine:
    def __init__(self):
        print("Loading RF model and feature metadata...")

        with open(MODELS_DIR / "rf_model.pkl", "rb") as f:
            self.rf = pickle.load(f)

        self.label_classes = np.load(DATA_DIR / "label_classes.npy", allow_pickle=True)

        # Load only the header row to get feature names cheaply
        cols = pd.read_csv(DATA_DIR / "processed_data.csv", nrows=0).columns.tolist()
        self.feature_names = [c for c in cols if c != "Label"]

        print(f"  [✓] RF loaded  ({len(self.feature_names)} features, {len(self.label_classes)} classes)")

        print("  Building SHAP TreeExplainer (this may take a moment)...")
        self.explainer = shap.TreeExplainer(self.rf)
        print("  [✓] SHAP explainer ready")

    def explain(self, feature_vector: np.ndarray) -> dict:
        """
        Parameters
        ----------
        feature_vector : np.ndarray of shape (77,)

        Returns
        -------
        dict:
            top_features  – list of {feature, value, importance}
            plain_text    – human-readable explanation string
        """
        x = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # Predicted class
        pred_class = int(self.rf.predict(x)[0])
        pred_label = str(self.label_classes[pred_class])

        # SHAP values: shape (n_classes, 1, n_features)
        shap_values = self.explainer.shap_values(x)

        # Per-feature importance for the predicted class
        class_shap = np.array(shap_values[pred_class])[0]  # (n_features,)
        abs_importance = np.abs(class_shap)

        top5_idx = np.argsort(abs_importance)[::-1][:5]

        top_features = []
        for idx in top5_idx:
            top_features.append({
                "feature":    self.feature_names[idx],
                "value":      round(float(x[0, idx]), 4),
                "importance": round(float(class_shap[idx]), 4),
            })

        # Plain-text explanation
        parts = []
        for f in top_features[:3]:
            pct = abs(f["importance"]) * 100
            direction = "above normal" if f["importance"] > 0 else "unusually low"
            parts.append(f"{f['feature']} was {pct:.0f}% {direction}")

        action = "benign traffic detected" if pred_label == "Benign" else f"threat detected: {pred_label}"
        plain_text = f"Alert triggered because {', and '.join(parts)}. Classification: {action}."

        return {
            "top_features": top_features,
            "plain_text":   plain_text,
        }


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = ExplainabilityEngine()

    rng = np.random.default_rng(7)
    test_vec = rng.random(77).astype(np.float32)

    print("\nExplaining random feature vector...")
    result = engine.explain(test_vec)

    print("\nTop 5 important features:")
    for i, f in enumerate(result["top_features"], 1):
        print(f"  {i}. {f['feature']:<40} value={f['value']:.4f}  shap={f['importance']:+.4f}")

    print(f"\nPlain-text explanation:")
    print(f"  {result['plain_text']}")
