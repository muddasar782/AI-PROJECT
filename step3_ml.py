"""
AIDRA - Step 3: Machine Learning — Survival Probability Estimation
===================================================================
We train TWO models to predict whether a victim will survive given:
  - severity score     (1=minor, 2=moderate, 3=critical)
  - distance to base   (how far the rescue team has to travel)
  - risk exposure      (how many hazard cells on the route)
  - rescue time        (estimated minutes to reach victim)

Models:
  1. k-Nearest Neighbours (kNN)
  2. Naive Bayes (GaussianNB)

Output: survival probability for each real victim → used by agent
        to prioritise and justify routing decisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from step1_environment import DisasterEnvironment, astar

# ─────────────────────────────────────────
# 1. GENERATE SYNTHETIC TRAINING DATA
#    (simulates many disaster scenarios)
# ─────────────────────────────────────────

np.random.seed(42)

def generate_dataset(n=300):
    """
    Simulate 300 past disaster rescue records.
    Features: [severity, distance, risk_exposure, rescue_time]
    Label   : 1 = survived, 0 = did not survive
    """
    X, y = [], []
    for _ in range(n):
        severity      = np.random.choice([1, 2, 3])        # 1=minor,2=mod,3=crit
        distance      = np.random.randint(1, 20)
        risk_exposure = np.random.randint(0, 8)
        rescue_time   = distance * 2 + risk_exposure       # rough estimate

        # Survival logic (domain-informed rules + noise)
        base_prob = 0.9
        base_prob -= (severity - 1) * 0.20    # critical → lower base
        base_prob -= distance      * 0.02     # farther → worse
        base_prob -= risk_exposure * 0.04     # more hazard → worse
        base_prob += np.random.normal(0, 0.08) # real-world noise
        base_prob = np.clip(base_prob, 0.05, 0.95)

        survived = 1 if np.random.random() < base_prob else 0
        X.append([severity, distance, risk_exposure, rescue_time])
        y.append(survived)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────
# 2. TRAIN & EVALUATE BOTH MODELS
# ─────────────────────────────────────────

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scale features (important for kNN)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    models = {
        "kNN (k=5)":    KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":  GaussianNB(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model":     model,
            "scaler":    scaler,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "cm":        confusion_matrix(y_test, y_pred),
            "y_test":    y_test,
            "y_pred":    y_pred,
        }

    return results


# ─────────────────────────────────────────
# 3. PREDICT SURVIVAL FOR REAL VICTIMS
# ─────────────────────────────────────────

SEVERITY_NUM = {"critical": 3, "moderate": 2, "minor": 1}

def predict_victim_survival(victims, env, results):
    """
    Use the best model to estimate each real victim's survival probability.
    Features extracted from the environment (route cost, hazard cells).
    """
    # Pick best model by F1
    best_name = max(results, key=lambda k: results[k]["f1"])
    best      = results[best_name]
    model     = best["model"]
    scaler    = best["scaler"]

    print(f"\n Using best model: {best_name} (F1={best['f1']:.3f})")
    print(f"\n{'Victim':<8}{'Severity':<12}{'Dist':<8}{'Risk':<8}"
          f"{'Time':<8}{'Survival%':<12}{'Prediction'}")
    print("-" * 65)

    predictions = []
    for v in victims:
        sev  = SEVERITY_NUM[v["severity"]]
        route = astar(env, env.base_pos, v["pos"], avoid_hazard=False)
        dist  = route["cost"] if route else 15

        # Count hazard cells on route
        hazard_count = sum(
            1 for pos in (route["path"] if route else [])
            if pos in env.hazard_cells
        )
        rescue_time = dist * 2 + hazard_count

        feat      = scaler.transform([[sev, dist, hazard_count, rescue_time]])
        prob      = model.predict_proba(feat)[0][1]   # P(survived=1)
        predicted = " Survive" if prob >= 0.5 else " At risk"

        print(f"{v['name']:<8}{v['severity']:<12}{dist:<8}{hazard_count:<8}"
              f"{rescue_time:<8}{prob*100:<12.1f}{predicted}")

        predictions.append({**v, "survival_prob": prob,
                             "dist": dist, "hazard": hazard_count,
                             "rescue_time": rescue_time})

    return predictions, best_name


# ─────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────

def visualise_ml(results, predictions):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    metric_names = ["accuracy", "precision", "recall", "f1"]
    model_names  = list(results.keys())
    colors       = ["#457b9d", "#e76f51"]

    # ── (0,0)+(0,1): metrics bar chart ──
    ax0 = fig.add_subplot(gs[0, :2])
    x   = np.arange(len(metric_names))
    w   = 0.35
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results[name][m] for m in metric_names]
        bars = ax0.bar(x + i*w, vals, w, label=name, color=color,
                       edgecolor='white', alpha=0.9)
        for bar, val in zip(bars, vals):
            ax0.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01, f"{val:.2f}",
                     ha='center', fontsize=8)
    ax0.set_xticks(x + w/2)
    ax0.set_xticklabels([m.capitalize() for m in metric_names])
    ax0.set_ylim(0, 1.15)
    ax0.set_title("Model Comparison — Classification Metrics", fontweight='bold')
    ax0.legend()
    ax0.set_ylabel("Score")

    # ── (0,2): confusion matrices side by side ──
    for i, (name, color) in enumerate(zip(model_names, colors)):
        ax = fig.add_subplot(gs[0, 2] if i == 0 else gs[1, 2])
        cm = results[name]["cm"]
        disp = ConfusionMatrixDisplay(cm, display_labels=["Did not survive","Survived"])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f"Confusion Matrix\n{name}", fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

    # ── (1,0)+(1,1): victim survival probability ──
    ax1 = fig.add_subplot(gs[1, :2])
    names = [p["name"] for p in predictions]
    probs = [p["survival_prob"] * 100 for p in predictions]
    bar_colors = ["#e63946" if p < 50 else "#2a9d8f" for p in probs]
    bars = ax1.bar(names, probs, color=bar_colors, edgecolor='white', width=0.5)
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% threshold')
    for bar, prob, p in zip(bars, probs, predictions):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.5, f"{prob:.1f}%",
                 ha='center', fontsize=9, fontweight='bold')
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height()/2, p["severity"][:3].upper(),
                 ha='center', va='center', fontsize=8, color='white')
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Survival Probability (%)")
    ax1.set_title("ML-Predicted Survival Probability per Victim", fontweight='bold')
    ax1.legend(fontsize=9)

    fig.suptitle("AIDRA — Step 3: Machine Learning Risk Estimation",
                 fontsize=13, fontweight='bold', y=1.01)
    return fig


# ─────────────────────────────────────────
# 5. RUN DEMO
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AIDRA — Step 3: ML Survival Prediction")
    print("=" * 60)

    # Generate dataset & train
    X, y = generate_dataset(n=300)
    print(f"\n Dataset: {len(X)} samples  |  "
          f"Survived: {y.sum()}  |  Did not: {(1-y).sum()}")

    results = train_and_evaluate(X, y)

    print(f"\n{'Model':<16} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8}")
    print("-" * 56)
    for name, r in results.items():
        print(f"{name:<16} {r['accuracy']:>10.3f} {r['precision']:>10.3f} "
              f"{r['recall']:>8.3f} {r['f1']:>8.3f}")

    # Predict for real victims
    env = DisasterEnvironment()
    predictions, best_model = predict_victim_survival(
        env.victims, env, results
    )

    # Save chart
    fig = visualise_ml(results, predictions)
    fig.savefig("/mnt/user-data/outputs/aidra_ml.png",
                dpi=150, bbox_inches='tight')
    print("\n ML chart saved to aidra_ml.png")
    print("Step 3 complete! Next: Fuzzy Logic (uncertainty handling)")
