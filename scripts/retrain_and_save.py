"""Retrain model pipeline (scaler + RandomForest) and save artifacts.

This reproduces the notebook's data generation, trains a RandomForest, saves a Pipeline
to model/entrepreneurial_skill_model_v2.joblib, writes metadata.json, and exports a
feature importance plot to docs/figures/feature_importance.png.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'model'
FIG_DIR = ROOT / 'docs' / 'figures'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def generate_data(n_samples=200, random_state=42):
    np.random.seed(random_state)
    data = pd.DataFrame({
        'age': np.random.randint(15, 20, n_samples),
        'prior_business_exposure': np.random.binomial(1, 0.3, n_samples),
        'risk_taking': np.random.rand(n_samples),
        'decision_speed': np.random.rand(n_samples),
        'creativity_score': np.random.rand(n_samples),
        'leadership_experience': np.random.binomial(1, 0.4, n_samples),
    })
    prob = (
        0.2 * data['prior_business_exposure'] +
        0.2 * data['risk_taking'] +
        0.2 * data['decision_speed'] +
        0.2 * data['creativity_score'] +
        0.2 * data['leadership_experience']
    )
    data['entrepreneurial_skill'] = (prob + 0.1 * np.random.randn(n_samples) > 0.7).astype(int)
    return data


def train_and_save():
    data = generate_data(n_samples=200)
    X = data.drop('entrepreneurial_skill', axis=1)
    y = data['entrepreneurial_skill']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
    }

    # Save model
    model_path = MODEL_DIR / 'entrepreneurial_skill_model_v2.joblib'
    joblib.dump(pipeline, model_path)

    # Save metadata
    metadata = {
        'model_path': str(model_path.relative_to(ROOT)),
        'sklearn_version': __import__('sklearn').__version__,
        'features': list(X.columns),
        'metrics': metrics,
    }
    with open(MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Feature importances
    rf = pipeline.named_steps['rf']
    importances = rf.feature_importances_
    feat = list(X.columns)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feat, importances)
    ax.set_xlabel('Importance')
    ax.set_title('Feature importances')
    plt.tight_layout()
    fig_path = FIG_DIR / 'feature_importance.png'
    fig.savefig(fig_path)

    print('Saved model to', model_path)
    print('Saved metadata to', MODEL_DIR / 'metadata.json')
    print('Saved feature importance to', fig_path)
    print('Metrics:', metrics)


if __name__ == '__main__':
    train_and_save()
