import os
import joblib
import numpy as np


def test_model_load_and_predict():
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root, "model", "entrepreneurial_skill_model.joblib")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    model = joblib.load(model_path)

    # Create a single sample with the expected number of features (6 features used in the app)
    sample = np.array([[16, 1, 3.5, 4.0, 75, 1]])

    # model should have a predict method
    assert hasattr(model, "predict"), "Loaded model has no predict method"

    pred = model.predict(sample)
    assert hasattr(pred, "__len__"), "Prediction is not an array-like"
    assert len(pred) == 1, "Prediction should return a single value for one sample"

    # If model supports predict_proba, ensure it returns shape (1, n_classes)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)
        assert probs.shape[0] == 1, "predict_proba should return one row for one sample"
