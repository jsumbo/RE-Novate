import streamlit as st
import numpy as np
import joblib
import os
from pathlib import Path
import time
import pandas as pd
import sklearn

# Resolve model path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_V2 = BASE_DIR / 'model' / 'entrepreneurial_skill_model_v2.joblib'
MODEL_V1 = BASE_DIR / 'model' / 'entrepreneurial_skill_model.joblib'
MODEL_PATH = MODEL_V2 if MODEL_V2.exists() else MODEL_V1

if not MODEL_PATH.exists():
    st.title("Entrepreneurial Skill Simulator")
    st.error(f"Model file not found: {MODEL_PATH}\nMake sure a joblib model is present in the model/ folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Try to load metadata if present
metadata_path = BASE_DIR / 'model' / 'metadata.json'
model_meta = None
if metadata_path.exists():
    try:
        import json

        with open(metadata_path, 'r') as f:
            model_meta = json.load(f)
    except Exception:
        model_meta = None

if model_meta:
    st.sidebar.write(f"Loaded model: {MODEL_PATH.name}")
    st.sidebar.write(f"sklearn: {model_meta.get('sklearn_version')}")
else:
    import sklearn

    st.sidebar.write(f"Loaded model: {MODEL_PATH.name}")
    st.sidebar.write(f"sklearn (env): {sklearn.__version__}")

st.set_page_config(page_title="RE-Novate", layout='centered')
st.title("Entrepreneurial Skill Simulator")
st.write("Fill in the details below to see your entrepreneurial skill prediction!")


# Sidebar: explanation and demo link
with st.sidebar:
    st.header("About this demo")
    st.markdown(
        "This app loads a trained Random Forest model and produces a probability of exhibiting entrepreneurial skill. "
        "See `docs/RESULTS.md` in the repo for numeric metrics and notes about model limitations.")
    st.markdown("---")
    st.write(f"scikit-learn (installed): {sklearn.__version__}")
    if st.button("Open demo checklist"):
        st.write("See README.md -> Demo checklist for the 5-minute walkthrough.")


# User input form with demo prefill support
def user_inputs(prefill=None):
    if prefill is None:
        age = st.number_input('Age', min_value=15, max_value=19, value=17, step=1)
        prior_business_exposure = st.selectbox('Prior Business Exposure', [0,1])
        risk_taking = st.slider('Risk-taking Score', 0.0, 1.0, 0.5)
        decision_speed = st.slider('Decision Speed', 0.0, 1.0, 0.5)
        creativity_score = st.slider('Creativity Score', 0.0, 1.0, 0.5)
        leadership_experience = st.selectbox('Leadership Experience', [0,1])
    else:
        age = st.number_input('Age', min_value=15, max_value=19, value=prefill.get('age', 17), step=1)
        prior_business_exposure = st.selectbox('Prior Business Exposure', [0,1], index=1 if prefill.get('prior_business_exposure', 0)==1 else 0)
        risk_taking = st.slider('Risk-taking Score', 0.0, 1.0, prefill.get('risk_taking', 0.5))
        decision_speed = st.slider('Decision Speed', 0.0, 1.0, prefill.get('decision_speed', 0.5))
        creativity_score = st.slider('Creativity Score', 0.0, 1.0, prefill.get('creativity_score', 0.5))
        leadership_experience = st.selectbox('Leadership Experience', [0,1], index=1 if prefill.get('leadership_experience', 0)==1 else 0)

    return np.array([[age, prior_business_exposure, risk_taking, decision_speed, creativity_score, leadership_experience]])


# Demo mode
if 'demo_mode' not in st.session_state:
    st.session_state['demo_mode'] = False

col1, col2 = st.columns([3,1])
with col2:
    demo_checkbox = st.checkbox('Demo mode', value=st.session_state.get('demo_mode', False))
    # Update session_state from the checkbox; Streamlit will rerun automatically
    st.session_state['demo_mode'] = demo_checkbox

prefill = None
if st.session_state.get('demo_mode'):
    # Representative sample values (tune these if you want different demo behaviour)
    prefill = {
        'age': 17,
        'prior_business_exposure': 1,
        'risk_taking': 0.7,
        'decision_speed': 0.6,
        'creativity_score': 0.8,
        'leadership_experience': 1,
    }

input_data = user_inputs(prefill=prefill)

if st.button('Predict Skill Level'):
    # The model expects the same preprocessing as training (scaling/encoding).
    try:
        # Convert input to DataFrame with feature names to match how the model was trained
        features = ['age', 'prior_business_exposure', 'risk_taking', 'decision_speed', 'creativity_score', 'leadership_experience']
        input_df = pd.DataFrame(input_data, columns=features)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
    except Exception as e:
        st.error(f"Error running model prediction: {e}")
    else:
        if prediction == 1:
            if probability is not None:
                st.success(f"High Entrepreneurial Skill! (Confidence: {probability:.2f})")
            else:
                st.success("High Entrepreneurial Skill!")
            st.write("You demonstrate high potential for entrepreneurship. Keep building your creativity and leadership skills!")
        else:
            if probability is not None:
                st.warning(f"Lower estimated entrepreneurial skill. (Confidence: {1-probability:.2f})")
            else:
                st.warning("Lower estimated entrepreneurial skill.")

            # Improved, student-friendly recommendation templates: Quick / Medium / Long
            st.markdown("**Suggested next steps (quick → long-term)**")
            # Build templated recommendations based on top model signals
            # We'll approximate which features are limiting the score by checking where the user's value is below the median (if metadata exists)
            features = ['age', 'prior_business_exposure', 'risk_taking', 'decision_speed', 'creativity_score', 'leadership_experience']
            user_vals = {}
            for f in features:
                try:
                    v = input_df.at[0, f]
                    user_vals[f] = float(v) if v is not None else None
                except Exception:
                    user_vals[f] = None
            medians = {}
            if model_meta and 'feature_medians' in model_meta:
                medians = {f: model_meta['feature_medians'].get(f) for f in features}

            # Determine a few low features to target: where user < median (fallback: numeric thresholds)
            low_features = []
            for f in features:
                fv = user_vals.get(f, 0)
                med = medians.get(f)
                if med is not None:
                    if fv < med:
                        low_features.append(f)
                else:
                    # simple heuristic thresholds for this demo
                    if f in ['risk_taking', 'decision_speed', 'creativity_score'] and fv < 0.6:
                        low_features.append(f)
                    if f in ['prior_business_exposure', 'leadership_experience'] and fv < 1:
                        low_features.append(f)

            # Compose three templated suggestions
            quick = "Join a local entrepreneurship club or take part in one extracurricular project this month to gain exposure. (Quick win)"
            medium = "Run a small idea test: build a simple plan and get feedback from peers or teachers; target one deliverable in 2–4 weeks. (Medium effort)"
            long = "Develop a longer-term project where you lead a small team or startup-style project over a semester — this builds leadership and practical experience. (Long-term)"

            # Personalize by mentioning one low feature if present
            if low_features:
                first_low = low_features[0]
                mapping = {
                    'risk_taking': 'Take more calculated risks by trying small projects or pitches',
                    'decision_speed': 'Practice quick decision-making in timed exercises',
                    'creativity_score': 'Work on idea generation exercises and keep an ideas journal',
                    'prior_business_exposure': 'Seek short internships or shadow a local business owner',
                    'leadership_experience': 'Volunteer to lead a group task in class or clubs',
                    'age': 'Engage with age-appropriate mentorship programs'
                }
                personalized = mapping.get(first_low, '')
                # Insert the personalized line into the medium suggestion for clarity
                medium = personalized + ". " + medium

            st.write(f"Quick: {quick}")
            st.write(f"Medium: {medium}")
            st.write(f"Long: {long}")

                # model-level explanations removed per user request

        # Optional feature importance display
        if hasattr(model, 'feature_importances_'):
            st.markdown('---')
            st.subheader('Feature importances (model)')
            fi = model.feature_importances_
            features = ['age', 'prior_business_exposure', 'risk_taking', 'decision_speed', 'creativity_score', 'leadership_experience']
            importance_pairs = list(zip(features, fi))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            for feat, val in importance_pairs:
                st.write(f"{feat}: {val:.3f}")

