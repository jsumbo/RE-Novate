# Entrepreneurial Skills Simulator – ML Capstone

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-orange)](#tests)
[![License-MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Short summary: a small Random Forest‑based demo that predicts a student's entrepreneurial-skill likelihood from simple inputs and exposes a tidy Streamlit interface for quick demos and grading.

<!-- TOC -->
## Table of contents

- [Project structure](#project-structure)
- [Getting Started](#getting-started-quick-reproducible-setup)
- [Demo checklist](#demo-checklist-5-minute-walkthrough)
- [Design](#design)
- [Results & limitations](#results--limitations)
- [Deployment](#deployment)
- [Tests](#tests)
- [License](#license)

<!-- /TOC -->

---

## 🚀 Project Structure

RE-Novate/
├── deployment/
│ └── Dockerfile
├── model/
│ └── entrepreneurial_skill_model.joblib
├── notebook/
│ └── ml_capstone_notebook.ipynb
├── streamlit_app/
│ └── app.py
├── requirements.txt
├── README.md
└── .gitignore


---

## 📑 Description

- **notebook/ml_capstone_notebook.ipynb:** Data analysis, visualization, model training & evaluation.
- **model/:** Saved model and scaler for app prediction.
- **streamlit_app/app.py:** Student-facing Streamlit UI for skill prediction.
- **deployment/Dockerfile:** Docker setup for containerized deployment.
- **requirements.txt:** Python dependencies.

---

## 🔗 GitHub Repo

https://github.com/jsumbo/RE-Novate.git

---

## ⚡ Getting Started (quick reproducible setup)

Follow these steps to create a local development environment, install dependencies, and run the app.

### Quick start (copy/paste)
Open PowerShell and run:

```powershell
Set-Location -LiteralPath 'C:\Users\admin\Documents\RE-Novate'
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app\app.py
```

1. Clone the repository

```bash
git clone https://github.com/your-username/your-capstone-repo.git
cd RE-Novate
```

2. Create and activate a Python virtual environment

PowerShell (recommended):

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
# If you get an execution policy error, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Windows (cmd.exe):

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

macOS / Linux (bash / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. (Optional) Run the Jupyter notebook

```bash
cd notebook
jupyter notebook RE_Novate.ipynb
```

5. Run the Streamlit app locally

From the project root:

```bash
streamlit run streamlit_app\app.py
```

---

Demo checklist (5-minute walkthrough)

- Show the project README and explain the goal (predict entrepreneurial skill likelihood).
- Open `notebook/RE_Novate.ipynb` and briefly point out the dataset, model choice (RandomForest), and the key numeric results in `docs/RESULTS.md`.
- Launch the Streamlit app and use the "Demo mode" (pre-filled sample inputs) to run a prediction and show the probability/confidence.
- Explain model limitations (class imbalance / low precision-recall) and point to the interpretability section (feature importance plot) in `docs/RESULTS.md`.
- Run the unit test suite (or at least `tests/test_inference.py`) to show the model loads and performs a smoke inference.

Notes

- The repository uses a hidden `.venv` directory (recommended) to keep your environment local to the project. Use the activation commands above for your shell.
- For reproducible installs, `requirements.txt` should contain pinned versions. If you see unpinned packages, run `pip freeze > requirements.txt` from the activated `.venv` to capture exact versions.
- To run tests (if added): `pytest -q`



---

## 🌐 Deployment

### Deploy on Streamlit Community Cloud (Recommended)

1. Push your repo to GitHub.
2. Go to https://streamlit.io/cloud
3. Click **New app**, pick your repo/branch, set app path to `streamlit_app/app.py`.
4. Click **Deploy**.

### Deploy using Docker

cd deployment
docker build -t entrepreneur-simulator .
docker run -p 8501:8501 entrepreneur-simulator


Visit [`http://localhost:8501`](http://localhost:8501) in your browser.

---

## 🎬 Video Demo

View the 5-minute demo recording here:

[Demo video (Google Drive)](https://drive.google.com/drive/folders/1Y2HqxgIwvHjurHoLTAARfOk0zUH2i_AO?usp=sharing)


---

## Results & limitations

See `docs/RESULTS.md` for numeric metrics and interpretation. Short summary:

- Accuracy reported in the notebook: ~0.88
- Precision/recall for the positive class in the notebook were reported as 0.00 (UndefinedMetricWarning) — likely due to class imbalance or the classifier predicting a single class. Please see `docs/RESULTS.md` for next steps (rebalancing, threshold tuning).

Limitations:

- Small, simulated dataset in the notebook is for demonstration; treat model outputs as illustrative rather than production-ready.
- If you see an sklearn unpickle warning when running tests, consider retraining or re-saving the model with the scikit-learn version in `requirements.txt`.

### Screenshot

Below is a placeholder for a Streamlit app screenshot. Replace `docs/images/app_screenshot.png` with a real image (or run the app and take a screenshot).

![App screenshot](docs/images/app_screenshot.svg)


---

## 🖼️ Design

*Add Figma or mockup screenshots and user flows as needed here.*

---

## ☑️ Rubric Checklist

- [x] **Review of requirements & tools**
- [x] **Development environment setup**
- [x] **Navigation & layout**
- [x] **Initial software demo (notebook + Streamlit app)**

---

## 📄 License

MIT
