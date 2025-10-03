# Entrepreneurial Skills Simulator – ML Capstone

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-orange)](#tests)
[![License-MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Random Forest‑based demo that predicts a student's entrepreneurial-skill likelihood from simple inputs and exposes a tidy Streamlit interface for quick demos and grading.

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


```text
RE-Novate/
├── deployment/
│   └── Dockerfile
├── model/
│   └── entrepreneurial_skill_model.joblib
├── notebook/
│   └── ml_capstone_notebook.ipynb
├── streamlit_app/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
---

## 📑 Description

- **notebook/ml_capstone_notebook.ipynb:** Data analysis, visualization, model training & evaluation.
- **model/:** Saved model and scaler for app prediction.
- **streamlit_app/app.py:** Student-facing Streamlit UI.
- **deployment/Dockerfile:** Docker setup for containerized deployment.
- **requirements.txt:** Python dependencies.

---

## 🔗 GitHub Repo

- Repository: https://github.com/jsumbo/RE-Novate.git

- Live demo: https://re-novate.streamlit.app/

---

## ⚡ Getting Started

Follow these steps to create a local development environment, install dependencies, and run the app.

### Quick start
Open PowerShell and run:

```powershell
Set-Location -LiteralPath 'C:\Users\admin\Documents\RE-Novate'
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app\app.py
```

1. Clone the repository

```bash
git clone https://github.com/jsumbo/RE-Novate.git
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

- Accuracy reported in the notebook: ~0.88
- Precision/recall for the positive class in the notebook were reported as 0.00 (UndefinedMetricWarning) — likely due to class imbalance or the classifier predicting a single class. Please see `docs/RESULTS.md` for next steps (rebalancing, threshold tuning).

Limitations:

- Small, simulated dataset in the notebook is for demonstration; treat model outputs as illustrative rather than production-ready.
- If you see an sklearn unpickle warning when running tests, consider retraining or re-saving the model with the scikit-learn version in `requirements.txt`.

---

## 📄 License

MIT
