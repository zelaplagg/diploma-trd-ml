# Predictive Diagnostics Subsystem for Turbofan Engine Health

> Machine-learning subsystem that estimates the technical condition of a turbofan
> engine from multivariate sensor data, using an ensemble of Random Forest and
> XGBoost. Built as a Bachelor's thesis project at Igor Sikorsky Kyiv Polytechnic
> Institute (KPI).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Accuracy](https://img.shields.io/badge/Accuracy-98.46%25-brightgreen)

---

## Overview

This project implements an end-to-end machine-learning pipeline for the **predictive
diagnostics of turbofan engines**. Given multivariate time-series readings from
engine sensors, the model classifies the engine's technical condition, supporting
early fault detection and condition-based maintenance.

The full lifecycle is covered: **data preprocessing → feature engineering → model
training → evaluation → deployment** through an interactive web interface.

- **Dataset:** NASA C-MAPSS (subset **FD001**)
- **Approach:** ensemble of **Random Forest + XGBoost**
- **Result:** **98.46% accuracy** on the test set
- **Interface:** interactive **Streamlit** web application with live prediction

---

## Key Features

- 🔧 **End-to-end ML pipeline** — from raw sensor data to a deployed predictor.
- 📊 **Multivariate time-series processing** — cleaning, normalization, and feature
  engineering on high-dimensional sensor signals.
- 🌲 **Ensemble model** — Random Forest combined with XGBoost gradient boosting for
  robust, high-accuracy classification.
- 🧪 **Reproducible evaluation** — train/test split, cross-validation, and standard
  classification metrics.
- 🖥️ **Interactive web app** — upload data or use the bundled demo file to get a
  prediction in real time, with an animated progress indicator and session state.

---

## Dataset

The project uses the **NASA Commercial Modular Aero-Propulsion System Simulation
(C-MAPSS)** turbofan degradation dataset, specifically the **FD001** subset.

Each record contains operational settings and 21 sensor measurements collected over
the engine's run-to-failure cycles. The data is preprocessed (cleaning, scaling,
feature selection) before being passed to the model.

> The C-MAPSS dataset is publicly available from the NASA Prognostics Center of
> Excellence (PCoE) data repository.

---

## Results

| Metric        | Value     |
|---------------|-----------|
| Model         | Random Forest + XGBoost (ensemble) |
| Accuracy      | **98.46%** |
| Dataset       | NASA C-MAPSS FD001 |

<!-- TODO: add precision / recall / F1 and a confusion-matrix image once exported -->

---

## Tech Stack

- **Language:** Python 3.10+
- **ML / Data:** scikit-learn, XGBoost, Pandas, NumPy
- **Visualization:** Matplotlib / Seaborn
- **Web interface:** Streamlit
- **Tooling:** Git, virtual environments

---

## Project Structure

```text
.
├── data/                 # Dataset and demo CSV
│   └── demo.csv          # Sample input for the web app
├── notebooks/            # Exploratory analysis and training experiments
├── src/                  # Source code (preprocessing, training, inference)
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── models/               # Saved trained models
├── app.py                # Streamlit web application
├── requirements.txt
└── README.md
```

> ℹ️ Adjust the paths above to match your actual repository layout.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/zelaplagg/<your-repo-name>.git
cd <your-repo-name>

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the web application

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).
Upload your sensor data — or load the bundled `demo.csv` — to get a prediction of
the engine's technical condition.

### Train the model from scratch

```bash
python src/train.py
```

### Run a prediction from the command line

```bash
python src/predict.py --input data/demo.csv
```

> ℹ️ Update the commands to match your actual script names and arguments.

---

## Screenshots

<img width="2940" height="1496" alt="image" src="https://github.com/user-attachments/assets/95c6b0a1-04e7-4777-a78f-a4988a115ce3" />

---

## Future Work

- Extend support to the remaining C-MAPSS subsets (FD002–FD004).
- Add Remaining Useful Life (RUL) regression alongside condition classification.
- Containerize the application with Docker for easier deployment.

---

## Author

**Anastasiia Zelenska**
Bachelor of Computer Systems Engineering — Igor Sikorsky Kyiv Polytechnic Institute (KPI)

- GitHub: [zelaplagg](https://github.com/zelaplagg)
- LinkedIn: [Anastasiia Zelenska](https://www.linkedin.com/in/anastasiia-zelenska-50404a27a)
