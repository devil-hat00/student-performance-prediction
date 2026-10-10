# Student Performance Prediction

## Project overview
Train a Random Forest classifier to predict a student's simulated pass/fail outcome from seven study and engagement features.

**Dataset caveat:** The included 600-row dataset is synthetic, not real student data. Performance metrics are for demonstration only.

## Files
- `student_performance.csv` — dataset
- `train_model.py` — trains/evaluates model and regenerates plots
- `predict_sample.py` — predicts the sample row
- `sample_input.csv` — sample input
- `requirements.txt` — dependencies
- `REPORT.md` — submission-ready report; fill in your details
- `outputs/` — metrics, classification report, and graphs

## Run on Windows
Open terminal in this folder and run:
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python predict_sample.py
```
If PowerShell blocks activation, use Command Prompt or invoke `.venv\Scripts\python.exe` directly.

## Workflow
Load data → preprocess numeric and categorical columns → split 80/20 → train Random Forest → evaluate → save graphs and metrics.

## Current run metrics
- Accuracy: 0.708
- Precision: 0.684
- Recall: 0.531
- F1-score: 0.598
- ROC-AUC: 0.762

Run `train_model.py` to regenerate these outputs in your own environment. Do not use this educational model for consequential real-student decisions.

## Submission checklist
- [ ] Run training script without errors.
- [ ] Include CSV, Python scripts, requirements, and outputs folder.
- [ ] Fill your name, roll number, and date in `REPORT.md`.
- [ ] Capture terminal output and graphs if screenshots are requested.
- [ ] Upload the ZIP file to your task portal.
