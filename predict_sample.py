from pathlib import Path
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parent
model_path=ROOT/"student_pass_model.joblib"
if not model_path.exists(): raise SystemExit("Run python train_model.py first.")
model=joblib.load(model_path)
sample=pd.read_csv(ROOT/"sample_input.csv")
prediction=int(model.predict(sample)[0])
prob=float(model.predict_proba(sample)[0,1])
print("Predicted outcome:", "Passed" if prediction else "Not passed")
print(f"Estimated probability of passing: {prob:.1%}")
print("Educational demonstration only; not suitable for real academic decisions.")
