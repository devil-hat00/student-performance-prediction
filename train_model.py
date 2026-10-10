from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "student_performance.csv")
features = ["study_hours_per_day","attendance_percent","previous_exam_score","assignment_average","sleep_hours_per_day","internet_access","extracurricular_activities"]
X, y = df[features], df["passed"]
numeric, categorical = features[:5], features[5:]
prep = ColumnTransformer([
 ("numeric", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]), numeric),
 ("categorical", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
pipeline = Pipeline([("preprocessor",prep),("classifier",RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced"))])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
pipeline.fit(X_train,y_train)
pred = pipeline.predict(X_test)
prob = pipeline.predict_proba(X_test)[:,1]
cm = confusion_matrix(y_test,pred)
fpr,tpr,_ = roc_curve(y_test,prob)
metrics = {"dataset_rows":len(df),"train_rows":len(X_train),"test_rows":len(X_test),
 "accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),
 "recall":recall_score(y_test,pred,zero_division=0),"f1_score":f1_score(y_test,pred,zero_division=0),
 "roc_auc":auc(fpr,tpr),"confusion_matrix":cm.tolist(),"confusion_matrix_labels":["Not passed","Passed"]}
(OUT/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2))
(OUT/"classification_report.txt").write_text(classification_report(y_test,pred,target_names=["Not passed","Passed"],zero_division=0))
counts=df["passed"].map({0:"Not passed",1:"Passed"}).value_counts().reindex(["Not passed","Passed"],fill_value=0)
plt.figure(figsize=(6,4)); counts.plot(kind="bar"); plt.title("Student Outcome Distribution"); plt.xlabel("Outcome"); plt.ylabel("Number of students"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/"outcome_distribution.png",dpi=160); plt.close()
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Not passed","Passed"]).plot(values_format="d"); plt.title("Confusion Matrix — Test Set"); plt.tight_layout(); plt.savefig(OUT/"confusion_matrix.png",dpi=160); plt.close()
plt.figure(figsize=(6,4)); plt.plot(fpr,tpr,label=f"Random Forest (AUC = {metrics['roc_auc']:.3f})"); plt.plot([0,1],[0,1],linestyle="--",label="Random baseline"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"roc_curve.png",dpi=160); plt.close()
names=pipeline.named_steps["preprocessor"].get_feature_names_out()
imp=pipeline.named_steps["classifier"].feature_importances_
imp_df=pd.DataFrame({"feature":names,"importance":imp}).sort_values("importance",ascending=False).head(10)
plt.figure(figsize=(8,5)); plt.barh(imp_df["feature"][::-1],imp_df["importance"][::-1]); plt.title("Top 10 Feature Importances"); plt.xlabel("Importance"); plt.tight_layout(); plt.savefig(OUT/"feature_importance.png",dpi=160); plt.close()
joblib.dump(pipeline,ROOT/"student_pass_model.joblib")
print("Student Performance Prediction completed.")
print(f"Rows: {len(df)} | Training: {len(X_train)} | Testing: {len(X_test)}")
for k in ["accuracy","precision","recall","f1_score","roc_auc"]: print(f"{k}: {metrics[k]:.3f}")
print("Graphs and metrics saved in outputs/")
