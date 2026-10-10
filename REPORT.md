# Project Report: Predictive Modeling Using Machine Learning

**Project:** Student Performance Prediction  
**Algorithm:** Random Forest Classifier  
**Language:** Python | **Libraries:** pandas, NumPy, Matplotlib, scikit-learn  
**Student name:** ____________________  
**Roll number:** ____________________  
**Course/section:** ____________________  
**Submission date:** ____________________

## 1. Abstract
This project demonstrates supervised machine learning by training a Random Forest classifier to predict whether a student is likely to pass. Inputs include study hours, attendance, previous exam score, assignment average, sleep duration, internet access, and extracurricular participation. A held-out test set is used for evaluation.

## 2. Problem statement and objectives
The objective is to demonstrate how a classification model learns from tabular data and predicts a binary outcome. Tasks include preparing the dataset, training a classifier, evaluating it with standard metrics, and visualizing the results.

## 3. Dataset
The CSV contains 600 **synthetic** examples generated for this educational project; it is not real student data.

| Feature | Meaning |
|---|---|
| study_hours_per_day | Daily study hours |
| attendance_percent | Attendance percentage |
| previous_exam_score | Previous score out of 100 |
| assignment_average | Assignment average out of 100 |
| sleep_hours_per_day | Daily sleep hours |
| internet_access | Yes/No |
| extracurricular_activities | Yes/No |
| final_marks | Simulated marks out of 100 |
| passed | Target: 1 if final marks ≥ 60, otherwise 0 |

## 4. Methodology
1. Load data with pandas.
2. Use seven student features as inputs and `passed` as the target.
3. Split into 80% training and 20% testing using a fixed random seed and stratification.
4. Impute missing values, scale numeric features, and one-hot encode categorical features.
5. Train a Random Forest classifier with 250 trees.
6. Evaluate predictions using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
7. Save metrics and four visualizations.

## 5. Algorithm
Random Forest is an ensemble method that trains multiple decision trees and combines their predictions. For classification, the ensemble predicts the class receiving the most votes. Randomization across trees can reduce the variance of a single decision tree.

## 6. Evaluation results
The test set contains 120 rows; the training set contains 480 rows.

| Metric | Result |
|---|---:|
| Accuracy | 0.708 |
| Precision | 0.684 |
| Recall | 0.531 |
| F1-score | 0.598 |
| ROC-AUC | 0.762 |

Accuracy measures overall correctness. Precision measures the fraction of predicted passes that are correct. Recall measures the fraction of actual passes detected. F1-score balances precision and recall. ROC-AUC measures discrimination across thresholds.

## 7. Visualizations
- `outcome_distribution.png`: counts of passing and not-passing records.
- `confusion_matrix.png`: correct and incorrect test predictions by class.
- `roc_curve.png`: true-positive versus false-positive rate.
- `feature_importance.png`: relative importance assigned by the fitted Random Forest.

## 8. Conclusion
The project demonstrates a complete supervised learning workflow from preprocessing through training, evaluation, and visualization. Because the dataset is synthetic, the results demonstrate code execution and methodology only; they do not establish real-world predictive accuracy.

## 9. Limitations and future work
Use a representative, authorized dataset before drawing real conclusions. Future work can compare Logistic Regression and Decision Tree baselines, use cross-validation, evaluate fairness across groups, and build a simple interface. Do not use this demonstration to grade or make consequential decisions about real students.

## 10. References
- scikit-learn: https://scikit-learn.org/stable/
- pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/
