# Diabetes Prediction using Machine Learning

A multi-model classification project predicting diabetes likelihood using the PIMA Indians Diabetes dataset. Five supervised learning algorithms are compared with full EDA, feature engineering, and model evaluation.

> **Collaborative project** — developed with Bhavya Sharma as part of End-Term AI/ML coursework at The NorthCap University (Semester 5).

---

## 📊 Model Comparison

| Model | Accuracy | Precision (Diabetic) | Recall (Diabetic) | F1 (Diabetic) |
|-------|----------|---------------------|-------------------|---------------|
| **Logistic Regression** | **79%** | 0.73 | 0.62 | 0.67 |
| Random Forest | 75% | — | — | — |
| KNN (k=20, CV) | 75% | — | — | — |
| Decision Tree | 75% | 0.62 | 0.65 | 0.64 |
| Naive Bayes | 75% | 0.65 | 0.58 | 0.61 |

> **Best model: Logistic Regression** — highest accuracy (79%) and best F1 score for diabetic class detection (0.67).

---

## 🎯 Objective

Predict whether a patient is diabetic based on medical diagnostic measurements including glucose concentration, BMI, insulin levels, age, and number of pregnancies.

---

## 📂 Dataset

- **Source:** [Kaggle — PIMA Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Samples:** 768 total · 154 test samples
- **Features:** num_preg, glucose_conc, diastolic_bp, thickness, insulin, bmi, diab_pred, age, skin
- **Target:** diabetes (True/False → 1/0)
- **Class distribution:** ~65% non-diabetic · ~35% diabetic

---

## 🔧 Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Environment:** Jupyter Notebook / Google Colab

---

## 🚀 Workflow

1. **Data Loading & Exploration** — shape, dtypes, statistical summary
2. **Null Value Handling** — fill missing values with mean/mode
3. **EDA & Visualizations** — countplot, bar graph, box plot, pair plot
4. **Label Encoding** — convert boolean diabetes column to 0/1
5. **Feature Selection** — Chi-Squared test (SelectKBest) to identify top 6 features
6. **Feature Scaling** — StandardScaler for mean=0, std=1 normalization
7. **Train/Test Split** — 80/20 split with random_state=5
8. **Model Training** — 5 classifiers trained and evaluated
9. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix

---

## 📈 Key Insights

- **Glucose concentration** is the strongest predictor of diabetes (highest Chi-Squared score)
- **Insulin** and **BMI** are the next most important features
- Patients under 30 with diabetes show high variance in diastolic blood pressure
- All models struggle with diabetic class recall — class imbalance (35% positive) is the main challenge
- Logistic Regression outperforms tree-based models on this dataset

---

## 🔍 Feature Importance (Chi-Squared Test)

Top features selected for model training:
```
num_preg, glucose_conc, thickness, insulin, bmi, age
```
These 6 features were selected out of 10 based on Chi-Squared scores.

---

## ⚠️ Known Limitations

- Class imbalance (35% diabetic) affects recall for the positive class
- KNN uses cross-validation accuracy — direct test set evaluation not included
- Future improvements: SMOTE oversampling, XGBoost, hyperparameter tuning

---

## 📁 Files

| File | Description |
|------|-------------|
| `diabetes-analysis.ipynb` | Full notebook — EDA, preprocessing, all 5 models |
| `Diabetes_dataset.csv` | PIMA Indians diabetes dataset |
| `AIML Project Report.pdf` | Full project report with methodology and findings |

---

## ▶️ How to Run

```bash
git clone https://github.com/yukta31/diabetes-prediction-ml.git
cd diabetes-prediction-ml
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook diabetes-analysis.ipynb
```

Or open in Google Colab — upload `Diabetes_dataset.csv` to the Colab session files first.

---

## 👩‍💻 Authors

**Yukta Batra** · **Bhavya Sharma**
The NorthCap University · B.Tech Computer Science · Semester 5

[Portfolio](https://yukta-batra.vercel.app) · [LinkedIn](https://linkedin.com/in/yuktabatra31) · [GitHub](https://github.com/yukta31)
