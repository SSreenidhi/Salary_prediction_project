# 🧠 Salary Prediction using Artificial Neural Networks

This project focuses on predicting salary (in USD) using various job-related features, including experience level, education, remote ratio, company and employee location, skills, and job description metadata. The goal is to build a high-performing regression model using **Artificial Neural Networks (ANNs)** that can capture non-linear relationships in real-world salary data.

---

## 📌 Why This Project?

Salary prediction is a real-world challenge influenced by various dynamic factors like experience, skills, location, education, and industry.  
By solving this problem:
- I learned how to **clean and analyze messy real-world datasets**
- Applied **feature engineering** and **deep learning (ANNs)** for regression
- Understood trade-offs between model complexity, overfitting, and accuracy
- Delivered a solid, explainable model with strong predictive performance

---

## 📊 Dataset Overview

The dataset consists of job listings with the following fields:
- `salary_usd` (target)
- `experience_level`, `education_required`, `employment_type`
- `company_location`, `employee_residence`, `company_size`
- `required_skills`, `job_description_length`, `benefits_score`
- `posting_date`, `application_deadline`, `industry`

---

## 🧼 Data Cleaning & Preprocessing

- **Removed outliers** using 99th percentile and IQR method
- Extracted date features: `posting_month`, `deadline_month`, `days_until_deadline`, `is_short_deadline`
- Transformed `required_skills` into binary flags: `has_Python`, `has_SQL`, `has_NLP`, etc.
- Applied **Label Encoding** and **One-Hot Encoding** to categorical features
- Dropped noisy/unnecessary columns: `job_id`, `company_name`, etc.
- Scaled numerical features using **StandardScaler**

---

## 🧠 Model Building (ANN)

Several ANN architectures were tested (Model 1 to Model 8), with variations in:
- Number of layers and neurons
- Use of `BatchNormalization` and `Dropout`
- Activation functions (`ReLU`, `Linear`)
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`

---

## 🧪 Final Results

📌 **Best Performing Model**: `Model 8`

| Metric | Value |
|--------|-------|
| MAE    | ~14,108 USD |
| RMSE   | ~17,530 USD |
| R² Score | **0.8828** ✅ |

✅ The model explains approximately **88.28% of the variance** in salary data — a strong performance given the unpredictable nature of salaries.

---

## 📌 Key Learnings

- Feature engineering and encoding decisions have a **huge impact** on model performance
- ANN requires **careful tuning** (dropout, batch norm, early stopping) to avoid overfitting
- Real-world data is noisy — handling outliers improves generalization
- Regression metrics like **MAE, RMSE, R²** must be analyzed together, not in isolation

---

## 🚀 Tools & Technologies

- Python, Pandas, NumPy
- Seaborn & Matplotlib (EDA/Visualization)
- Scikit-learn (Encoding, Scaling, Evaluation)
- TensorFlow/Keras (Modeling)
- Google Colab (Execution)

