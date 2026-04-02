# Heart Disease & Stroke Diagnosis with Random Forest

**Course:** Intelligent Systems (CO5134) — Ho Chi Minh University of Technology (VNU-HCM)
**Instructor:** Prof. Quan Thanh Tho
**Students:** Vuong Minh Toan (2491057), Tran Nguyen Huan (2370692)
**Date:** November 2024
**Deliverable:** Web application (React + Flask) & ML pipeline for diagnostic support

---

## 📌 Objectives

* Develop a **Random Forest model** to predict **heart disease** and **stroke** based on clinical and lifestyle data.
* Provide a **web application** for the community: input health indicators → receive predictions + basic recommendations → download reports.

---

## 👥 Stakeholders & Benefits

* **General users:** self-assess risk, improve awareness, low cost.
* **Community organizations/NGOs:** support health campaigns, identify high-risk groups.
* **Government/Public health sector:** monitor health trends (anonymized), design early interventions.

---

## 🧩 System Requirements

**Functional:** input personal data, display predictions + insights, export reports.
**Non-functional:** intuitive UI, **response time < 5 seconds**, target uptime **99.9%**.

---

## 🏗️ Architecture & Technology

* **Frontend:** React.js — input forms, results page, responsive design, API integration via Axios.
* **Backend:** Flask (Python) — load *.pkl* models, data processing, prediction, return results; modular architecture.
* **Overall flow:** user input → frontend calls REST API → backend preprocessing & inference → return results → user downloads report.
* **Deployment (suggested):** React (Netlify/Vercel/S3), Flask (Gunicorn/uWSGI on EC2/Heroku/Azure), monitoring via Sentry/Prometheus/Grafana, analytics with Google Analytics.

---

## 🗃️ Data

* **heart.csv** (~1025 samples) — features: *age, sex, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target*.
  **`target`**: 1 = heart disease, 0 = no disease.

* **stroke.csv** (~5110 samples) — features: *gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status, stroke*.
  **`stroke`**: 1 = stroke, 0 = no stroke.

### Key Preprocessing Steps

* **Encoding:** binary label mapping; **One-Hot Encoding** for categorical variables (work_type, residence_type, smoking_status…).
* **Missing data:** ~201 NULL values in `bmi` (stroke.csv) → **dropped** due to sufficient dataset size and numeric nature.
* **Imbalanced data:** stroke dataset is highly imbalanced → applied **SMOTE** for minority class oversampling.
* **Scaling & splitting:** train/test split; normalization using `StandardScaler` and `shuffle` when needed.

---

## 🤖 Model & Pipeline

1. **Exploration:** used **LazyPredict** to benchmark multiple supervised models → selected **Random Forest** due to strong performance on both tasks.

2. **Training:**
   `RandomForestClassifier(random_state=42)` + hyperparameter tuning using **GridSearchCV**
   (parameters: `n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features`).

3. **Evaluation:** Accuracy, Confusion Matrix, Classification Report, **ROC/AUC**.

### Results (from report)

* **Heart disease:** *Accuracy ≈ 0.99*, **AUC = 1.00** (ROC reaches top-left corner).
* **Stroke:** *Accuracy ≈ 0.98*, **AUC = 1.00** (very high discriminative performance).

> Note: Some reported values in the confusion matrix section are not fully consistent. It is recommended to rerun the notebook/script to extract final metrics from `best_estimator_`.

---

## 🔌 API (Draft)

* `POST /api/predict/heart` → receive health indicators → return heart disease prediction.
* `POST /api/predict/stroke` → receive health indicators → return stroke prediction.
* `GET /api/healthz` → check server status.
* `POST /api/feedback` → collect user feedback (optional).

> Backend handles data type conversion, loads *.pkl* models, performs inference, and returns JSON responses; frontend displays results and allows report download.

---

## 🧪 How to Run (Suggested)

### Backend (Flask)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place models at ./models/heart_rf.pkl and ./models/stroke_rf.pkl
export FLASK_APP=app.py
flask run --host 0.0.0.0 --port 8000
```

### Frontend (React)

```bash
cd web
npm install
npm run dev  # or npm run build && npm run preview
```

---

## 🔒 Ethical & Legal Considerations

* This is a **decision-support tool**, **not a substitute for medical diagnosis**. Users are strongly advised to **consult healthcare professionals**.
* Ensure **data anonymization** and **security compliance** with local regulations.

---

## 🔗 Source Code & Demo

* **Source code:** (add Git/Drive link) — currently referenced as **Google Drive**.
* **Demo:** (add link) — currently referenced as **YouTube**.

---

## 📣 Future Improvements

* Tune **decision thresholds** to improve **recall** (reduce false negatives).
* Add **class weighting**, **probability calibration** (Platt/Isotonic), and **explainability** (SHAP).
* Implement **data drift monitoring** and post-deployment model tracking.

---

