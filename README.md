# 🩺 Diabetes Risk Predictor

A machine learning-powered web application that predicts diabetes risk based on health parameters. Built with a **Random Forest Classifier**, a **Flask REST API**, and a **React frontend**.

---

## 📁 Project Structure

```
diabetes-risk-predictor/
├── ml/
│   ├── train.py               # Model training script
│   ├── diabetes.csv           # Training dataset (Pima Indians Diabetes)
│   ├── model.pkl              # Trained model (generated after training)
│   └── confusion_matrix.png  # Evaluation chart (generated after training)
├── backend/
│   └── app.py                 # Flask API server
├── frontend/
│   └── ...                    # React frontend
└── README.md
```

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Dataset:** Pima Indians Diabetes Dataset (`diabetes.csv`)
- **Target:** `Outcome` — `1` (Diabetic) / `0` (Non-Diabetic)
- **Accuracy:** ~76%

### Features Used

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function score |
| Age | Age in years |

### Train the Model

```bash
cd ml
pip install pandas scikit-learn matplotlib seaborn
python train.py
```

This generates `model.pkl` and `confusion_matrix.png` inside the `ml/` folder.

---

## 🔌 Backend — Flask API

### Setup & Run

```bash
cd backend
pip install flask flask-cors numpy pandas scikit-learn
python app.py
```

API runs at `http://localhost:5000`

### Endpoints

#### `GET /`
Health check.

```json
{
  "success": true,
  "status": "online",
  "message": "Diabetes Prediction API is running."
}
```

#### `POST /predict`
Predict diabetes risk from health inputs.

**Required fields:** `glucose`, `bloodPressure`, `bmi`, `age`

**Optional fields (with defaults):** `pregnancies`, `skinThickness`, `insulin`, `diabetesPedigree`

**Request Body:**
```json
{
  "glucose": 148,
  "bloodPressure": 72,
  "bmi": 33.6,
  "age": 50,
  "pregnancies": 6,
  "skinThickness": 35,
  "insulin": 0,
  "diabetesPedigree": 0.627
}
```

**Response:**
```json
{
  "success": true,
  "isDiabetic": true,
  "diabetesChance": 74.5,
  "confidence": 49.0,
  "riskLevel": "High",
  "bmiCategory": "Obese",
  "contributingFactors": [
    "High glucose level detected",
    "BMI indicates obesity",
    "Age is a risk factor (above 45)"
  ],
  "healthTips": [
    "Consult a doctor immediately for clinical evaluation",
    "Reduce sugar and refined carbohydrate intake"
  ],
  "featureImportance": { "glucose": 38, "bmi": 21, "age": 14, "...": "..." },
  "modelAccuracy": "76%",
  "message": "High diabetes risk detected."
}
```

### Field Validation Ranges

| Field | Min | Max |
|---|---|---|
| pregnancies | 0 | 17 |
| glucose | 50 | 300 |
| bloodPressure | 30 | 140 |
| skinThickness | 0 | 99 |
| insulin | 0 | 846 |
| bmi | 10 | 70 |
| diabetesPedigree | 0.05 | 2.5 |
| age | 1 | 120 |

---

## 💻 Frontend — React

### Setup & Run

```bash
cd frontend
npm install
npm start
```

App runs at `http://localhost:3000`

Make sure the Flask backend is running before using the app.

---

## ⚙️ Full Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/diabetes-risk-predictor.git
cd diabetes-risk-predictor

# 2. Train the model
cd ml && python train.py && cd ..

# 3. Start the backend
cd backend && python app.py &

# 4. Start the frontend
cd frontend && npm install && npm start
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Python, scikit-learn, pandas |
| Backend | Flask, Flask-CORS |
| Frontend | React |
| Serialization | Pickle |

---

## ⚠️ Disclaimer

This tool is intended for **educational and informational purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for any health concerns.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
