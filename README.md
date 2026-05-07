# 🚗 AutoVal AU — Australian Used Car Price Predictor

> An end-to-end machine learning project that predicts used car prices in Australia using real market data.  
> **Live demo:** [autoval-au.onrender.com](https://autoval-au.onrender.com) *(deploy on Render to activate)*

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R² (test set) | **0.8064** |
| Mean Absolute Error | **$5,132** |
| Root Mean Square Error | **$10,378** |
| Training samples | 11,876 |

---

## 🏗️ Project Pipeline

```
Raw Data → Validation → Cleansing & EDA → Feature Engineering → Modelling → Deployment
```

### 1. Data Validation
- Source: `AustralianVehiclePrices.csv` (~14,000 real Australian car listings)
- Cross-validated brand coverage against US MPG dataset (1984–2023)
- Identified 32 Australian-exclusive brands (Holden, HSV, GWM, Haval, etc.)

### 2. Data Cleansing & Feature Engineering
- Removed year < 1998 and top 0.5% price outliers
- Extracted engine cylinders and capacity from raw text strings
- Parsed state from location strings; imputed missing categorical values
- Applied **log-price transformation** to normalise the target distribution
- Imputed missing numerical values with column medians

### 3. Tree-Based Models

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Decision Tree (baseline) | — | — | — |
| Decision Tree (tuned, GridSearchCV) | — | — | — |
| Random Forest | — | — | — |
| Random Forest (tuned) | — | — | — |
| **Gradient Boosting (tuned)** | **$5,132** | **$10,378** | **0.8064** |

GBM was tuned via 3-fold cross-validation across `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, and `max_features`.

### 4. Neural Network Architectures

Two architectures were explored using TensorFlow/Keras:

- **Model 1 — Tabular MLP**: structured inputs → OHE/OrdinalEncoder/StandardScaler → Dense(128) → Dense(128) → Dense(32) → output
- **Model 2 — Combined model**: structured inputs + GloVe word embeddings on text features (Title, Model, ColourExtInt) → Concatenate → Dense layers → output. Includes a dropout variant for regularisation.

### 5. Deployment
- Best model (GBM) serialised as `gbm_pipeline.pkl` via `joblib`
- Served via a **Flask REST API** (`app.py`)
- Interactive **HTML/JS front-end** (`index.html`) with live price estimation, feature importance breakdown, and confidence range

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/keithtang2025/UsedCarPrediction2026.git
cd UsedCarPrediction2026

# Install dependencies
pip install -r requirements.txt

# Start the API + UI
python app.py

# Open http://localhost:5000
```

## 🌐 API Usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "year": 2019,
    "usedOrNew": "USED",
    "transmission": "Automatic",
    "driveType": "Front",
    "fuelType": "Unleaded",
    "fuelConsumption": 7.5,
    "kilometres": 50000,
    "cylindersInEngine": "4 cyl",
    "bodyType": "SUV",
    "doors": "4 Doors",
    "seats": "5 Seats",
    "state": "NSW",
    "cylinders": 4,
    "capacity": 2.5
  }'
```

**Response:**
```json
{
  "predicted_price": 34700,
  "range_low": 27800,
  "range_high": 43300,
  "status": "success"
}
```

---

## 📁 Repository Structure

```
UsedCarPrediction2026/
├── AI_in_ACTL_byKeithTANG.ipynb   # Full analysis notebook
├── app.py                          # Flask REST API
├── index.html                      # Interactive front-end UI
├── gbm_pipeline.pkl                # Trained sklearn pipeline
├── requirements.txt                # Python dependencies
├── DEPLOY.md                       # Deployment guide (Render, Fly.io, Railway)
└── README.md
```

---

## 🔑 Key Features Used by the Model

| Feature | Importance |
|---------|-----------|
| Year of manufacture | 34.2% |
| Odometer (km) | 18.0% |
| Fuel type | 6.1% |
| Body type | 3.8% |
| Cylinders in engine | 4.2% |
| Engine capacity | 3.9% |
| Drive type | 3.6% |
| Condition (Used/New/Demo) | 2.6% |

---

## 👤 Author

**Keith Tang**  
Quantitative Analyst | Sydney, Australia  
[LinkedIn](https://linkedin.com/in/keithtang) · [GitHub](https://github.com/keithtang2025)

---

*Built as a demonstration of end-to-end ML project capability — from raw data to deployed web application.*
