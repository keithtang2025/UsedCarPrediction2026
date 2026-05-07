# AutoVal AU — Deployment Guide

## Model Performance
- **Algorithm**: Gradient Boosting Machine (GBM)
- **R² Score**: 0.8064
- **MAE**: $5,132
- **RMSE**: $10,378
- **Training samples**: 11,876

## Local Setup

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## API Usage

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

## Cloud Deployment (Fly.io / Railway / Render)

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

2. Set `debug=False` and `host='0.0.0.0'` in `app.py`
3. Deploy via your platform's CLI

## Files
- `gbm_pipeline.pkl` — Trained sklearn pipeline (preprocessor + GBM)
- `app.py` — Flask REST API
- `index.html` — Frontend UI (works standalone or served by Flask)
- `requirements.txt` — Python dependencies
