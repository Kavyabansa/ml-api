# 🏠 House Prices Prediction API

A production-ready ML API serving a LightGBM model trained on the Kaggle House Prices dataset.
Containerized with Docker and deployed on Render.

## 🌐 Live API
**Base URL:** https://house-prices-api-13u5.onrender.com

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Check API status |
| /predict | POST | Predict house price |
| /docs | GET | Interactive API documentation |

---

## 📊 Model Performance
- **Algorithm:** LightGBM (Gradient Boosting)
- **Dataset:** Kaggle House Prices (1,460 houses)
- **Kaggle Score:** 0.12930 — Top 20% globally
- **Features:** 86 features including engineered (TotalSF, HouseAge, HasGarage)

---

## 🚀 Quick Start

### Test the live API:
```bash
curl -X POST "https://house-prices-api-13u5.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"GrLivArea": 1500, "OverallQual": 7, "TotalBsmtSF": 800, "GarageCars": 2, "YearBuilt": 2005, "TotalSF": 2300, "HouseAge": 21}'
```

### Response:
```json
{
  "predicted_price": 170822.04,
  "currency": "USD",
  "model": "LightGBM"
}
```

### Python:
```python
import requests

response = requests.post(
    "https://house-prices-api-13u5.onrender.com/predict",
    json={
        "GrLivArea": 1500,
        "OverallQual": 7,
        "TotalBsmtSF": 800,
        "GarageCars": 2,
        "YearBuilt": 2005,
        "TotalSF": 2300,
        "HouseAge": 21
    }
)
print(response.json())
# {'predicted_price': 170822.04, 'currency': 'USD', 'model': 'LightGBM'}
```

---

## 📥 Input Features

| Feature | Type | Description |
|---|---|---|
| GrLivArea | float | Above ground living area (sq ft) |
| OverallQual | int | Overall material and finish quality (1-10) |
| TotalBsmtSF | float | Total basement area (sq ft) |
| GarageCars | int | Garage capacity (number of cars) |
| YearBuilt | int | Year house was built |
| TotalSF | float | Total square footage (optional, default 0) |
| HouseAge | int | Age of house in years (optional, default 0) |

---

## 🏗️ Architecture

Request → FastAPI → Pydantic Validation → LightGBM Pipeline → Response

FastAPI (main.py)
├── GET /health → status check
└── POST /predict → house features → predicted price

Pipeline (lgb_final.pkl)
├── SimpleImputer → fills missing values with medians
├── OrdinalEncoder → encodes categorical features
├── StandardScaler → normalizes numeric features
└── LightGBM → predicts log(SalePrice)
→ np.expm1() → actual price


---

## 🐳 Run Locally with Docker

```bash
# clone repo
git clone https://github.com/Kavyabansa/ml-api.git
cd ml-api

# build image
docker build -t house-prices-api .

# run container
docker run -p 8000:8000 house-prices-api

# test
curl http://localhost:8000/health
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| Server | Uvicorn |
| ML Model | LightGBM |
| Validation | Pydantic |
| Containerization | Docker |
| Deployment | Render.com |
| Language | Python 3.11 |

---

## 📁 Project Structure


ml-api/
main.py ← FastAPI application
lgb_final.pkl ← trained LightGBM pipeline
requirements.txt ← Python dependencies
Dockerfile ← container blueprint
test.py ← local API testing
test_cloud.py ← cloud API testing
README.md ← this file


---

## 🔧 Local Development

```bash
# install dependencies
pip install -r requirements.txt

# run locally
python main.py

# API available at http://localhost:8000
# docs at http://localhost:8000/docs
```

---

## 📈 Sample Predictions

| House | GrLivArea | Quality | YearBuilt | Predicted Price |
|---|---|---|---|---|
| Modest | 1,200 | 5 | 1990 | ~$120,000 |
| Average | 1,500 | 7 | 2005 | $170,822 |
| Premium | 2,000 | 9 | 2015 | $233,327 |

---

## 👩‍💻 Author

**Kavya Bansal** — B.Tech CSE (AI), NIET Greater Noida

- 🔗 [GitHub](https://github.com/Kavyabansa/ml-journey)
- 📊 [Kaggle](https://www.kaggle.com/kavyasd/code)
- 💼 [LinkedIn](https://www.linkedin.com/in/kavya-bansal-746a65330)
