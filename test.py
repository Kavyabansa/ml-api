import requests

# test health
health = requests.get("http://localhost:8000/health")
print(f"Health: {health.json()}")

# test prediction
house = {
    "GrLivArea": 1500,
    "OverallQual": 7,
    "TotalBsmtSF": 800,
    "GarageCars": 2,
    "YearBuilt": 2005,
    "TotalSF": 2300,
    "HouseAge": 21
}

response = requests.post("http://localhost:8000/predict", json=house)
print(f"Prediction: {response.json()}")