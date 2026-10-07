from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

# Load trained pipeline
model = joblib.load("lgb_final.pkl")

# Exact 86 features expected by the trained model
MODEL_FEATURES = [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
    'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
    'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
    'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
    'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
    'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
    'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'TotalSF', 'TotalBath',
    'HouseAge', 'RemodAge', 'HasPool', 'HasGarage', 'HasBasement'
]

app = FastAPI(
    title="House Prices Prediction API",
    description="Predicts house prices using LightGBM",
    version="1.0.0"
)


class HouseFeatures(BaseModel):
    GrLivArea: float
    OverallQual: int
    TotalBsmtSF: float
    GarageCars: int
    YearBuilt: int
    TotalSF: float = 0.0
    HouseAge: int = 0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "LightGBM House Prices"
    }


@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        # Create all 86 columns with NaN.
        # The trained pipeline's SimpleImputer will handle missing values.
        data = pd.DataFrame(
            np.nan,
            index=[0],
            columns=MODEL_FEATURES
        )

        # Put user's supplied values into the correct columns
        data.loc[0, "GrLivArea"] = features.GrLivArea
        data.loc[0, "OverallQual"] = features.OverallQual
        data.loc[0, "TotalBsmtSF"] = features.TotalBsmtSF
        data.loc[0, "GarageCars"] = features.GarageCars
        data.loc[0, "YearBuilt"] = features.YearBuilt
        data.loc[0, "TotalSF"] = features.TotalSF
        data.loc[0, "HouseAge"] = features.HouseAge

        # Additional logical values
        data.loc[0, "1stFlrSF"] = features.GrLivArea
        data.loc[0, "OverallCond"] = 5
        data.loc[0, "YearRemodAdd"] = features.YearBuilt

        # Prediction
        prediction = model.predict(data)

        # Model was trained on log(SalePrice)
        price = float(np.expm1(prediction[0]))

        return {
            "predicted_price": round(price, 2),
            "currency": "USD",
            "model": "LightGBM"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )