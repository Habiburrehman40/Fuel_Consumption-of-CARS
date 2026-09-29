"""
Model train karo aur save karo.
Usage:  python train_model.py FuelConsumption.xlsx     (ya .csv)
Output: model.pkl  +  options.json  (website ke dropdowns ke liye)
"""
import json
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "FuelConsumption.xlsx"

# ---- 1. Data load ----
if DATA_FILE.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(DATA_FILE)
else:
    df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
df = df.dropna()

# ---- 2. Features / target (aapke code jaisa) ----
X = df.drop(columns=["COEMISSIONS"])
y = df["COEMISSIONS"]

categorical_features = list(X.select_dtypes(include=["object"]).columns)
numerical_features = list(X.select_dtypes(exclude=["object"]).columns)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", "passthrough", numerical_features),
    ]
)

model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = {
    "mse": float(mean_squared_error(y_test, y_pred)),
    "mae": float(mean_absolute_error(y_test, y_pred)),
    "r2": float(r2_score(y_test, y_pred)),
}
print("Model Results:", metrics)

# ---- 3. Model save ----
joblib.dump(model, "model.pkl")

# ---- 4. Website dropdown options ----
latest = df.sort_values("Year").groupby(["MAKE", "MODEL"]).tail(1)
models_by_make = {
    make: sorted(grp["MODEL"].astype(str).unique().tolist())
    for make, grp in df.groupby("MAKE")
}
defaults = {}
for _, r in latest.iterrows():
    defaults[f"{r['MAKE']}||{r['MODEL']}"] = {
        "VEHICLE CLASS": str(r["VEHICLE CLASS"]),
        "ENGINE SIZE": float(r["ENGINE SIZE"]),
        "CYLINDERS": int(r["CYLINDERS"]),
        "TRANSMISSION": str(r["TRANSMISSION"]),
        "FUEL": str(r["FUEL"]),
        "FUEL CONSUMPTION": float(r["FUEL CONSUMPTION"]),
    }

options = {
    "features": list(X.columns),
    "numeric": numerical_features,
    "year_min": int(df["Year"].min()),
    "year_max": int(df["Year"].max()),
    "makes": sorted(models_by_make.keys()),
    "models_by_make": models_by_make,
    "classes": sorted(df["VEHICLE CLASS"].astype(str).unique().tolist()),
    "transmissions": sorted(df["TRANSMISSION"].astype(str).unique().tolist()),
    "fuels": sorted(df["FUEL"].astype(str).unique().tolist()),
    "cylinders": sorted(int(c) for c in df["CYLINDERS"].unique()),
    "defaults": defaults,
    "metrics": metrics,
}
with open("options.json", "w") as f:
    json.dump(options, f)

print("Saved: model.pkl, options.json")
