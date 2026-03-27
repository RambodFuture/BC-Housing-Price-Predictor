import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso

# Import preprocessing pipeline + data
from preprocessing import preprocessor, X_train, X_test, y_train, y_test

# -----------------------------
# Decision Tree
# -----------------------------
dt_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", DecisionTreeRegressor())
])
dt_model.fit(X_train, y_train)

# -----------------------------
# Random Forest
# -----------------------------
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor())
])
rf_model.fit(X_train, y_train)

# -----------------------------
# KNN
# -----------------------------
knn_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", KNeighborsRegressor())
])
knn_model.fit(X_train, y_train)

# -----------------------------
# Ridge Regression
# -----------------------------
ridge_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", Ridge())
])
ridge_model.fit(X_train, y_train)

# -----------------------------
# Lasso Regression
# -----------------------------
lasso_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", Lasso())
])
lasso_model.fit(X_train, y_train)

# Save models for Member 2 + 3
import joblib
joblib.dump(dt_model, "decision_tree.pkl")
joblib.dump(rf_model, "random_forest.pkl")
joblib.dump(knn_model, "knn.pkl")
joblib.dump(ridge_model, "ridge.pkl")
joblib.dump(lasso_model, "lasso.pkl")
