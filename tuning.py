import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline

# Load processed data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Load preprocessor
preprocessor = joblib.load("preprocessor.pkl")

print("Loaded preprocessed data and pipeline.\n")

def run_grid_search(model_name, model, param_grid):
    print(f"Tuning {model_name}...")

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print(f"\nBest params for {model_name}: {grid.best_params_}")
    print(f"Best RMSE: {-grid.best_score_:.4f}\n")

    joblib.dump(grid.best_estimator_, f"best_{model_name.lower().replace(' ', '_')}.pkl")
    print(f"Saved: best_{model_name.lower().replace(' ', '_')}.pkl\n")

    return grid

# Parameter grids
dt_params = {
    "model__max_depth": [5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

knn_params = {
    "model__n_neighbors": [3, 5, 7, 9],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2]
}

ridge_params = {
    "model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
}

lasso_params = {
    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "model__max_iter": [5000]
}

# Run tuning
run_grid_search("Decision Tree", DecisionTreeRegressor(), dt_params)
run_grid_search("Random Forest", RandomForestRegressor(), rf_params)
run_grid_search("KNN", KNeighborsRegressor(), knn_params)
run_grid_search("Ridge", Ridge(), ridge_params)
run_grid_search("Lasso", Lasso(), lasso_params)

print("Hyperparameter tuning complete.")
