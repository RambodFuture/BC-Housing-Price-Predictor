# BC-Housing-Price-Predictor


# Preprocessing

The script expects the cleaned dataset: `cleaned_bc_data.csv`.

### How to Run
`python3 preprocessing.py`

### Processing Steps
1. Load the cleaned dataset using **pandas**
2. Seperate the dataset into:
    * Features(X)
    * Target variable (y): `Price`
3. Remove `Province` column since all entries belong to BC
4. Identify feature types:
    * Categorical: `City`, `Property Type`
    * Numerical: `Latitude`, `Longitude`, `Bedrooms`, `Bathrooms`, `Square Footage`
5. Split dataset into 80% training and 20% testing data.
6. Building a preprocessing pipeline using **ColumnTransformer**
7. Fit the preprocessing pipeline on the training data only to avoid data leakage
8. Transform both training and test sets

### Output
The script produces processed datasets saved as NumPy arrays:
```
X_train.npy
X_test.npy
y_train.npy
y_test.npy
```

# Regression Model
Make sure the .npy files have been created from the Preprocessing step. Run the model by running the following command:

python3 linear_regression.py 

This will train the linear regression model and print evaluation metrics given the test set (MSE, RMSE, MAE, R^2)
