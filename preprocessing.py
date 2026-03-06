'''
Usage: 
python3 preprocessing.py

Person 4:
Refer to bottom of the file for saving as a NumPy for easy access
'''

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("cleaned_bc_data.csv")

# seperate features + target
X = df.drop(["Price", "Province"], axis=1)
y = df["Price"]

# we remove province cause theyre all BC anyway
categorical_cols = ["City", "Property Type"]
numerical_cols = [
    "Latitude",
    "Longitude",
    "Bedrooms",
    "Bathrooms",
    "Square Footage"
]

print("Dataset shape:", df.shape)
print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# preprocesing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# fit pp on training data
X_train_processed = preprocessor.fit_transform(X_train)

# apply pp to test data
X_test_processed = preprocessor.transform(X_test)

print("Training data shape:", X_train_processed.shape)
print("Test data shape:", X_test_processed.shape)

# optional: 
#
# if person 4 wants to load the processed data
# directly (instead of rerunning preprocessing), they can
# uncomment the lines below. thisll save the train/test
# datasets as NumPy arrays that can be loaded later
#
# example usage:
# X_train = np.load("X_train.npy")
# X_test = np.load("X_test.npy")
# y_train = np.load("y_train.npy")
# y_test = np.load("y_test.npy")

# import numpy as np
#
# np.save("X_train.npy", X_train_processed)
# np.save("X_test.npy", X_test_processed)
# np.save("y_train.npy", y_train)
# np.save("y_test.npy", y_test)