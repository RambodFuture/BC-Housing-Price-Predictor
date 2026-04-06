# Housing Price Predictor (BC Real Estate)
This project implements an end to end machine learning pipeline to predict housing prices in British Columbia using structured real estate data. The system covers data cleaning, preprocessing, model training, hyperparameter tuning, evaluation, and deployment through a prediction interface.

## Overview
The objective of this project is to build and compare multiple regression models to estimate property prices based on features such as location, property type, and size. The pipeline is designed to be reproducible and scalable, with a focus on proper data handling and model evaluation.

Key components include:

- Data cleaning and filtering
- Feature preprocessing using scaling and encoding
- Training multiple regression models
- Hyperparameter tuning using grid search
- Model evaluation with metrics and visualizations
- A demo script for real input prediction

## Project Structure
```
├── data_cleaning.py        # Cleans and filters raw dataset
├── preprocessing.py        # Train test split and feature processing
├── milestone2_models.py    # Trains models and saves them as .pkl
├── tuning.py               # Hyperparameter tuning with GridSearchCV
├── evaluation.py           # Evaluation metrics and visualizations
├── demo.py                 # Predicts price for a sample input
├── run_pipeline.py         # Runs the full pipeline
├── linear_regression.py    # Baseline model using NumPy inputs
├── raw_data.csv            # Original dataset
├── cleaned_bc_data.csv     # Cleaned dataset
├── *.pkl                   # Saved trained models
├── *.npy                   # Processed datasets
└── plots/                  # Generated plots
```

## Installation
Make sure Python 3.10 or later is installed.

Install dependencies:
```
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

## Usage

## Run the pipeline
```
python run_pipeline.py
```
This script will:

1. Train models if saved files are not found
2. Run evaluation and generate plots
3. Execute a demo prediction

## Run each step manually
**Data Cleaning**
```
python data_cleaning.py
```
Filters the dataset to British Columbia properties and removes outliers.

**Preprocessing**
```
python preprocessing.py
```
- Splits data into training and test sets
- Applies scaling to numerical features
- Applies one hot encoding to categorical features
- Saves processed arrays and preprocessing pipeline

**Model training**
```
python milestone2_models.py
```
Trains the following models:
- Decision Tree
- Random Forest
- KNN
- Ridge Regression
- Lasso Regression

Each model is saved as a ```.pkl``` file.

**Hyperparameter tuning**
```
python tuning.py
```
Runs grid search for each model and saves the best performing versions.

**Evaluation**
```
python evaluation.py
```
Generates:
- Model comparison table
- Metric plots for MAE, RMSE, and R²
- Actual versus predicted plots
- Residual analysis
- Feature importance plots
- Partial dependence plots

All outputs are saved in the ```plots``` directory.

**Demo prediction**
```
python demo.py
```

Example input:

```
{
    "City": "Vancouver",
    "Property Type": "House",
    "Latitude": 49.2827,
    "Longitude": -123.1207,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Square Footage": 1800
}
```
The model returns a predicted price using the trained pipeline.

## Models
The following models are implemented and compared:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- K Nearest Neighbors

## Evaluation Metrics

Model performance is evaluated using:

- Mean Absolute Error
- Root Mean Squared Error
- R² Score

Additional analysis includes residual distributions, outlier detection, and feature importance.

## Key Design Choices

- A pipeline based approach ensures consistent preprocessing and prediction
- Data is split before preprocessing to prevent leakage
- Multiple models are trained to allow comparison
- Saved pipelines allow prediction directly from raw input
- A single orchestration script simplifies execution

## Notes
- Ensure ```raw_data.csv``` is present before running the pipeline
- Run model training before using the demo script if ```.pkl``` files are missing
- Generated plots are stored in the ```plots``` directory

## Summary
This project demonstrates a complete machine learning workflow from raw data to deployment ready prediction. It emphasizes reproducibility, proper evaluation, and practical usability through a structured pipeline design.










