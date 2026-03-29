import os
import joblib
import pandas as pd


DEFAULT_MODEL_PATH = "random_forest.pkl"


def predict_price(sample_input, model_path=DEFAULT_MODEL_PATH):
    """
    Predicts a house price from a single sample input dictionary.

    Expected keys:
    - City
    - Property Type
    - Latitude
    - Longitude
    - Bedrooms
    - Bathrooms
    - Square Footage
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Run milestone2_models.py first."
        )

    model = joblib.load(model_path)
    sample_df = pd.DataFrame([sample_input])
    prediction = model.predict(sample_df)[0]
    return prediction


if __name__ == "__main__":
    sample_home = {
        "City": "Vancouver",
        "Property Type": "House",
        "Latitude": 49.2827,
        "Longitude": -123.1207,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Square Footage": 1800
    }

    try:
        predicted_price = predict_price(sample_home)
        print("\nSample Input:")
        for key, value in sample_home.items():
            print(f"  {key}: {value}")

        print(f"\nPredicted Price: ${predicted_price:,.2f}")

    except Exception as e:
        print(f"\nDemo prediction failed: {e}")