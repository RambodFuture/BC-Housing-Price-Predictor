import pandas as pd

df = pd.read_csv("./raw_data.csv")

bc_df = df[df["Province"] == "BC"]

drop_cols = [
"Garage","Parking","Basement","Exterior","Fireplace","Heating",
"Flooring","Roof","Waterfront","Sewer","Pool","Garden","Balcony", "Acreage"
]

bc_df = bc_df.drop(columns=drop_cols, errors="ignore")

bc_df = bc_df[(bc_df["Price"] >= 100000) & (bc_df["Price"] <= 5000000)]

bc_df = bc_df[(bc_df["Square Footage"] >= 200) & (bc_df["Square Footage"] <= 10000)]

bc_df = bc_df[(bc_df["Bedrooms"] >= 1) & (bc_df["Bedrooms"] <= 10)]
bc_df = bc_df[(bc_df["Bathrooms"] >= 1) & (bc_df["Bathrooms"] <= 10)]

print("Rows after cleaning:", len(bc_df))

bc_df.to_csv("cleaned_bc_data.csv", index=False)

