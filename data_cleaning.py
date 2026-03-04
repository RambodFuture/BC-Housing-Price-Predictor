import pandas as pd

df = pd.read_csv("./raw_data.csv")

bc_df = df[df["Province"] == "British Columbia"]

drop_cols = [
"Garage","Parking","Basement","Exterior","Fireplace","Heating",
"Flooring","Roof","Waterfront","Sewer","Pool","Garden","Balcony", "Acreage"
]

bc_df = bc_df.drop(columns=drop_cols, errors="ignore")

bc_df.to_csv("cleaned_bc_data.csv", index=False)
