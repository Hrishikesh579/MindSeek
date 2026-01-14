import pandas as pd

# Load CSV
df = pd.read_csv("mhqa_cleaned.csv")

# Convert to JSON - array of objects
df.to_json("mhqa_cleaned.json", orient="records", indent=4)
