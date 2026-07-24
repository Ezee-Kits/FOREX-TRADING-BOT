import joblib
import pandas as pd

file_path = "ALL_MODELS/HL_MAIN_THL_3H_10M_EURUSD_model.pkl"

bundle = joblib.load(file_path)

# Extract parts
model = bundle["model"]
features = bundle["features"]
importance = bundle["feature_importance"]

# Put into readable table
imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort most important first
imp_df = imp_df.sort_values(by="Importance", ascending=False)

print(imp_df.to_string(index=False))
