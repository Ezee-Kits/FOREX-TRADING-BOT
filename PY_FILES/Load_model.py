import joblib
import pandas as pd

# Load model bundle
bundle = joblib.load("CSV FILES/EURUSD_lgbm_bundle.pkl")
model = bundle["model"]
feature_columns = bundle["features"]

# Load backtest data
backtest_df = pd.read_csv("CSV FILES/BACKTEST_DATA.csv")

X_test = backtest_df[feature_columns]

y_pred = model.predict(X_test)
# print(y_pred)
proba = model.predict_proba(X_test)
# print(proba[:,0])


pred_df = pd.DataFrame()
pred_df['ACTUAL'] = backtest_df['target']
pred_df['PREDICTED'] = y_pred
pred_df['SAME'] = (pred_df['ACTUAL'] == pred_df['PREDICTED']).astype(int)
pred_df['LOW PROBA'] = proba[:,0]*100
pred_df['HIGH PROBA'] = proba[:,1]*100
print(pred_df.head(50).to_string( ))
print(pred_df['SAME'].to_list().count(1), 'of out', len(pred_df))
print(backtest_df['target'].value_counts(normalize=True))





