
import ta
import joblib
import numpy as np
import pandas as pd
import mplfinance as mpf
from lightgbm import LGBMClassifier
from main import apply_features









data = pd.read_csv('CSV FILES/MT5_EURUSD_Exchange_Rate_Dataset.csv')
df = apply_features(data)
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)


train_df = df.iloc[:41087, :].copy()  # Adjusted to 41087 rows for training/testing split
backtest_df = df.iloc[41087:, :].copy()  # Separate backtest dataset
backtest_df.to_csv('CSV FILES/BACKTEST_DATA.csv')

X_train = train_df.drop(columns=['target'])
y_train = train_df['target']


model = LGBMClassifier()
model.fit(X_train, y_train)
importance = model.feature_importances_
feature_names = X_train.columns.to_list()
sort_indx = np.argsort(importance)[::-1]

top_90_indx = sort_indx[:76 ]
top90_features = [feature_names[i] for i in top_90_indx]
print(f'TOP 76 FEATURES : ',top90_features)

# Reduce training data
X_train_top90 = X_train[top90_features]
model_top90 = LGBMClassifier()
model_top90.fit(X_train_top90, y_train)

joblib.dump({"model": model_top90,"features": top90_features}, "CSV FILES/EURUSD_lgbm_bundle.pkl")







# fex_list = []
# for fex_indx in range(2,100):
#     top_90_indx = sort_indx[:fex_indx]
#     top90_features = [feature_names[i] for i in top_90_indx]
#     print(f'TOP {fex_indx} FEATURES : ',top90_features)

#     # Reduce training data
#     X_train_top90 = X_train[top90_features]
#     model_top90 = LGBMClassifier()
#     model_top90.fit(X_train_top90, y_train)

#     X_test = backtest_df[top90_features]
#     y_pred = model_top90.predict(X_test)
#     print(y_pred)

#     pred_df = pd.DataFrame()
#     pred_df['ACTUAL'] = backtest_df['target']
#     pred_df['PREDICTED'] = y_pred
#     pred_df['SAME'] = (pred_df['ACTUAL'] == pred_df['PREDICTED']).astype(int)
#     print(pred_df.head(50).to_string( ))
#     print(pred_df['SAME'].to_list().count(1), 'of out', len(pred_df))
#     print(backtest_df['target'].value_counts(normalize=True))
#     fex_list.append(pred_df['SAME'].to_list().count(1))

#     print(f'MAXIMUN VALUE :({max(fex_list)}) WITH NUMBER :( {fex_list.index(max(fex_list))} )FEATURES')
    
#MAXIMUN VALUE :(3766) WITH NUMBER :( 61 )FEATURES

# print(df)

