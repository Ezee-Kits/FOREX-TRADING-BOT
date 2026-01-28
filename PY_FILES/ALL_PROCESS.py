
import ta
import joblib
import numpy as np
import pandas as pd
import mplfinance as mpf
from lightgbm import LGBMClassifier
from func import apply_features,create_targets,SYMBOL









data = pd.read_csv('CSV_FILES/MT5_5M_EURUSD_Exchange_Rate_Dataset.csv') 
df = apply_features(data)
df = create_targets(df)
df.dropna(inplace=True)

all_target = ['T_5M','T_10M','T_15M','T_20M','T_30M']
train_df = df.copy()
X_train = train_df.drop(columns=all_target)

for target in all_target:
    y_train = train_df[target]
    model = LGBMClassifier(n_estimators=200,random_state=42)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    feature_names = X_train.columns.to_list()
    sort_indx = np.argsort(importance)[::-1]

    top_76_indx = sort_indx[:76]
    top76_features = [feature_names[i] for i in top_76_indx]
    print(f'TOP 76 FEATURES : ',top76_features)

    # Reduce training data
    X_train_top76 = X_train[top76_features]
    model_top76 = LGBMClassifier(n_estimators=200,random_state=42)
    model_top76.fit(X_train_top76, y_train)

    joblib.dump({"model": model_top76,"features": top76_features}, f"ALL_MODELS/{SYMBOL}_lgbm_{target}.pkl")
    print(f'Model for {target} trained and saved.')
    print('-------------------------------------')