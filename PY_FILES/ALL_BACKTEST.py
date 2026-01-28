import joblib
import pandas as pd
from func import apply_features,SYMBOL,create_targets,trade_backtest,analyze_results








backtest_data = pd.read_csv(f"CSV_FILES/MT5_5M_BT_{SYMBOL}_Dataset.csv")
backtest_df = apply_features(backtest_data)
backtest_df = create_targets(backtest_df)
backtest_df.dropna(inplace=True)



pred_df = pd.DataFrame()
all_target = ['T_5M','T_10M','T_15M','T_20M','T_30M']


# # for target in all_target:
main_res = []
for target in all_target:
    bundle = joblib.load(f"ALL_MODELS/{SYMBOL}_lgbm_{target}.pkl")
    model = bundle["model"]
    feature_columns = bundle["features"]

    results = trade_backtest(df=backtest_df, model=model, feature_cols=feature_columns,threshold=55   )
    print(f"Backtest Results for target {target}:")
    analysis = analyze_results(results)
    main_res.append(analysis)
    print(analysis)
print(main_res)


# X_test = backtest_df[feature_columns]
# y_pred = model.predict(X_test)
# proba = model.predict_proba(X_test)
# pred_df[f'{target}_DOWN_PROB'] = (proba[:, 0] * 100).round(2)
# pred_df[f'{target}_UP_PROB']   = (proba[:, 1] * 100).round(2)
# pred_df['ACTUAL'] = backtest_df[target].to_list()
# pred_df[f'{target}_pred'] = (proba>=55).astype(int)
# pred_df['SAME'] = (pred_df['ACTUAL'] == pred_df[f'{target}_pred']).astype(int) 
# print(pred_df)
# print(pred_df['SAME'].to_list().count(1), 'of out', len(pred_df))
# print(backtest_df[target].value_counts(normalize=True))

