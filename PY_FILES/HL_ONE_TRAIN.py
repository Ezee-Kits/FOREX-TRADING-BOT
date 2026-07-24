
import joblib
import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
from func import apply_features
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score




# 'AUDCAD' =1H-59-0.585,AUDJPY = 1H-56-0.63,'AUDUSD' = 2H-76-0.528, BTCUSD = NONE,CADJPY = 2H-64-0.518, ETHUSD = NONE,EURJPY = NONE, 
#  "EURUSD" = 1H-57-0.61,GBPJPY = NONE,
#  "GBPUSD"=1H-61-0.63, NZDCAD = 1H-68-0.65, 'NZDUSD' = 1H-67-0.655,'USDCAD'=1H-58-0.62, 'USDCHF' = NONE, "USDJPY"=1H-57-0.56
#  "USDSEK" =1H-101-59 , 'XAGUSD'=NONE, "XAUUSD"=2H-0.52


SYMBOL = 'AUDCAD'
CD_TIME = '10M'
print(F'CURRENTLY RUNNING SYMBOL IS {SYMBOL}')
# ==========================================================
# LOAD DATA
# ==========================================================
# data = pd.read_csv(f'/content/FOREX TRADING/MT5_{CD_TIME}_{SYMBOL}_Exchange_Rate_Dataset.csv')
data = pd.read_csv(f'CSV_FILES/MT5_10M_BT_{SYMBOL}_Exchange_Rate_Dataset.csv') 

#===========================================================================
data['Date'] = pd.to_datetime(data['Date'])
BAD_HOURS = [21, 22]
data = data[~data['Date'].dt.hour.isin(BAD_HOURS)].reset_index(drop=True)
print('DATA FILE LENGTH BEFORE FILTERING :', len(data))
#===========================================================================

df = apply_features(data)
print('DATA FILE LENGTH AFTER APPLYING FEATURES :', len(df))
df.dropna(inplace=True)
print('DATA FILE LENGTH AFTER DROPNA :', len(df))

df.reset_index(drop=True, inplace=True)
print('COLUMNS:', len(df.columns))
print('ROWS:', len(df))


# ==========================================================
# PARAMETERS
# ==========================================================
tp_mult = 1.75
sl_mult = 1.75

# TARGET_HORIZONS = [1, 2, 3, 4, 6]
# horizon_names = ['THL_5M','THL_10M','THL_15M','THL_20M','THL_30M']

TARGET_HORIZONS = [12]
horizon_names = ['THL_2H']

labels_dict = {name: [] for name in horizon_names}

# ==========================================================
# STEP 2: FINAL LABEL GENERATION (SINGLE PASS, CLEAN LOGIC)
# ==========================================================
for i in range(len(df) - max(TARGET_HORIZONS) - 1):

    entry = df['Close'].iloc[i]
    atr = df['ATR'].iloc[i]

    TP = entry + tp_mult * atr
    SL = entry - sl_mult * atr

    buffer = 0.1 * atr

    for idx, horizon in enumerate(TARGET_HORIZONS):

        highs = df['High'].iloc[i+1:i+1+horizon].values
        lows  = df['Low'].iloc[i+1:i+1+horizon].values

        label = -1

        for h in range(horizon):
            hit_tp = highs[h] > TP + buffer
            hit_sl = lows[h] < SL - buffer

            if hit_tp and hit_sl:
                break       # keep label = -1 (ambiguous)

            elif hit_tp:
                label = 1
                break

            elif hit_sl:
                label = 0
                break

        labels_dict[horizon_names[idx]].append(label)



# Trim df to match label length
df = df.iloc[:len(df) - max(TARGET_HORIZONS) - 1].copy()

for name in horizon_names:
    df[name] = labels_dict[name]

print("Label Distribution:")
for t in horizon_names:
    print(t, df[t].value_counts(normalize=True))


# ==========================================================
# SETTINGS
# ==========================================================
N_SPLITS = 5
N_TRIALS = 30

# ==========================================================
# EVALUATION FUNCTION (TIME SERIES SAFE)
# ==========================================================
def evaluate_model(X, y, params):

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []

    for train_idx, val_idx in tscv.split(X):

        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

        classes = np.unique(y_train)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )

        class_weight_dict = {
            classes[i]: class_weights[i]
            for i in range(len(classes))
        }

        params_local = params.copy()
        params_local["class_weight"] = class_weight_dict

        model = LGBMClassifier(**params_local)

        model.fit(
            X_train,
            y_train,

            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",

            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(0)  # silent
            ])

        preds = model.predict_proba(X_valid,num_iteration=model.best_iteration_)[:, 1]

        auc = roc_auc_score(y_valid, preds)
        scores.append(auc)

    return np.mean(scores)


def remove_correlated_features(X, threshold=0.9):

    split_idx = int(len(X) * 0.7)
    X_train = X.iloc[:split_idx]

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f" \n 🧹 Dropping {len(to_drop)} correlated features")
    print(f"Features Before Corr Filter: {X.shape[1]}")
    print(f"Features After Corr Filter: {X.shape[1] - len(to_drop)} \n")
    return X.drop(columns=to_drop)



for target in horizon_names:

    print(f"\n==============================")
    print(f"🚀 FULL FEATURE TRAINING: {target}")
    print(f"==============================")

    # Prevent leakage
    drop_targets = [t for t in horizon_names if t != target]
    df_train_target = df.drop(columns=drop_targets)
    df_train_target = df_train_target[df_train_target[target] != -1].copy()
    df_train_target.reset_index(drop=True, inplace=True)

    X_full = df_train_target.drop(columns=[target])
    y = df_train_target[target]

    # Handle class imbalance
    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    print(y.value_counts(normalize=True))

    print("Class Weights:", class_weight_dict)
    print('LEN X AFTER PREPROCESSING : ',len(X_full))
    print(y.value_counts(normalize=True))

    # ======== REMOVING CORRELATED FEATURES FROM [X] ===========
    X = remove_correlated_features(X_full, threshold=0.9)
    print(X)



    # # ======================================================
    # # OPTUNA TUNING (FULL FEATURES)
    # # ======================================================
    # def objective(trial):

    #     params = {
    #         "objective": "binary",
    #         "metric": "auc",
    #         "n_estimators": trial.suggest_int("n_estimators", 200, 900),
    #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
    #         "num_leaves": trial.suggest_int("num_leaves", 20, 300),
    #         "max_depth": trial.suggest_int("max_depth", 3, 12),
    #         "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    #         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    #         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    #         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    #         "random_state": 42,
    #         "n_jobs": -1,
    #         "class_weight": class_weight_dict
    #     }

    #     return evaluate_model(X, y, params)

    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=N_TRIALS)

    # print("🏆 BEST AUC:", study.best_value)
    # print("🏆 BEST PARAMS:", study.best_params)

    # # ======================================================
    # # TRAIN FINAL MODEL
    # # ======================================================
    # final_model = LGBMClassifier(
    #     **study.best_params,
    #     objective="binary",
    #     random_state=42,
    #     n_jobs=-1,
    #     class_weight=class_weight_dict
    # )

    # split = int(len(X) * 0.9)

    # X_train, X_val = X.iloc[:split], X.iloc[split:]
    # y_train, y_val = y.iloc[:split], y.iloc[split:]

    # final_model.fit(
    #     X_train, y_train,
    #     eval_set=[(X_val, y_val)],
    #     eval_metric="auc",
    #     callbacks=[early_stopping(50), log_evaluation(0)]
    # )

    # # Validation score
    # val_preds = final_model.predict_proba(
    #     X_val, num_iteration=final_model.best_iteration_)[:, 1]
    # val_auc = roc_auc_score(y_val, val_preds)

    # # Save
    # joblib.dump(
    #     {
    #         "model": final_model,
    #         "features": X.columns.tolist(),
    #         "target": target,
    #         "params": study.best_params,
    #         "best_iteration": final_model.best_iteration_,
    #         "val_auc": val_auc,
    #         "feature_importance": final_model.feature_importances_.tolist()
    #     },
    #     f"/content/FOREX TRADING/HL_MAIN_{target}_{CD_TIME}_{SYMBOL}_model.pkl"
    # )

    # print(f"💾 FULL MODEL SAVED: {target} | AUC: {val_auc:.4f}")
