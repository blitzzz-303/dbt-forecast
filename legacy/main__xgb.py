from numpy import asarray
import pandas as pd
import optuna
import xgboost as xgb
import numpy as np
import sklearn
import json
import os
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from xgboost import XGBRegressor
from snowflake.snowpark import Session
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from prophet import Prophet

factors = {
    'dow': 48,
    'doy': 365,
    'dom': 12,
    'm1': 12,
    'h': 24,
    'wd': 48,
    'wkend': 48,
    'q': 4,
    'm%3': 4,
    'm%4': 3,
    'm6': 2,
    'y': 1,
    'wom': 52,
    'm': 12
}

def udf(session, tbl_name, LABEL = 'LABEL', MODE = 'PERF'):
    if session != None:
        df = get_df_from_sf(session, tbl_name)
    else:
        df = pd.read_csv('/sf/python/work_data.csv')
        df['MONTH'] = df['MONTH'].astype('datetime64')

    df_enhanced = generate_features(df, LABEL)
    # df_enhanced.drop(columns=['MONTH_doy'], inplace = True)
    global X_train, y_train, X_test, y_test
    X_train, y_train, X_test, y_test = tt_split(df_enhanced, LABEL)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50,  timeout=600)

    # trial = study.best_trial
    # print("  Value: {}".format(trial.value))
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

    reg = XGBRegressor(**study.best_params)
    # reg = get_XGB_MODEL(MODE)
    reg.fit(X_train, y_train)
    
    X_test['__PREDICT'] = reg.predict(X_test)
    df[f'{LABEL}__PREDICT'] = X_test['__PREDICT']
    df['XGB_GAIN'] = str(reg.get_booster().get_score(importance_type='weight'))
    important_features = reg.get_booster().get_score(importance_type='weight')

    df_enhanced['MONTH'] = df['MONTH']

    df_unique = df_enhanced.nunique(axis=0).to_frame().reset_index()
    df_unique['feature'] = df_unique['index']
    df_unique.drop(columns='index', inplace=True)
    df_unique['0'] = 1 / df_unique[0]
    df_unique['gain'] = df_unique.apply(lambda x: float(important_features.get(x['feature'], '0')), axis=1)
    df_unique['x'] = df_unique.apply(lambda x: factors.get(x['feature'].split('__')[-1]), axis=1)
    df_unique['factors'] = df_unique.apply(lambda x: float(factors.get(x['feature'].split('__')[-1], 1)), axis=1)
  
    df_unique['gain_t'] = df_unique['0'] * df_unique['gain'] / df_unique['factors']
    total = df_unique.sum()['gain_t']

    df_unique['gain_t_perc'] = df_unique['gain_t'] / total
    df_unique.sort_values(by=['gain_t'], inplace=True, ascending = False)
    df_unique = df_unique[df_unique['gain_t_perc'] > 0.2]
    high_feature = df_unique['feature'].values.tolist()
    df_agg = df_enhanced.groupby(by=['MONTH__dow', 'MONTH__dom']).agg(
                                                        MONTH=('MONTH', np.max),
                                                        LABEL=('LABEL', np.sum))[['MONTH', 'LABEL']].sort_values(by=['MONTH'])
    print(df_agg)
    train, test = df_agg[df_agg['MONTH'] < '2022-12-03'], df_agg[df_agg['MONTH'] >= '2022-12-03']
    print(test)
    predictions = prophet_features(train, test, 'MONTH', 'LABEL')
    # print(df[f'{LABEL}__PREDICT'])
    print(predictions)
    print(predictions['multiplicative_terms'])
    df[f'{LABEL}__PREDICT'] = df[f'{LABEL}__PREDICT'] * 1.22
    print(df[f'{LABEL}__PREDICT'])
    if session != None:
        return_df = session.create_dataframe(df) 
        return_df.write.mode("overwrite").save_as_table(tbl_name + '__PREDICT')
    return 'finished'

def prophet_features(train, test, DT, LABEL):
    train.rename(columns={DT: 'ds', LABEL: 'y'}, inplace=True)
    test.rename(columns={DT: 'ds', LABEL: 'y'}, inplace=True)
    m = Prophet()
    m.fit(train)
    predictions_train = m.predict(train.drop('y', axis=1))
    predictions_test = m.predict(test.drop('y', axis=1))
    predictions = pd.concat([predictions_train, predictions_test], axis=0)
    return predictions

def get_df_from_sf(session, tbl_name):
    df = session.table(tbl_name).to_pandas().copy()
    df.sort_values(by=[df.select_dtypes(include=['datetime64']).columns[0]], inplace=True)
    df.reset_index(drop = True, inplace = True)
    return df

def get_XGB_MODEL(MODE):
    if MODE == 'PERF':
        return XGBRegressor()

def tt_split(df, LABEL):
    df_train, df_test = df[df[LABEL].notnull()], df[df[LABEL].isnull()]
    X_train, y_train = split_feature_label(df_train, LABEL)
    X_test, y_test = split_feature_label(df_test, LABEL)
    return X_train, y_train, X_test, y_test

def encode(df, LABEL):
    df_encode = df.drop(columns= [LABEL])
    ndt_cols = df_encode.select_dtypes(exclude=['datetime64']).columns
    # df[ndt_cols] = df[ndt_cols].astype('category')
    df_encode = pd.get_dummies(data=df_encode, columns=ndt_cols)
    df[df_encode.columns] = df_encode[df_encode.columns]
    df.drop(columns= ndt_cols, inplace=True)
    return df

def generate_features(df, LABEL):
    df_enhanced = encode(df, LABEL)
    df_enhanced = generate_ts_features(df_enhanced, LABEL)
    return df_enhanced

def generate_ts_features(df, LABEL):
    """
    Generate time series features from datetime index
    """
    df_enhanced = df.copy()
    dt_cols = df_enhanced.select_dtypes(include=['datetime64']).columns
    for dt_col in dt_cols:
        df_enhanced[f'{dt_col}__h'] = df[dt_col].dt.hour
        df_enhanced[f'{dt_col}__dow'] = df[dt_col].dt.dayofweek
        df_enhanced[f'{dt_col}__wd'] = df[dt_col].dt.dayofweek < 4
        df_enhanced[f'{dt_col}__wkend'] = df[dt_col].dt.dayofweek > 4
        df_enhanced[f'{dt_col}__q'] = df[dt_col].dt.quarter
        df_enhanced[f'{dt_col}__m'] = df[dt_col].dt.month
        df_enhanced[f'{dt_col}__%3'] = df[dt_col].dt.month % 3
        df_enhanced[f'{dt_col}__%4'] = df[dt_col].dt.month % 4
        df_enhanced[f'{dt_col}__%6'] = df[dt_col].dt.month % 6
        df_enhanced[f'{dt_col}__y'] = df[dt_col].dt.year
        df_enhanced[f'{dt_col}__doy'] = df[dt_col].dt.dayofyear
        df_enhanced[f'{dt_col}__dom'] = df[dt_col].dt.day
        df_enhanced[f'{dt_col}__wom'] = df[dt_col].apply(lambda d: (d.day - 1) // 7 + 1)
        df_enhanced = df_enhanced.drop(columns= [dt_col])
    return df_enhanced

def split_feature_label(df, LABEL):
    y = df[LABEL]
    X = df.drop(columns= [LABEL])
    return X, y

def convert_date_field_to_str(df):
    dt_cols = df.select_dtypes(include=['datetime64']).columns
    for dt_col in dt_cols:
        df[dt_col] = df[dt_col].astype('str')
    return df

def revert_ts(df_enhanced, df):
    dt_cols = df.select_dtypes(include=['datetime64']).columns
    for dt_col in dt_cols:
        df_enhanced[dt_col] = df[dt_col]
    return convert_date_field_to_str(df_enhanced)

def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, shuffle=True,  random_state=42)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "tree_method": "exact",
        "booster": trial.suggest_categorical("booster", ["gbtree"]),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
        "nrounds": trial.suggest_float('nrounds', 5000, 20000, step=5000),
        "base_margin": trial.suggest_float('base_margin', 1.2, 1.2)
    }

    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=3)
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 3, 9, step=3)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy

# def main():
#     connection_parameters = {
#     "account": "infinitelambda.eu-west-1",
#     "user": "hung",
#     "password": "Avenged7Fold@1990",
#     "role": "HT_DEV_PII_RW",  # optional
#     "warehouse": "HT_PII_DEMO_WH",  # optional
#     "database": "HT_DEV",  # optional
#     "schema": "HT_SNOWPARK",  # optional
#   }  
#     session = Session.builder.configs(connection_parameters).create()
#     # udf(session, 'int_sales_sample_train', 'SALES')
#     udf(session, 'seasonal_data_test_week')
    
def main():
    udf(None, 'seasonal_data_test_week')

if __name__ == "__main__":
    main()