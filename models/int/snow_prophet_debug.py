from snow_prophet import snowpark_forecast
from snowflake.snowpark import Session
import json, joblib, sys, os, pandas as pd

def getsession():
    # udf(None, 'seasonal_data_test_week', 'ref', _TRIALS = 100)
    connection_parameters = json.load(open('/dbt/python/snow_session.json'))
    session = Session.builder.configs(connection_parameters).create()
    return session

def train():
    session = getsession()
    # get data from table HT_DEV.SNOWPARK.INT_SALES_DS_TRAIN in snowflake
    ref_df = session.table('HT_DEV.SNOWPARK.INT_SALES_DS_TRAIN').toPandas().copy()
    # get distinct categories list for field STORE_DEPT_PK
    categories = ref_df['STORE_DEPT_PK'].unique()
    print(categories)
    # loop through categories
    for category in categories:
        # filter df by category
        df = ref_df[ref_df['STORE_DEPT_PK'] == category]
        sf = snowpark_forecast(df, session, None)
        sf._label_fld = 'WEEKLY_SALES'
        sf._cat_fld = 'STORE_DEPT_PK'
        sf._ts_fld = 'WEEK_DATE'
        sf._holiday = 'US'
        sf._is_local = True
        sf.process()

def inference() -> str:
    session = getsession()
    filename = 'INT_SALES_DS_PREDICT'
    with open(os.path.join(filename), 'rb') as file:
        m = joblib.load(file)
    df = session.table('HT_DEV.SNOWPARK.INT_SALES_DS_EVALUATE').toPandas().copy()
    df['LABEL'] = df['WEEKLY_SALES']
    df.rename(columns={'WEEK_DATE': 'ds', 'LABEL': 'y'}, inplace=True)
    # convert ds to date
    df['ds'] = pd.to_datetime(df['ds'])
    # convert y to float
    df['y'] = df['y'].astype(float)

    df['floor'] = 0
    df['cap'] = 1000000
    output = m.predict(df)
    output['combined'] = f'{output["yhat"]}, {output["yhat_lower"]}, {output["yhat_upper"]}'
    # get only yhat, yhat_lower, yhat_upper
    # output = output[['yhat', 'yhat_lower', 'yhat_upper']]
    # get first row
    output = str(output[['yhat', 'yhat_lower', 'yhat_upper', 'trend']].iloc[0])
    # convert string to pandas series
    output = pd.Series(output)
    print(type(output))
    print('output: ', output)
    return output

if __name__ == "__main__":
    inference()