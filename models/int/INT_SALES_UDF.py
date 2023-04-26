from snowflake.snowpark.types import PandasSeries, PandasDataFrame

import snowflake.snowpark.functions as F
import joblib, sys, os, json, pandas as pd, numpy as np

def model(dbt, session):
    dbt.config(
        packages = ["joblib", "pandas", "prophet", "scikit-learn", "snowflake-snowpark-python"]
    )
    ref_df = dbt.ref("INT_SALES_TRAIN").to_pandas()
    # get all files from stage
    stage_models = session.sql("list @INT_SALES_DS_PREDICT").collect()
    stage_models = [x['name'] for x in stage_models]
    # import all files from stage
    for file in stage_models:
        session.add_import(f"@{file}")

    def read_file(filename):
        #where all imports located at
        import_dir = sys._xoptions.get("snowflake_import_directory")

        if import_dir:
            with open(os.path.join(import_dir, filename), 'rb') as file:
                m = joblib.load(file)
                return m
        
    #register UDF
    @F.udf(name = 'predict_sales', is_permanent = True, replace = True, stage_location = '@INT_SALES_DS_PREDICT')
    def predict_sales(ds: PandasSeries[dict]) -> PandasSeries[str]:
        # later we will input train data as JSON object
        # hance, we have to convert JSON object as pandas DF
        df = pd.io.json.json_normalize(ds)
        if df.empty:
            return pd.Series([])  # Return an empty Series if df is empty

        # rename field DS to ds and Y to y and convert to proper data type
        df.rename(columns={'DS': 'ds', 'Y': 'y'}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = 0

        # if floor and cap are not provided, Prophet will use 0 and infinity
        if 'FLOOR' in df.columns and 'CAP' in df.columns:
            df['floor'] = df['FLOOR'].astype(float)
            df['cap'] = df['CAP'].astype(float)
        else:
            df['floor'] = 0
            df['cap'] = 1000000

        def predict(_dict = {}):
            df = pd.io.json.json_normalize(_dict)
            category = df['STORE_DEPT_PK'][0]
            filename = f'{category}__INT_SALES_DS_PREDICT'
            pipeline = read_file(filename)
            pred = pipeline.predict(df)
            # calculate confidence level base on yhat_lower & yhat_upper
            pred['confidence'] = (pred['yhat_upper'] - pred['yhat_lower']) / pred['yhat']
            # combine all yhat, yhat_lower, yhat_upper into one column, json object
            return pred.apply(lambda x: json.dumps({'yhat': x['yhat'], 'yhat_lower': x['yhat_lower'], 'yhat_upper': x['yhat_upper'], 'category': category}), axis=1)
        df['output'] = df.apply(lambda x: predict(x.to_dict()), axis=1)
        return df['output'].astype(str)

    return ref_df