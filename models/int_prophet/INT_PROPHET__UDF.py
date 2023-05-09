from snowflake.snowpark.types import PandasSeries
import snowflake.snowpark.functions as F
import sys, os, json, pandas as pd, numpy as np, cloudpickle

def model(dbt, session):
    dbt.config(
        packages = ["cloudpickle", "pandas", "prophet", "snowflake-snowpark-python"]
    )
    ref_df = dbt.ref("INT_PROPHET__ML_TRAIN")
    # get all files from stage
    stage_models = session.sql("list @HT_PROPHET_MODELS").collect()
    stage_models = [x['name'] for x in stage_models]
    # import all files from stage
    for file in stage_models:
        session.add_import(f"@{file}")

    def read_file(filename):
        import_dir = sys._xoptions.get("snowflake_import_directory")
        with open(os.path.join(import_dir, filename), 'rb') as file:
            return cloudpickle.loads(file.read())
        
    #register UDF
    @F.udf(name = 'predict_sales', is_permanent = True, replace = True, stage_location = '@HT_PROPHET_MODELS')
    def predict_sales(ds: PandasSeries[dict]) -> PandasSeries[str]:
        # later we will input train data as JSON object
        # hance, we have to convert JSON object as pandas DF
        df = pd.io.json.json_normalize(ds)
        if df.empty:
            return pd.Series([])  # Return an empty Series if df is empty
        
        # batch prediction, by category
        # create empty column for prediction
        df['yhat'], df['yhat_lower'], df['yhat_upper'] = np.nan, np.nan, np.nan

        # get all categories
        categories = df['STORE_DEPT_PK'].unique()

        # loop through all categories
        for category in categories:
            filename = f'{category}__RETAIL_SALES'
            check = [x for x in stage_models if filename in x]
            filter_cond = df['STORE_DEPT_PK'] == category
            if check == []:
                df.loc[filter_cond, 'yhat'] = 0
                df.loc[filter_cond, 'yhat_lower'] = 0
                df.loc[filter_cond, 'yhat_upper'] = 0
                continue
            pipeline = read_file(filename)
            pred_df = pipeline.predict(df.loc[filter_cond, 'ds'].to_frame())
            df.loc[filter_cond, 'yhat'] = pred_df['yhat'].values
            df.loc[filter_cond, 'yhat_lower'] = pred_df['yhat_lower'].values
            df.loc[filter_cond, 'yhat_upper'] = pred_df['yhat_upper'].values

        # combine all yhat, yhat_lower, yhat_upper into one column, json object
        df['output'] = df.apply(lambda x: json.dumps({'yhat': x['yhat'], # prediction
                                                    'yhat_lower': x['yhat_lower'], # prediction lower interval
                                                    'yhat_upper': x['yhat_upper'] # prediction upper interval
                                                    }), axis=1)
        return df['output'].astype(str)

    return ref_df