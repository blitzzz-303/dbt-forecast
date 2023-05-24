from snowflake.snowpark.types import PandasSeries
import snowflake.snowpark.functions as F
import sys, os, json, pandas as pd, numpy as np, cloudpickle

def model(dbt, session):
    dbt.config(
        packages = ["cloudpickle", "prophet", "snowflake-snowpark-python"]
    )
    ref_df = dbt.ref("INT_PROPHET__ML_TRAIN")
    # get all files from stage
    stage_models = session.sql("list @HT_PROPHET_MODELS").collect()
    stage_models = [x['name'] for x in stage_models]
    # import all files from stage
    for file in stage_models:
        session.add_import(f"@{file}")

    # chec Snowlfake import directory and get the model
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

        # loop through all categories
        for category in df['STORE_DEPT_PK'].unique():
            filename = f'{category}__RETAIL_SALES'

            # check if model exists
            check = [x for x in stage_models if filename in x]
            category_filter = df['STORE_DEPT_PK'] == category

            if check == []:
                df.loc[category_filter, ['yhat', 'yhat_lower', 'yhat_upper']] = np.nan, np.nan, np.nan
                continue
            
            try:
                # load model
                pipeline = read_file(filename)

                # predict
                pred_df = pipeline.predict(df.loc[category_filter, 'ds'].to_frame())

                # assign prediction to df
                df.loc[category_filter, 'yhat'] = pred_df['yhat'].values
                df.loc[category_filter, 'yhat_lower'] = pred_df['yhat_lower'].values
                df.loc[category_filter, 'yhat_upper'] = pred_df['yhat_upper'].values
            # capture error when a model is deleted from stage
            except:
                df.loc[category_filter, ['yhat', 'yhat_lower', 'yhat_upper']] = np.nan, np.nan, np.nan
                continue
        # combine all yhat, yhat_lower, yhat_upper into one column, json object
        df['output'] = df.apply(lambda x: json.dumps({'yhat': x['yhat'], # prediction
                                                    'yhat_lower': x['yhat_lower'], # prediction lower interval
                                                    'yhat_upper': x['yhat_upper'] # prediction upper interval
                                                    }), axis=1)
        return df['output'].astype(str)

    return ref_df