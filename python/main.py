from snow_prophet import batch_forecast
from snowflake.snowpark import Session
import json

def main():
    # udf(None, 'seasonal_data_test_week', 'ref', _TRIALS = 100)
    connection_parameters = json.load(open('/dbt/python/snow_session.json'))
    session = Session.builder.configs(connection_parameters).create()
    # configs = {'session':None, '_df_path':'/dbt/python/sales_dataset_train.csv', 
    # '_ts_fld':'WEEK_DATE', '_cat_fld':'STORE','_label_fld': 'WEEKLY_SALES'}
    
    # configs = {'session':session, 'tbl_source':'sales_data_sample_summary_train', 'tbl_target': 'sales_data_sample_summary_train_predict'}
    configs = {'session':session, "tbl_source": "HT_DEV.SNOWPARK.INT_SALES_DS_TRAIN",
                     "tbl_target": "HT_DEV.SNOWPARK.INT_SALES_DS_PREDICT",
                     "_ts_fld":"WEEK_DATE", "_cat_fld":"STORE_DEPT_PK", "_label_fld": "WEEKLY_SALES"}
    p = batch_forecast(**configs)

if __name__ == "__main__":
    main()