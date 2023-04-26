import pandas as pd
import multiprocessing, joblib, psutil, datetime, numpy as np
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from snowflake.snowpark import types as T

output_cols = ['category', 'RMAE', 'MAPE',
              'params', 'training_time_second',
              'max_workers', 'memory', 'train_at']

class snowpark_forecast():
    _label_fld = 'y'
    _params_FLD = 'PARAMS'
    _holiday_FLD = 'COUNTRY'
    _category = 'default'
    _holiday = None
    _is_local = False
    RMAE_score = 0
    MAPE_score = 0
    best_params = None
    metadata = None
    df = None
    _trials = -1
    _max_workers = multiprocessing.cpu_count()
    _tt_split = 0.33
    _CHANGEPOINT_PRIOR_SCALE = [0.05, 0.5]
    _CHANGEPOINT_RANGE = [0.8]
    _SEASONALITY_PRIOR_SCALE = [10]
    _cap_quantile = .75
    _floor_quantile = .25

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def detect_daily_seasonality(s):
        df = s.df.copy()
        # detect daily seasonality
        df['delta_days'] = df['ds'].diff().dt.days
        # get the most frequent delta days
        delta_days = df['delta_days'].value_counts().index[0]
        # if the most frequent delta days is between 1 and 3, then daily seasonality is probably present
        s.daily_seasonality = [True, False] if delta_days > 1 and delta_days < 3 else [False]

    def detect_weekly_seasonality(s):
        df = s.df.copy()
        # detect weekly seasonality
        df['delta_days'] = df['ds'].diff().dt.days
        # get the most frequent delta days
        delta_days = df['delta_days'].value_counts().index[0]
        # if the most frequent delta days is between 1 and 3, then weekly seasonality is probably present
        s.weekly_seasonality = [True, False] if delta_days > 5 and delta_days < 9 else [False]

    def process(s):
        s.start_time = datetime.datetime.now()

        # define datetime column
        s.define_dt_col()

        # detect inter quartile range
        s.define_iqr()

        # split train and test
        s.train_test_dt_split()

        # detect seasonality
        s.detect_daily_seasonality()
        s.detect_weekly_seasonality()

        # hyperparameter tuning
        s.get_optimal_model()

        # store model to Snowflake stage
        s.store_model()

        # return metadata for logging purpose
        return s.build_metadata()
    
    def build_metadata(s):
        s.end_time = datetime.datetime.now()
        # get training time in seconds
        s.train_time = s.end_time - s.start_time
        # return a dataframe with metadata
        return pd.DataFrame([{ 
                                'category': s._category,
                                'RMAE': s.coalesce(s.RMAE_score),
                                'MAPE': s.coalesce(s.MAPE_score),
                                'params': s.coalesce(s.best_params),
                                'holiday': '' if s._holiday is None else s._holiday,
                                'training_time_second': s.coalesce(s.train_time),
                                'max_workers': s.coalesce(s._max_workers),
                                'memory': s.coalesce(psutil.virtual_memory().total),
                                'train_at': s.coalesce(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            }])
    
    def coalesce(s, value, default = '', convert_to_str = True):
        return default if value is None else str(value) if convert_to_str else value
    
    def define_iqr(s):
        _1st_quantile = s.df[s._label_fld].quantile(s._floor_quantile)
        _3rd_quantile = s.df[s._label_fld].quantile(s._cap_quantile)
        # measure interquartile range, exclude outliers
        iqr = _3rd_quantile - _1st_quantile
        floor = _1st_quantile - 1.5 * iqr if _1st_quantile - 1.5 * iqr > 0 else 0
        s.df['cap'] = _3rd_quantile + 1.5 * iqr
        s.df['floor'] = floor

    def define_dt_col(s):
        s.dt_col = s.df.select_dtypes(include=['datetime64']).columns[0]
        s.df.sort_values(s.dt_col, inplace=True)
        s.df.rename(columns={s.dt_col: 'ds', s._label_fld: 'y'}, inplace=True)

    def train_test_dt_split(s):
        s.train, s.test = train_test_split(s.df, test_size=s._tt_split, shuffle=False)
        s.train.rename(columns={s._label_fld: 'y'}, inplace=True)
        s.test.rename(columns={s._label_fld: 'y'}, inplace=True)

    def get_optimal_model(s):
        s.train_test_dt_split()
        params_grid = {'seasonality_mode': ['multiplicative','additive'],
                        'changepoint_prior_scale': s._CHANGEPOINT_PRIOR_SCALE,
                        'growth': ['linear', 'logistic'],
                        'changepoint_range': s._CHANGEPOINT_RANGE,
                        'seasonality_prior_scale': s._SEASONALITY_PRIOR_SCALE,
                        'daily_seasonality': s.daily_seasonality,
                        'weekly_seasonality': s.weekly_seasonality,
                        'yearly_seasonality': [True, False],
                        'uncertainty_samples': [None]
                        }
    
        grid = ParameterGrid(params_grid)

        tasks = ([(s.prophet_model(p), p, s.train, s.test, f'{idx} / {len(grid)}') 
                    for idx, p in enumerate(grid) if idx < s._trials or s._trials == -1])

        with ThreadPoolExecutor(max_workers=s._max_workers) as executor:
            results = list(executor.map(s.prophet_helper, tasks))
            model_parameters = pd.DataFrame(results)

        s.best_params = (model_parameters.sort_values(by=['RMAE'])
                                            .reset_index(drop=True)['PARAMS'][0])
        s.best_params['uncertainty_samples'] = True
        s.RMAE_score = (model_parameters.sort_values(by=['RMAE'])['RMAE'][0])
        s.MAPE_score = (model_parameters.sort_values(by=['RMAE'])['MAPE'][0])
        s.best_model = s.prophet_model(s.best_params)

    def store_model(s):
        model = s.best_model
        # combine s.train and s.test before fitting
        s.train = pd.concat([s.train, s.test])

        model.fit(s.train)
        model_name = 'INT_SALES_DS_PREDICT'
        if not s._is_local:
            joblib.dump(model, f'/tmp/{s._category}__{model_name}')
            s.session.sql(f'CREATE STAGE IF NOT EXISTS {model_name}')
            s.session.file.put(f'/tmp/{s._category}__{model_name}', f'@{model_name}', auto_compress=False, overwrite=True)
        else:
            joblib.dump(model, "./" + model_name)

    def prophet_model(s, p):
        m = Prophet(**p)
        m.add_country_holidays(country_name=s._holiday) if s._holiday else None
        return m

    def prophet_helper(s, _args):
        return s.get_prophet_rmae_score(_args[0], _args[1], _args[2], _args[3], _args[4])

    def get_prophet_rmae_score(s, m, p, train, valid, idx,
                        _RMAE_COL = 'RMAE', _MAPE_COL = 'MAPE', _params_COL = 'PARAMS'):
        try:
            m.fit(train)
            preds = m.predict(s.test)
            preds.index = s.test.index
            rmae = ((s.test.y - preds.yhat) ** 2).mean() ** 0.5
            
            # calculate MAPE
            mape = np.mean(np.abs((s.test.y - preds.yhat) / s.test.y))

            return {_RMAE_COL: rmae, _MAPE_COL: mape,  _params_COL: p}
        except Exception as e:
            print(f'Error: {e}')
            return {_RMAE_COL: None, _MAPE_COL: None, _params_COL: None}

def model(dbt, session):
    # load python libraries
    dbt.config(
        packages = ["joblib", "pandas", "prophet", "scikit-learn"]
    )

    ref_df = dbt.ref("INT_SALES_DS_TRAIN").to_pandas().copy()

    # adapter: rename label column to LABEL
    ref_df.rename(columns={'WEEKLY_SALES': 'y'}, inplace=True)

    # get distinct categories list for field STORE_DEPT_PK
    categories = ref_df['STORE_DEPT_PK'].unique()
    # create df to store results, predefined columns
    results = pd.DataFrame(columns=output_cols)
    # loop through categories
    for category in categories:
        # filter df by category
        config = {
                'df': ref_df[ref_df['STORE_DEPT_PK'] == category],
                'session': session,
                'dbt': dbt,
                '_category': category
        }
        # get optimal model for category
        sf = snowpark_forecast(**config)
        metadata = sf.process()
        results = pd.concat([results, metadata], ignore_index=True)

    # get optimal model for all categories
    config = {
            'df': ref_df,
            'session': session,
            'dbt': dbt,
            '_category': 'ALL'
        }
    sf = snowpark_forecast(**config)
    metadata = sf.process()
    results = pd.concat([results, metadata], ignore_index=True)
    return results
