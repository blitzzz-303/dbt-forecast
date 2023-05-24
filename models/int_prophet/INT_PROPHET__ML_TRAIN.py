import multiprocessing, cloudpickle, datetime
import pandas as pd, numpy as np, snowflake.snowpark.functions as F
from prophet import Prophet
from sklearn.model_selection import ParameterGrid, train_test_split
from concurrent.futures import ThreadPoolExecutor
output_cols = ['CATEGORY', 'MSE', 'RMSE', 'MAPE',
              'params', 'training_time_second',
              'max_workers', 'memory', 'train_at']
trials = -1
max_workers = multiprocessing.cpu_count()
tt_split = 0.33

class SnowparkForecast:
    def __init__(self, df, session, category='default'):
        self.df = df.copy()
        # define datetime column & sort data
        self.staging()
        self.param_grid = {
            'seasonality_mode': ['multiplicative', 'additive'],
            'changepoint_prior_scale': [0.05, 0.5],
            'seasonality_prior_scale': [10],
            'growth': ['linear'],
            'changepoint_range': [1],
            'seasonality_prior_scale': [10],
            'n_changepoints': [25],
            'daily_seasonality': [False],
            'weekly_seasonality': [False],
            'yearly_seasonality': [True, False],
            'uncertainty_samples': [None],
            'holidays': [self.ds_holiday]
        }
        self.session = session
        self.category = category
        self.MSE_score = 0
        self.RMSE_score = 0
        self.MAPE_score = 0
        self.best_params = None
        self.best_model = None
        self.holiday = None

    def process(self):
        self.start_time = datetime.datetime.now()

        # split train and test
        self.train_test_dt_split()

        # hyperparameter tuning
        self.get_optimal_model()

        # store model to Snowflake stage
        self.store_model()

        # get training time in seconds
        self.train_time = datetime.datetime.now() - self.start_time

        # return metadata for logging purpose
        return self.build_metadata()

    def build_metadata(self):
        # return a dataframe with metadata
        return pd.DataFrame([{ 
                                'CATEGORY': self.category,
                                'MSE': self.coalesce(self.MSE_score),
                                'RMSE': self.coalesce(self.RMSE_score),
                                'MAPE': self.coalesce(self.MAPE_score),
                                'params': self.coalesce(self.best_params),
                                'holiday': '' if self.holiday is None else self.holiday,
                                'training_time_second': self.coalesce(self.train_time),
                                'max_workers': self.coalesce(max_workers),
                                'train_at': self.coalesce(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            }])

    def coalesce(s, value, default = '', convert_to_str = True):
        return default if value is None else str(value) if convert_to_str else value

    def staging(self):
        # sort data by datetime
        self.df.sort_values('ds', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # get first year value
        first_year = self.df['ds'].dt.year.min()

        # get holidays for the first year
        hol_filter = (self.df['ds'].dt.year == first_year) & (self.df['ISHOLIDAY'] == True)
        holidays = pd.to_datetime(self.df[hol_filter]['ds'].unique().tolist())

        # extrapolate holidays for the next 5 years
        for y in range(5):
            holidays = holidays.append(holidays + pd.DateOffset(years=y))

        # store holidays in a dataframe
        self.ds_holiday = pd.DataFrame({
                            'holiday': 'holiday input',
                            'ds': holidays,
                            'lower_window': -3,
                            'upper_window': 3})

    def train_test_dt_split(self):
        self.train, self.test = train_test_split(self.df, test_size=tt_split, shuffle=False)

    def get_optimal_model(self):
        self.train_test_dt_split()
        # declare model evaluation function
        grid = ParameterGrid(self.param_grid)

        # declare task list
        tasks = [(p) for idx, p in enumerate(grid) if idx < trials or trials == -1]

        # job mapping to thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.model_evaluation, tasks))
            model_parameters = pd.DataFrame(results)
        if model_parameters.empty:
            raise ValueError("No valid models found. Adjust the parameter grid.")

        # ignore na values, sort by MSE score, then by RMSE and MAPE
        # MSE for penalizing large errors
        model_parameters = model_parameters.sort_values(by=['MSE', 'RMSE', 'MAPE']).reset_index(drop=True)
        self.best_params = model_parameters['PARAMS'][0]
        self.best_params['uncertainty_samples'] = 100
        self.RMSE_score = model_parameters['RMSE'][0]
        self.MAPE_score = model_parameters['MAPE'][0]
        self.MSE_score = model_parameters['MSE'][0]

    def store_model(self):
        # from best_params, create a model and store it to Snowflake stage
        self.best_model = Prophet(**self.best_params)
        # add country holidays, if any
        self.best_model.add_country_holiwdays(country_name=self.holiday) if self.holiday else None
        self.train_test = pd.concat([self.train, self.test])  # Combine train and test data
        self.best_model.fit(self.train_test[['ds', 'y']])
        model_name = 'RETAIL_SALES'
        stage_name = 'HT_PROPHET_MODELS'
        # store model to warehouse tmp folder
        with open(f'/tmp/{self.category}__{model_name}', 'wb') as f:
            cloudpickle.dump(self.best_model, f)
        # create stage if not exists
        self.session.sql(f'CREATE STAGE IF NOT EXISTS {stage_name}')
        # put model to stage
        self.session.file.put(
            f'/tmp/{self.category}__{model_name}',
            f'@{stage_name}',
            auto_compress=False,
            overwrite=True
        )

    def model_evaluation(self, p):
        _MSE_COL, _RMSE_COL, _MAPE_COL, _params_COL = 'MSE', 'RMSE', 'MAPE', 'PARAMS'
        try:
            # set model parameters
            m = Prophet(**p)
            # add country holidays, if any
            m.add_country_holidays(country_name=self.holiday) if self.holiday else None
            # fit train data
            m.fit(self.train)
            # predict test data
            preds = m.predict(self.test)
            preds.index = self.test.index
            # caculate MSE
            mse = ((self.test.y - preds.yhat) ** 2).mean()
            rmse = ((self.test.y - preds.yhat) ** 2).mean() ** 0.5
            mape = np.mean(np.abs((self.test.y - preds.yhat) / self.test.y))
            return {_MSE_COL: mse, _RMSE_COL: rmse, _MAPE_COL: mape, _params_COL: p}
        except Exception as e:
            return {_MSE_COL: None, _RMSE_COL: None, _MAPE_COL: None, _params_COL: None}

def model(dbt, session):
    # declare python libs for training job
    dbt.config(packages=['cloudpickle', 'prophet', 'scikit-learn'],
               materialized = 'incremental', unique_key = 'CATEGORY')

    # getting dbt config variables
    stores = [s.strip() for s in dbt.config.get('stores').split(',')]
    depts = [d.strip() for d in dbt.config.get('depts').split(',')]
    
    # setup filter for store and dept
    store_dept_filter = (F.col('store').isin(stores)) & (F.col('dept').isin(depts))
    required_cols = ['STORE_DEPT_PK', 'WEEK_DATE', 'WEEKLY_SALES', 'ISHOLIDAY']

    # get data, filter and select required columns
    ref_df = (dbt.ref("INT_PROPHET__SALES_TRAIN")
                    .filter(store_dept_filter)
                    .select(required_cols)
                    .toPandas())

    # rename columns to match Prophet's requirements
    ref_df.rename(columns={'WEEK_DATE': 'ds',
                           'WEEKLY_SALES': 'y'}, inplace=True)

    # define output columns
    results = pd.DataFrame(columns=output_cols)

    for category in ref_df['STORE_DEPT_PK'].unique():
        config = {
            'df': ref_df[ref_df['STORE_DEPT_PK'] == category],
            'session': session,
            'category': category
        }
        sf = SnowparkForecast(**config)
        metadata = sf.process()
        results = pd.concat([results, metadata], ignore_index=True)
    
    # store results in a snowflake table
    return results