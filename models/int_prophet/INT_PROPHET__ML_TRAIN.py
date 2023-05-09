import pandas as pd, numpy as np
import multiprocessing, joblib, datetime, cloudpickle
from prophet import Prophet
from sklearn.model_selection import ParameterGrid, train_test_split
from concurrent.futures import ThreadPoolExecutor
output_cols = ['category', 'MSE', 'RMAE', 'MAPE',
              'params', 'training_time_second',
              'max_workers', 'memory', 'train_at']
is_local = False
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
            'changepoint_prior_scale': [0.005, 0.05, 0.5],
            'seasonality_prior_scale': [10],
            'growth': ['linear'],
            'changepoint_range': [1],
            'seasonality_prior_scale': [10],
            'n_changepoints': [25, 50],
            'daily_seasonality': [False],
            'weekly_seasonality': [False],
            'yearly_seasonality': [True, False],
            'uncertainty_samples': [None]
        }
        self.session = session
        self.category = category
        self.MSE_score = 0
        self.RMAE_score = 0
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
                                'category': self.category,
                                'MSE': self.coalesce(self.MSE_score),
                                'RMAE': self.coalesce(self.RMAE_score),
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
        dt_col = self.df.select_dtypes(include=['datetime64']).columns[0]
        self.df.sort_values(dt_col, inplace=True)
        self.df.rename(columns={dt_col: 'ds'}, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def train_test_dt_split(self):
        self.train, self.test = train_test_split(self.df, test_size=tt_split, shuffle=False)

    def get_optimal_model(self):
        self.train_test_dt_split()
        grid = ParameterGrid(self.param_grid)
        tasks = [(p) for idx, p in enumerate(grid) if idx < trials or trials == -1]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.model_evaluation, tasks))
            model_parameters = pd.DataFrame(results)
        if model_parameters.empty:
            raise ValueError("No valid models found. Adjust the parameter grid.")

        # ignore na values, sort by RMAE score, then by MAPE score
        model_parameters = model_parameters.sort_values(by=['MSE', 'RMAE', 'MAPE']).reset_index(drop=True)
        self.best_params = model_parameters['PARAMS'][0]
        self.best_params['uncertainty_samples'] = 100
        self.RMAE_score = model_parameters['RMAE'][0]
        self.MAPE_score = model_parameters['MAPE'][0]
        self.MSE_score = model_parameters['MSE'][0]

    def store_model(self):
        self.best_model = Prophet(**self.best_params)
        self.best_model.add_country_holidays(country_name=self.holiday) if self.holiday else None
        self.train_test = pd.concat([self.train, self.test])  # Combine train and test data
        self.best_model.fit(self.train_test[['ds', 'y']])
        model_name = 'RETAIL_SALES'
        stage_name = 'HT_PROPHET_MODELS'
        if not is_local:
            with open(f'/tmp/{self.category}__{model_name}', 'wb') as f:
                cloudpickle.dump(self.best_model, f)
            self.session.sql(f'CREATE STAGE IF NOT EXISTS {stage_name}')
            self.session.file.put(
                f'/tmp/{self.category}__{model_name}',
                f'@{stage_name}',
                auto_compress=False,
                overwrite=True
            )
        else:
            joblib.dump(model, "./" + model_name)

    def model_evaluation(self, p):
        _MSE_COL, _RMAE_COL, _MAPE_COL, _params_COL = 'MSE', 'RMAE', 'MAPE', 'PARAMS'
        try:
            m = Prophet(**p)
            m.add_country_holidays(country_name=self.holiday) if self.holiday else None
            m.fit(self.train)
            preds = m.predict(self.test)
            preds.index = self.test.index
            # caculate MSE
            mse = ((self.test.y - preds.yhat) ** 2).mean()
            rmae = ((self.test.y - preds.yhat) ** 2).mean() ** 0.5
            mape = np.mean(np.abs((self.test.y - preds.yhat) / self.test.y))
            return {_MSE_COL: mse, _RMAE_COL: rmae, _MAPE_COL: mape, _params_COL: p}
        except Exception as e:
            return {_MSE_COL: None, _RMAE_COL: None, _MAPE_COL: None, _params_COL: None}

def model(dbt, session):
    dbt.config(packages=["cloudpickle", "joblib", "pandas", "prophet", "scikit-learn"])
    ref_df = dbt.ref("INT_PROPHET__SALES_TRAIN").to_pandas()
    ref_df.rename(columns={'WEEKLY_SALES': 'y'}, inplace=True)

    categories = ref_df['STORE_DEPT_PK'].unique()
    results = pd.DataFrame(columns=output_cols)

    for category in categories:
        config = {
            'df': ref_df[ref_df['STORE_DEPT_PK'] == category],
            'session': session,
            'category': category
        }
        sf = SnowparkForecast(**config)
        metadata = sf.process()
        results = pd.concat([results, metadata], ignore_index=True)
    return results
