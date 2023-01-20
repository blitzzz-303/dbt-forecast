{% macro create_stored_procedure()%}

{% set store_procedure_txt = stored_procedure_gen() %}
{{ print(store_procedure_txt) }}
{%- endmacro %}

{% macro stored_procedure_gen() -%}

CREATE OR REPLACE PROCEDURE {{target.database}}.{{target.schema}}.prophet_predict(params varchar)
returns variant
language python
runtime_version = 3.8
packages = ('snowflake-snowpark-python', 'prophet', 'scikit-learn', 'pandas', 'numpy')
handler = 'udf'
as $$


import pandas as pd
pd.options.mode.chained_assignment = None
import multiprocessing
from snowflake.snowpark import Session
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

def udf(session, params):
    configs = eval(params)
    configs['session'] = session
    
    p = batch_forecast(**configs)
    return 'success'


class batch_forecast():
    _params = None
    sesion = None
    tbl_source = None
    _max_workers = multiprocessing.cpu_count()
    _USE_LAST_params = True
    _label_fld = 'LABEL'
    _params_FLD = 'PARAM'
    _holiday_FLD = 'COUNTRY'
    def __init__(s, **kwargs):
        s.__dict__.update(**kwargs)
        s.get_params_from_previous_run()
        s.get_df_from_sf(s.tbl_source)
        s.convert_and_sort_timestamp_field()

        categories = s.df[s._cat_fld].unique() if s._cat_fld in s.df else [None]
        df_output = pd.DataFrame()
        for category in categories:
            if s._cat_fld in s.df:
                df_filter = s.df[(s.df[s._cat_fld] == category) | (category is None)]
                if category in s.previous_params:
                    df_filter[s._params_FLD] = s.previous_params[category]
                elif 'generic' in s.previous_params:
                    df_filter[s._params_FLD] = str(s.previous_params['generic'])
                df_category = df_filter
            else:
                df_category = s.df
            
            holiday = s.df[s._holiday_FLD][0] if s._holiday_FLD in s.df else None

            pf = prophet_forecast(**s.__dict__)
            pf._params_FLD = s._params_FLD
            pf.df = df_category
            pf._holiday = holiday
            df_pred = pf.forecast()
            df_output = pd.concat([df_output, df_pred])

        if s.session != None:
            return_df = s.session.create_dataframe(df_output)
            return_df.write.mode("overwrite").save_as_table(s.tbl_target)
        else:
            file_output = s._df_path + '__predict.csv'
            df_output.to_csv(file_output)
            print(df_output)
            print(df_output[s._params_FLD][0])

    def get_df_from_sf(s, table_name):
        if s.session is not None:
            try:
                s.df = s.session.table(table_name).to_pandas().copy()
            except:
                print(f'Table {table_name} does not exist')
        else:
            s.df = pd.read_csv(s._df_path)

    def convert_and_sort_timestamp_field(s):
        s.df[s._ts_fld] = s.df[s._ts_fld].astype('datetime64')
        s.df.sort_values(by=[s._cat_fld, s._ts_fld], inplace=True)
        s.df.reset_index(drop = True, inplace = True)

    def get_params_from_previous_run(s):
        if s._params is not None:
            s._params = eval(s._params)
            return
        elif s._USE_LAST_params and 'tbl_target' in s.__dict__:
            target_df = s.get_df_from_sf(s.tbl_target)
            if target_df is not None and s._params_FLD in target_df and s._cat_fld in target_df:
                s.previous_params = target_df.groupby(by=[s._cat_fld, s._params_FLD]).count().to_dict('records')
            elif target_df is not None and s._params_FLD in target_df:
                s.previous_params = {'generic': target_df[s._params_FLD].unique()}
            else:
                s.previous_params = {}
        else:
            s.previous_params = {}


class prophet_forecast:
    _label_fld = 'LABEL'
    _holiday = None
    _params = None
    df = None
    _trials = -1
    _max_workers = multiprocessing.cpu_count()
    _tt_split = 0.33
    _CHANGEPOINT_PRIOR_SCALE = [0.5]
    _CHANGEPOINT_RANGE = [0.8]
    _cap_quantile = .8
    _floor_quantile = .2

    def __init__(s, **kwargs):
        s.__dict__.update(**kwargs)

    def forecast(s):
        s.df['cap'] = s.df[s._label_fld].quantile(s._floor_quantile)
        s.df['floor'] = s.df[s._label_fld].quantile(s._cap_quantile)
        s.train, s.test, s.dt_col = s.train_test_dt_split()
        s.get_optimal_model()
        s.prophet_predict()
        return s.df_pred

    def train_test_dt_split(s):
        df = s.df.copy()
        train, test = df[df[s._label_fld].notnull()], df[df[s._label_fld].isnull()]
        dt_col = df.select_dtypes(include=['datetime64']).columns[0]
        train.rename(columns={dt_col: 'ds', s._label_fld: 'y'}, inplace=True)
        test.rename(columns={dt_col: 'ds', s._label_fld: 'y'}, inplace=True)
        return train, test, dt_col

    def prophet_predict(s):
        m = s.best_model
        m.fit(s.train)
        predictions_train = m.predict(s.train.drop(columns='y'))
        predictions_test = m.predict(s.test.drop(columns='y'))
        df_pred = pd.concat([predictions_train, predictions_test], axis=0)
        #update index for prediction df
        df_pred.index = s.df.index
        df = s.df
        df[df_pred.columns] = df_pred[df_pred.columns]
        df[s._params_FLD] = str(s.best_params)
        df[f'{s._label_fld}__PREDICT'] = df['yhat'].convert_dtypes()
        # df['RMAE'] = ((df[s._label_fld] - df.yhat) ** 2).mean() ** .5
        df['CPU_used'] = s._max_workers
        s.df_pred = df.drop(columns=['ds', 'yhat'])
    
    def get_optimal_model(s):
        try:
            if s._params_FLD in s.df:
                print(s.df[s._params_FLD][0])
                previous_params = eval(s.df[s._params_FLD][0])
                if previous_params is not None or previous_params != '':
                    previous_params = eval(previous_params)
                    s._params = previous_params
                    s.best_model = s.prophet_model(**s._params)
                    return
        except Exception:
            pass
        train, valid, _, _ = train_test_split(s.train, s.train['y'],
                                                test_size=s._tt_split, shuffle=False)
        print(valid)
        params_grid = {'seasonality_mode': ['multiplicative','additive'],
                        'changepoint_prior_scale': s._CHANGEPOINT_PRIOR_SCALE,
                        'growth': ['linear', 'logistic'],
                        'changepoint_range': s._CHANGEPOINT_RANGE,
                        'daily_seasonality': [True, False],
                        'weekly_seasonality': [True, False],
                        'yearly_seasonality': [True, False]
                        }
    
        grid = ParameterGrid(params_grid)

        tasks = ([(s.prophet_model(p), p, train, valid, f'{idx} / {len(grid)}') 
                    for idx, p in enumerate(grid) if idx < s._trials or s._trials == -1])

        with ThreadPoolExecutor(max_workers=s._max_workers) as executor:
            results = list(executor.map(s.prophet_helper, tasks))
            model_parameters = pd.DataFrame(results)

        s.best_params = (model_parameters.sort_values(by=['rmae'])
                                            .reset_index(drop=True)['params'][0])
        s.best_model = s.prophet_model(s.best_params)

    def prophet_model(s, p):
        m = Prophet(**p)
        m.add_country_holidays(country_name=s._holiday) if s._holiday else None
        return m

    def prophet_helper(s, _args):
        return s.get_prophet_rmae_score(_args[0], _args[1], _args[2], _args[3], _args[4])

    def get_prophet_rmae_score(s, m, p, train, valid, idx,
                        _RMAE_COL = 'rmae', _params_COL = 'params'):
        try:
            m.fit(train)
            preds = m.predict(valid)
            preds.index = valid.index
            print(preds.yhat)
            print(valid.y)
            # mape = ((valid.y-preds.yhat).abs() / valid.y).mean()
            rmae = ((valid.y - preds.yhat) ** 2).mean() ** .5
            print('# ', idx)
            print(p)
            print('rmae ----------- ', rmae)
            return {_RMAE_COL:rmae, _params_COL:p}
        except Exception:
            return {_RMAE_COL:s.error(), _params_COL:None}

    def error(s):
        return 9999


    $$

{%- endmacro %}