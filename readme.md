## Introduction
dbt-sf-propeht is an experimental project to enable better, more user friendly forecast for Snowflake.

The forecast solution is defined as a Stored Procedure, run via Snowpark (Snowflake integrated python environment).

This demo support 1 category for the forecast, which enable to forecast multiple product at one run. Thus, this enables user to have more granularity for analysis and improve overall accuracy.

For some reasons, this package is not compatiple with generate_schema_name from dbt. This should be fixed in later release.

Main data set we should use to test https://www.kaggle.com/code/aremoto/retail-sales-forecast/data?select=stores+data-set.csv (already included in seed)





## Usage

1. Clone the repo
2. Build a docker container, by using below command
    
    ``docker-compose build``
3. Start your new docker container, by using below command
    
    ``docker-compose run --rm sf``
    
4. (optional) to generate the Snowpark function, after login to your container, use below command
    ``python scripts/sf_func_gen.py``

5. start trying the function by running ``dbt build`` statement

## Extra feature

### Support country holiday in prediction
In order to use this feature, just include COUNTRY as a field name from your input model. Check ``int_sds_train_country`` as example use case.

### Reuse the last best params for future runs (still testing)
To improve the performance, the function re-use the best params from previous run, instead running simulation to search for best params again (this process is very expensive).

## Extra note
Param grid used in Snowpark (~32 simulation for each category)

    params_grid = {'seasonality_mode': ['multiplicative','additive'],
                    'changepoint_prior_scale': s._CHANGEPOINT_PRIOR_SCALE,
                    'growth': ['linear', 'logistic'],
                    'changepoint_range': s._CHANGEPOINT_RANGE,
                    'daily_seasonality': [True, False],
                    'weekly_seasonality': [True, False],
                    'yearly_seasonality': [True, False]
                    }