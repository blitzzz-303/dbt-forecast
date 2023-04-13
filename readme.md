## Introduction
dbt-forecast is an experimental project aimed at providing a more user-friendly and accurate forecast experience for Snowflake. The forecast solution is implemented as a Stored Procedure within Snowflake's integrated Python environment (Snowpark).

The current version of the project supports forecasting for one category, allowing the user to forecast multiple products with a single run. In the future, this feature will be further enhanced to provide more granular analysis and improve overall accuracy.

Please note that the package is not currently compatible with the generate_schema_name feature in dbt, but this issue will be addressed in future releases.

For testing purposes, you can use the retail sales forecast dataset available on Kaggle (https://www.kaggle.com/code/aremoto/retail-sales-forecast/data?select=stores+data-set.csv). The dataset is already included in the seed.

## Prophet or ARIMA
When choosing between Prophet and ARIMA for time series forecasting, it's important to consider the complexity and required expertise of each model.

Prophet is a user-friendly and interpretable solution for time series forecasting, making it a more accessible option for those with limited knowledge in the field. However, it is optimized for time series with clear trends and seasonality and may not deliver optimal results for other types of series.

In contrast, ARIMA is a more complex model that demands advanced knowledge and expertise. While it offers greater flexibility, it may not perform well with time series that exhibit strong trends and seasonality.

One limitation of Prophet is its handling of outliers. To overcome this, the current version of the project uses +-1.5 interquartiles for the first and third quantiles to manage outliers.

## Evaluation
### Weekly
![Prophet evaluation](img/evaluation_store_1_2.png)

### Shaped to Monthly view
![Prophet evaluation](img/evaluation_store_1_2_m.png)
## Usage

1. Clone the repository

2. Build the Docker container using the following command
    
    ``docker-compose build``
3. Start the Docker container using the following command
    
    ``docker-compose run --rm sf``
    
4. (Optional) To generate the Snowpark function, after logging into the container, use the following command
    ``python scripts/sf_func_gen.py``

5. Start testing the function by running the following command
    ``dbt build``


## Obtaining Optimal Parameters

The parameters for the model are optimized using grid search, where a set of predefined values are iterated over to find the best set of parameters. The grid search is conducted by dividing the dataset into a train and test set, with a ratio of 0.67:0.33. The model is only allowed to capture 80% of the series to prevent overfitting.

For each category, the model will have its own set of optimal parameters, which are found by iterating through the following parameter grid:

```
params_grid = {
    'seasonality_mode': ['multiplicative', 'additive'],
    'changepoint_prior_scale': [0.5],
    'growth': ['linear', 'logistic'],
    'changepoint_range': [0.8],
    'seasonality_prior_scale': [10],
    'daily_seasonality': [True, False],
    'weekly_seasonality': [True, False],
    'yearly_seasonality': [True, False]
}
```

## Extra feature

### Support country holiday in prediction
In order to use this feature, please include ``COUNTRY`` as a field name from your input model. Check ``int_sds_train_country`` as example use case.

### Reuse the last best params for future runs (still testing)
To improve the performance, the function re-use the best params from previous run, instead running simulation to search for best params again (this process is very expensive).