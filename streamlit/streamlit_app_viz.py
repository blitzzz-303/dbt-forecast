import pandas as pd
import plotly as plotly
import plotly.graph_objects as go

# given this dataframe structure, make a line chart with plotly
# use separate scale for each measure
# WEEKLY_SALES	float64
# WEEK_DATE	datetime64[ns]
# TEMPERATURE	float64
def generate_temp_line_chart(df):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df['WEEK_DATE'],y=df['WEEKLY_SALES'],
    mode='lines',
    line=dict(color="rgba(0,176,246,1)"),
    name='Sales'))
    fig.add_trace(go.Scatter(x=df['WEEK_DATE'],y=df['TEMPERATURE'],
    mode='lines',
    fill='tozeroy',
    yaxis='y2',
    line=dict(color="rgba(255, 213, 128, 0.2)"),
    name='temperature'))
    fig.update_layout(title=f'Weekly Sales and Temperature for {df["STORE_DEPT_PK"].unique()[0]}',
    xaxis_title='Date',
    yaxis=dict(title='Sales'),
    yaxis2=dict(title='Temperature',overlaying='y',side='right'))
    return fig

# given this dataframe structure, make a line chart with plotly
# convert bool to 0 and 1
# use separated scale, color for each measure
# WEEKLY_SALES	float64
# WEEK_DATE	datetime64[ns]
# ISHOLIDAY	bool

def generate_holiday_chart(df):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig.update_layout(height=400)
    fig.add_trace(go.Scatter(x=df['WEEK_DATE'],y=df['WEEKLY_SALES'],
    mode='lines',
    line=dict(color="rgba(0,176,246,1)"),
    name='Sales'))
    # add holiday as a separate area
    
    fig.add_trace(go.Scatter(x=df['WEEK_DATE'],y=df['ISHOLIDAY'],
    mode='lines',
    fill='tozeroy',
    yaxis='y2',
    line=dict(color="rgba(144, 238, 144 ,0.2)"),
    name='holiday'))
    fig.update_layout(title=f'Weekly Sales and Holiday for {df["STORE_DEPT_PK"].unique()[0]}',
    xaxis_title='Date',
    yaxis=dict(title='Sales'),
    yaxis2=dict(title='Holiday',overlaying='y',side='right'))
    return fig



def generate_predict_viz(df):
    df["WEEK_DATE"] = pd.to_datetime(df["WEEK_DATE"])

    start_date = min(df["WEEK_DATE"])
    end_date = max(df["WEEK_DATE"])
    selected_store = df["STORE_DEPT_PK"].unique()[0]

    filtered_df = df[(df["WEEK_DATE"] >= start_date) & (df["WEEK_DATE"] <= end_date) & (df["STORE_DEPT_PK"] == selected_store)]

    filtered_df_predict = filtered_df[filtered_df['WEEK_DATE'] >= '2012-01-01']
    fig = fig = plotly.tools.make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(x=filtered_df["WEEK_DATE"], y=filtered_df["WEEKLY_SALES"], name="Weekly Sales", mode="lines"))
    fig.add_trace(go.Scatter(x=filtered_df_predict["WEEK_DATE"], y=filtered_df_predict["WEEKLY_SALE_PREDICT"], name="Predicted Sales", mode="markers+lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=filtered_df_predict["WEEK_DATE"], y=filtered_df_predict["WEEKLY_SALE_PREDICT_UPPER"], mode="lines", name="Upper Range", line=dict(color="rgba(144, 238, 144 ,0.2)")))
    fig.add_trace(go.Scatter(x=filtered_df_predict["WEEK_DATE"], y=filtered_df_predict["WEEKLY_SALE_PREDICT_LOWER"], mode="lines", name="Lower Range", line=dict(color="rgba(144, 238, 144 ,0.2)"), fill="tonexty"))

    fig.update_traces(selector=dict(name="Predicted Sales"), opacity=0.8)
    fig.update_traces(selector=dict(name="Upper Range"), opacity=0.5)
    fig.update_traces(selector=dict(name="Lower Range"), opacity=0.5)

    filtered_df_predict['WEEKLY_SALE_PREDICT'] = filtered_df_predict['WEEKLY_SALE_PREDICT'].round(0)
    fig.update_layout(title=f"Weekly Sales vs Predicted Sales for {selected_store}",
                      xaxis_title="Week Date", yaxis_title="Sales", height=400)

    return fig
